"""Grace - Offline voice accessibility assistant for Windows.

Main entry point. Wires together all components and runs the
main event loop: wake word → speech capture → transcription →
intent generation → tool execution → response TTS → repeat.
"""

import asyncio
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import warnings
from typing import Optional

# Disable online HuggingFace/Transformers network checks & silence PyTorch warnings
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Config field defaults are os.getenv(...) calls evaluated at import time, so
# .env must be applied before Config is imported.
from grace.env_loader import load_env
load_env()

from grace.config import Config
from grace.audio.capture import AudioCapture
from grace.audio.pump import AudioPump
from grace.audio.wake_word import WakeWordDetector
from grace.vad.detector import VadDetector
from grace.stt.whisper_stream import WhisperStreaming
from grace.llm.gemma_client import GemmaClient, RateLimitError
from grace.llm.model_swap import ModelSwapManager, LocalModelSpec
from grace.intent.prompt import get_system_prompt
from grace.intent.parser import IntentParser
from grace.automation.computer_use import ComputerUse
from grace.tools.dispatcher import Dispatcher
from grace.tts.kokoro_engine import KokoroEngine
from grace.ws_server import WsEventServer
from grace.tts.player import TTSPlayer
from grace.response.generator import ResponseGenerator
from grace.response.feedback import FeedbackSounds
from grace.util.timing import start_turn, stage
from grace.harness import get_recorder

# Configure logging. DEBUG formats an f-string per 32ms audio chunk, so it is
# opt-in via GRACE_LOG_LEVEL rather than the production default.
_LOG_LEVEL = getattr(logging, os.getenv("GRACE_LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("grace")

from grace.agent.loop import AgentLoop
from grace.intent.router import CapabilityRouter, TaskComplexity


class GraceApp:
    """Main Grace application coordinator.

    Wires all components together and runs the main event loop.
    """

    def __init__(self):
        self.config = Config()
        self._running = False
        self._llama_process: Optional[subprocess.Popen] = None
        self._audio_queue: Optional[queue.Queue] = None
        self._wake_event: Optional[asyncio.Event] = None
        self._last_spoken_text: str = ""

        # Audio
        self.capture = AudioCapture(
            sample_rate=self.config.mic_sample_rate,
            chunk_size=self.config.mic_chunk,
            channels=self.config.mic_channels,
            width=self.config.mic_width,
            device_index=self.config.mic_device_index,
        )

        # Blocking mic reads happen on the pump's thread, never on the loop.
        self.pump = AudioPump(self.capture)

        self.wake_word = WakeWordDetector(
            model_path=self.config.vosk_model_path,
            keyword=self.config.wake_word_keyword,
            threshold=self.config.wake_word_threshold,
            sample_rate=self.config.mic_sample_rate,
        )

        self.vad = VadDetector(
            threshold=self.config.whisper_vad_threshold,
            silence_duration_ms=self.config.whisper_silence_duration_ms,
        )

        # Speech-to-text
        self.whisper = WhisperStreaming(
            model_path=self.config.whisper_model_path,
            device="cuda",
            compute_type="float16",
        )

        # Model hot-swap: only needed when the planner is ALSO local, since
        # then it and UI-TARS both want the one GPU. When the planner stays on
        # Gemini, only UI-TARS ever needs the GPU and nothing has to swap.
        self._model_swap: Optional[ModelSwapManager] = None
        if self.config.model_swap_enabled:
            self._model_swap = ModelSwapManager(
                host=self.config.llama_host,
                port=self.config.llama_port,
                llama_server_exe=self.config.llama_server_exe,
                specs={
                    "planner": LocalModelSpec(
                        name="planner",
                        model_path=self.config.local_planner_model_path,
                        context_window=self.config.llama_context_window,
                        ngl=self.config.llama_ngl,
                        cache_type_k=self.config.llama_cache_type_k,
                        cache_type_v=self.config.llama_cache_type_v,
                    ),
                    "grounder": LocalModelSpec(
                        name="grounder",
                        model_path=self.config.llama_model_path,
                        mmproj_path=self.config.llama_mmproj_path,
                        context_window=self.config.llama_context_window,
                        ngl=self.config.llama_ngl,
                        cache_type_k=self.config.llama_cache_type_k,
                        cache_type_v=self.config.llama_cache_type_v,
                    ),
                },
            )

        # LLM (Cloud Gemini 3.6 Flash or local llama-server)
        api_key = self.config.gemini_api_key if self.config.use_cloud_llm else None
        self.gemma = GemmaClient(
            base_url=self.config.llama_server_url,
            api_key=api_key,
            model_name=self.config.gemini_model_name,
            before_request=(lambda: self._model_swap.swap_to("planner")) if self._model_swap else None,
        )

        # Dedicated UI-TARS Vision LLM client for AgentLoop CUA steps (local llama-server)
        self.ui_tars_client = GemmaClient(
            base_url=self.config.llama_server_url,
            api_key=None,  # Forces local llama-server for UI-TARS GUI vision
            before_request=(lambda: self._model_swap.swap_to("grounder")) if self._model_swap else None,
        ) if self.config.use_ui_tars_local else self.gemma

        # Intent
        self.intent_prompt = get_system_prompt()
        self.intent_parser = IntentParser()

        # WebSocket server (frontend)
        self.ws_server = WsEventServer(
            host=self.config.ws_host,
            port=self.config.ws_port,
            allowed_origins=self.config.ws_allowed_origins,
        )

        # Computer use
        self.computer_use = ComputerUse()

        # Dispatcher
        self.dispatcher = Dispatcher(
            computer_use=self.computer_use,
            ws_server=self.ws_server,
        )

        # Agentic loop. The planner (cloud, structural) decides what to do; the
        # grounder (UI-TARS, pixels) is only consulted when the element graph
        # cannot resolve a target, which is why llama-server starts on demand.
        self.agent_loop = AgentLoop(
            gemma=self.gemma,
            dispatcher=self.dispatcher,
            ws_server=self.ws_server,
            vision_llm=self.ui_tars_client,
            # When model_swap is active, the grounder's GemmaClient already
            # swaps the GPU to UI-TARS on every request via before_request, so
            # the old "start llama-server on first grounding call" path would
            # just fight the swap manager over the same process.
            start_grounding_backend=None if self._model_swap else self._ensure_grounding_backend,
        )

        # TTS
        self.kokoro = KokoroEngine(
            model_path=self.config.kokoro_model_path,
            voices_path=self.config.kokoro_voices_path,
            num_workers=self.config.kokoro_workers,
            device=self.config.kokoro_device,
            dtype=self.config.kokoro_dtype,
            warmup=self.config.kokoro_warmup,
            cache_size=self.config.kokoro_cache_size,
            speed=self.config.kokoro_speed,
        )
        self.dispatcher.set_kokoro_engine(self.kokoro)

        self.tts_player = TTSPlayer()

        # Response generator
        self.response_gen = ResponseGenerator(
            gemma_client=self.gemma,
            kokoro_engine=self.kokoro,
            tts_player=self.tts_player,
            ws_server=self.ws_server,
        )
        # Use a dedicated conversational prompt — NOT the intent-extraction prompt
        self.response_gen.set_system_prompt(
            "You are Grace, a warm, concise voice assistant helping a user with motor "
            "dysfunction control their Windows PC. Answer naturally in short spoken sentences. "
            "Do NOT output JSON. Do NOT use markdown, lists, or special formatting. "
            "Keep replies under 3 sentences unless the user asks for detail."
        )

    # An activation that hears nothing still holds the pill open this long,
    # counted from the wake word, so it never flashes open and shut.
    MIN_WAKE_TO_IDLE_SECONDS = 3.0

    _STOP_COMMANDS = {
        "stop", "cancel", "never mind", "nevermind", "quiet", "be quiet",
        "shut up", "pause", "halt", "stop speaking", "stop that", "abort",
    }
    _REPEAT_COMMANDS = {
        "repeat", "repeat that", "what did you say", "say that again",
        "say again", "pardon", "what was that", "can you repeat that",
    }

    async def _speak_response(self, text: str) -> None:
        """Speak response text and record it for voice repeat requests."""
        if not text or not text.strip():
            return
        self._last_spoken_text = text.strip()
        await self.response_gen.generate_and_speak_with_text(self._last_spoken_text)

    def _start_llama_server(self) -> None:
        """Start llama-server as a subprocess with UI-TARS 1.5-7B on GPU."""
        cfg = self.config
        if not os.path.exists(cfg.llama_server_exe):
            raise FileNotFoundError(
                f"llama-server executable not found at: {cfg.llama_server_exe}"
            )
        args = [
            cfg.llama_server_exe,
            "-m", cfg.llama_model_path,
            "-ngl", str(cfg.llama_ngl),
            "-c", str(cfg.llama_context_window),
            "--no-mmap",
            "--mlock",
            "--cache-type-k", cfg.llama_cache_type_k,
            "--cache-type-v", cfg.llama_cache_type_v,
            "--port", str(cfg.llama_port),
            "--host", cfg.llama_host,
        ]
        if cfg.llama_mmproj_path and os.path.exists(cfg.llama_mmproj_path):
            log.info(f"UI-TARS Vision Projector detected: {cfg.llama_mmproj_path}")
            args.extend(["--mmproj", cfg.llama_mmproj_path])
        else:
            log.warning("No mmproj file detected! UI-TARS will run without vision features until mmproj is provided.")

        server_dir = os.path.dirname(cfg.llama_server_exe)
        log.info(f"Starting llama-server from {server_dir}...")
        log.debug(f"Args: {' '.join(args)}")
        self._llama_process = subprocess.Popen(
            args,
            cwd=server_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _stop_llama_server(self) -> None:
        """Stop the llama-server subprocess."""
        if self._llama_process is not None:
            pid = self._llama_process.pid
            log.info(f"Stopping llama-server PID {pid}...")
            try:
                self._llama_process.terminate()
                try:
                    self._llama_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._llama_process.kill()
                    self._llama_process.wait(timeout=3)
            except Exception as e:
                log.warning(f"Error terminating llama-server process: {e}")
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
            except Exception:
                pass
            self._llama_process = None

    async def _ensure_grounding_backend(self) -> bool:
        """Start llama-server the first time UI-TARS grounding is actually needed."""
        if self._llama_process is not None:
            return True
        if not self.config.use_ui_tars_local:
            return True

        log.info("UI-TARS grounding requested; starting llama-server now...")
        try:
            self._start_llama_server()
        except Exception as e:
            log.error(f"Could not start llama-server for grounding: {e}")
            return False

        if await self._wait_for_llama_server(timeout=120):
            return True

        log.warning("llama-server did not become healthy; grounding will use the cloud model.")
        self.ui_tars_client = self.gemma
        return False

    async def _wait_for_llama_server(self, timeout: int = 120) -> bool:
        """Poll /health until the server responds, then verify model is loaded."""
        import aiohttp

        url = self.config.llama_server_url
        log.info(f"Waiting for llama-server at {url}...")
        deadline = asyncio.get_event_loop().time() + timeout
        async with aiohttp.ClientSession() as session:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status == 200:
                            log.info("llama-server health check passed, waiting for model to load...")
                            if await self._check_model_ready(session, url):
                                log.info("llama-server model is ready.")
                                return True
                except Exception:
                    pass
                await asyncio.sleep(2)
        log.error(f"llama-server did not become healthy within {timeout}s")
        return False

    async def _check_model_ready(self, session, url: str) -> bool:
        """Send a tiny completion to verify the model is actually loaded."""
        import aiohttp

        try:
            payload = {
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": False,
            }
            async with session.post(
                f"{url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    log.info(f"Model ready (test response: '{content.strip()[:50]}')")
                    return True
                log.debug(f"Model not ready yet (status={resp.status})")
                return False
        except asyncio.TimeoutError:
            log.debug("Model not ready yet (test request timed out)")
            return False
        except Exception as e:
            log.debug(f"Model not ready yet ({e})")
            return False

    async def start(self):
        """Start Grace and enter the main event loop."""
        self._running = True
        log.info("=" * 60)
        log.info("Grace v0.0.1 - Offline Voice Accessibility Assistant")
        log.info("=" * 60)

        # A tape is only useful if you know what produced it. Written before
        # any work happens so a session that crashes at startup still leaves an
        # identifiable, if short, tape behind.
        recorder = get_recorder()
        if recorder is not None:
            log.info(f"Session recording ACTIVE: {recorder.dir}")
            recorder.write_meta(self.config)

        # llama-server only has to be up at boot when it *is* the main LLM.
        # When it is only the grounder, starting it here cost up to 120s of
        # startup for a model most requests never touch - it now starts on the
        # first grounding call instead.
        if self._model_swap is not None:
            log.info("Model-swap mode ACTIVE: loading the local planner model onto the GPU...")
            if not await self._model_swap.swap_to("planner"):
                log.error("Failed to start the local planner model. Exiting.")
                await self.shutdown()
                return
        elif not self.config.use_cloud_llm:
            log.info("Initializing local llama-server as the primary LLM...")
            self._start_llama_server()
            if not await self._wait_for_llama_server(timeout=120):
                log.error("Failed to start llama-server. Exiting.")
                await self.shutdown()
                return
        elif self.config.use_ui_tars_local:
            log.info("Cloud LLM mode ACTIVE. UI-TARS grounding will start on first use.")
        else:
            log.info("Cloud LLM mode ACTIVE: Using Google Gemini Flash API.")

        # List audio devices
        devices = self.capture.list_devices()
        log.info(f"Found {len(devices)} audio device(s)")
        for dev in devices[:5]:  # Show first 5
            log.info(f"  [{dev['index']}] {dev['name']}")
        if len(devices) > 5:
            log.info(f"  ... and {len(devices) - 5} more")

        # Start computer use backend
        log.info("Starting computer use backend...")
        self.computer_use.start()

        # Start WebSocket server (frontend)
        log.info(f"Starting WebSocket server on ws://{self.config.ws_host}:{self.config.ws_port}...")
        await self.ws_server.start()
        self._wake_event = asyncio.Event()
        self.ws_server.set_on_wake(lambda: self._wake_event.set())

        # Initialize Kokoro
        log.info("Initializing Kokoro TTS engine...")
        self.kokoro.initialize()

        # Start audio capture
        log.info("Starting microphone...")
        self.capture.start()
        self.pump.start(asyncio.get_running_loop())

        # Start wake word detector with audio queue
        self._audio_queue = queue.Queue(maxsize=100)
        self.wake_word.start(audio_queue=self._audio_queue)

        # Warm the slow, lazily-initialised components in the background so the
        # first real request doesn't pay for them.
        self._start_background_warmup()

        log.info("Microphone active. Listening for 'Grace'...")

        # Enter main loop
        try:
            while self._running:
                # Feed audio chunk to wake word detector
                chunk = await self.pump.get(timeout=0.5)
                if chunk is None:
                    continue
                if self._audio_queue is not None:
                    try:
                        self._audio_queue.put(chunk, block=False)
                    except queue.Full:
                        pass

                if self.wake_word.detected:
                    log.info("Wake word detected!")
                    self.wake_word.reset()
                    await self._run_activation_turn()
                elif self._wake_event and self._wake_event.is_set():
                    log.info("Wake command received from frontend!")
                    self._wake_event.clear()
                    await self._run_activation_turn()
                else:
                    await asyncio.sleep(0)
        except KeyboardInterrupt:
            log.info("Shutdown requested.")
        finally:
            await self.shutdown()

    # A bare yes/no word. Alone it is a complete answer ("yes"); it is also a
    # valid lead-in before a short stock phrase ("yes go ahead", "no cancel
    # that") - natural speech rarely answers with the bare word alone.
    _YES_WORDS = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "affirmative", "correct"}
    _NO_WORDS = {"no", "nope", "negative"}
    # Stock phrases meaning "proceed" / "stop", with or without one of the
    # words above in front. Deliberately a closed, curated list rather than
    # "any text after yes/no": "yes" plus an unrelated new request ("okay open
    # spotify") must not be read as approval of the parked action.
    _PROCEED_PHRASES = {"go ahead", "go on", "go for it", "do it", "confirm", "please do"}
    _STOP_PHRASES = {
        "dont", "dont do that", "do not do that", "cancel", "cancel that",
        "stop", "stop that", "never mind", "nevermind", "abort",
    }
    # Any of these inside a would-be "yes" tail rules it out - belt-and-
    # suspenders for anything added to _PROCEED_PHRASES later.
    _NEGATION_WORDS = ("not", "no", "dont", "never")
    # Filler stripped from the ends before matching, so "um yes please" and
    # "grace yes" still count as plain answers.
    _CONFIRMATION_LEADING_FILLER = ("grace", "um", "uh", "umm", "uhh")
    _CONFIRMATION_TRAILING_FILLER = ("please",)

    @classmethod
    def _confirmation_answer(cls, transcript: str) -> Optional[bool]:
        """Read a yes/no out of an utterance. None means it wasn't a plain answer.

        Strict on purpose: the previous version matched a yes/no word *anywhere*
        in the sentence, so "I am not sure" and "okay open spotify" (an
        unrelated new request) both read as approval. Now the utterance, once
        lowercased and stripped of punctuation and filler, must reduce to a
        bare yes/no or one of a small set of stock confirmation phrases -
        optionally preceded by that yes/no ("yes go ahead", "no don't do
        that") - rather than merely containing one somewhere. Anything else
        returns None, which already cancels the pending action and treats the
        utterance as a new request.
        """
        text = re.sub(r"[^\w\s]", "", (transcript or "").lower()).strip()
        if not text:
            return None

        words = text.split()
        while words and words[0] in cls._CONFIRMATION_LEADING_FILLER:
            words.pop(0)
        while words and words[-1] in cls._CONFIRMATION_TRAILING_FILLER:
            words.pop()
        if not words:
            return None

        if words[0] in cls._YES_WORDS:
            tail_words = words[1:]
        elif words[0] in cls._NO_WORDS:
            tail_words = words[1:]
            tail = " ".join(tail_words)
            if not tail or tail in cls._STOP_PHRASES:
                return False
            return None  # "no" followed by something else isn't a plain answer
        else:
            tail_words = words

        tail = " ".join(tail_words)
        if not tail and words[0] in cls._YES_WORDS:
            return True
        if tail in cls._PROCEED_PHRASES and not (set(tail_words) & set(cls._NEGATION_WORDS)):
            return True
        if tail in cls._STOP_PHRASES:
            return False
        return None

    @staticmethod
    def _agent_response_text(agent_res: dict) -> Optional[str]:
        """Pick what to say for an agent result, whatever way it ended.

        A safety pause and a spent planning budget both return without a
        final_response; saying "Goal executed." for either would be a lie.
        """
        status = agent_res.get("status")
        if status == "safety_confirmation_required":
            return agent_res.get("confirmation_prompt")
        return agent_res.get("final_response") or "Goal executed."

    async def _resolve_pending_confirmation(self, transcript: str) -> bool:
        """Apply a yes/no to a parked action. Returns True if it was handled.

        Before this, SafetyGuard would park the step and ask the question, and
        the user's "yes" was then treated as a brand-new request - so a
        confirmed action simply never ran.
        """
        answer = self._confirmation_answer(transcript)
        if answer is None:
            log.info("Pending confirmation was not answered; treating as a new request.")
            self.agent_loop.cancel_pending()
            return False

        log.info(f"Resuming parked action with approval={answer}")
        result = await self.agent_loop.resume_pending(approved=answer)
        if result.get("status") == "safety_confirmation_required":
            # The approved step led straight into another one that also needs
            # confirming (e.g. delete this, then delete that too). Saying
            # "Done." here would tell the user their real action already
            # happened and drop the new pending step on the floor unasked.
            response_text = self._agent_response_text(result)
        else:
            response_text = result.get("final_response") or (
                "Done." if answer else "Alright, I won't do that."
            )
        await self.response_gen.generate_and_speak_with_text(response_text)
        return True

    async def _park_if_confirmation_required(self, intent, result: dict) -> bool:
        """Ask the question a refused dispatch came back with. True if parked.

        The fast path reaches `delete_file`, `close_app` and `lock_computer`
        without going anywhere near `AgentLoop`, so before the guard moved to
        the dispatcher those three ran unconfirmed whenever the router judged
        the request simple - which, being one-tool requests, is always.
        """
        if result.get("status") != "confirmation_required":
            return False

        prompt = result.get("confirmation_prompt") or "Should I go ahead?"
        log.warning(f"Fast-path '{intent.tool}' parked for confirmation: {prompt}")
        self.agent_loop.park_intent(intent, prompt)
        await self.response_gen.generate_and_speak_with_text(prompt)
        return True

    def _start_background_warmup(self) -> None:
        """Pay lazy initialisation costs up front, off the event loop.

        Whisper loads its model on the first transcribe and AppIndexer globs
        Program Files on the first "open X" - both of which used to land in the
        middle of a user's first request.
        """

        def _warm() -> None:
            try:
                self.whisper.warmup()
            except Exception as e:
                log.warning(f"Whisper warmup failed (will initialise on first use): {e}")
            try:
                from grace.automation.app_indexer import AppIndexer

                self.dispatcher.set_app_indexer(AppIndexer())
            except Exception as e:
                log.warning(f"App index warmup failed (will build on first use): {e}")

        threading.Thread(target=_warm, name="grace-warmup", daemon=True).start()

    async def _run_activation_turn(self) -> None:
        """Run one activation turn without letting it take Grace down with it.

        The main loop used to catch only KeyboardInterrupt, so any exception
        escaping a turn - a bad transcript, a dispatcher bug, an LLM client
        raising something nobody expected - shut Grace down entirely. A user
        who navigates by voice has no other way to restart it.
        """
        try:
            await self._handle_activation()
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as e:
            log.error(f"Activation turn crashed: {e}", exc_info=True)
            FeedbackSounds.play_error()
            # Best-effort: _handle_activation already resumes on most exit
            # paths, but this catches whatever crashed outside that guard.
            self.wake_word.resume()
            await self.ws_server.emit({"type": "Idle"})

    async def _handle_activation(self):
        """Time one activation cycle end to end, then hand off to the follow-up window.

        The follow-up window runs outside this trace so each of its turns can be
        measured on its own rather than inflating the wake-word turn.
        """
        turn_result = False
        with start_turn("activation") as trace:
            try:
                turn_result = await self._handle_activation_turn(trace)
            finally:
                log.info(trace.summary())
                try:
                    await self.ws_server.emit({"type": "TurnTrace", "trace": trace.to_dict()})
                except Exception as e:
                    log.debug(f"Failed to emit TurnTrace: {e}")
                # The follow-up window pauses/resumes the detector itself once it
                # actually starts. Every other exit from this turn - an aborted
                # False, or an exception below - must resume it here instead, or
                # "Grace" is never heard again.
                if turn_result is False:
                    self.wake_word.resume()

        if turn_result is not False:
            await self._run_followup_window(timeout_seconds=self.config.followup_timeout_seconds)

    async def _handle_activation_turn(self, trace) -> bool:
        """Handle a wake word activation cycle. Returns False if aborted/silence, True on turn success."""
        # 1. Play activation chime
        with trace.stage("chime"):
            FeedbackSounds.play_chime()
        await self.ws_server.emit({"type": "WakeWordDetected"})

        log.info("Listening...")
        await self.ws_server.emit({"type": "ListeningStarted"})

        # Temporarily disable wake word detector to avoid re-triggering on user speech
        self.wake_word.pause()

        # 2. Reset and listen for speech
        self.vad.reset()
        self.whisper.reset_buffer()

        audio_buffer = bytearray()
        # Anything captured while the previous turn was still working is stale.
        self.pump.drain()

        # Listen for speech with initial silence timeout (default 6s) and maximum timeout (30s)
        import time
        listen_start = time.time()
        max_listen_seconds = 30
        initial_timeout = self.config.initial_listen_timeout_seconds
        has_speech_started = False

        with trace.stage("listen") as listen_stage:
            while self._running:
                try:
                    chunk = await self.pump.get(timeout=1.0)
                    now = time.time()
                    if chunk is None:
                        if not has_speech_started and (now - listen_start > initial_timeout):
                            log.info(f"Initial listen timeout reached ({initial_timeout}s without speech).")
                            break
                        if now - listen_start > max_listen_seconds:
                            log.info("Listen timeout reached.")
                            break
                        continue
                    audio_buffer.extend(chunk)
                    if log.isEnabledFor(logging.DEBUG):
                        log.debug(f"Audio chunk: {len(chunk)} bytes, RMS={self.capture.get_rms(chunk):.1f}")
                    if self.capture.get_rms(chunk) > self.config.whisper_vad_threshold * 32767:
                        has_speech_started = True
                    if self.vad.process_chunk(chunk, self.capture):
                        log.info(f"VAD turn-end detected after {len(audio_buffer)} bytes")
                        break
                    if not has_speech_started and (now - listen_start > initial_timeout):
                        log.info(f"Initial listen timeout reached ({initial_timeout}s without speech).")
                        break
                    if now - listen_start > max_listen_seconds:
                        log.info("Listen timeout reached.")
                        break
                except Exception as e:
                    log.error(f"Audio error: {e}")
                    break
            listen_stage.detail(f"{len(audio_buffer)} bytes")

        # 3. Transcribe. GPU inference is seconds of blocking work, so it goes
        # to a worker thread rather than stalling the WebSocket server.
        # A transcription failure ends the turn; it must never end the process.
        try:
            async with trace.stage("whisper"):
                self.whisper.add_buffer(audio_buffer)
                transcript = await asyncio.to_thread(self.whisper.transcribe)
        except Exception as e:
            log.error(f"Transcription failed: {e}", exc_info=True)
            self.whisper.reset_buffer()
            FeedbackSounds.play_error()
            await self.ws_server.emit({"type": "ConversationFinished"})
            await self.ws_server.emit({"type": "Idle"})
            return False

        if not transcript.strip():
            log.info("No speech detected.")
            remaining = self.MIN_WAKE_TO_IDLE_SECONDS - (time.time() - listen_start)
            if remaining > 0:
                await asyncio.sleep(remaining)
            FeedbackSounds.play_cancel()
            await self.ws_server.emit({"type": "ConversationFinished"})
            await self.ws_server.emit({"type": "Idle"})
            return False

        log.info(f"Transcribed: '{transcript}'")
        await self.ws_server.emit({"type": "FinalTranscript", "text": transcript})
        await self.ws_server.emit({"type": "ListeningStopped"})

        clean_transcript = re.sub(r"[^\w\s]", "", transcript.strip().lower()).strip()

        # Immediate voice stop reflex
        if clean_transcript in self._STOP_COMMANDS or clean_transcript.startswith("stop ") or clean_transcript == "stop":
            log.info(f"Immediate stop command detected: '{transcript}'")
            self.response_gen.stop_speaking()
            if self.agent_loop.has_pending_confirmation:
                self.agent_loop.cancel_pending()
            FeedbackSounds.play_cancel()
            await self.ws_server.emit({"type": "ConversationFinished"})
            await self.ws_server.emit({"type": "Idle"})
            return False

        # Immediate voice replay reflex
        if clean_transcript in self._REPEAT_COMMANDS:
            log.info(f"Immediate repeat command detected: '{transcript}'")
            if self._last_spoken_text:
                await self._speak_response(self._last_spoken_text)
            else:
                await self._speak_response("I haven't said anything yet.")
            await self.ws_server.emit({"type": "ConversationFinished"})
            return True

        # 3b. If a step is parked waiting for a yes/no, this utterance answers it.
        if self.agent_loop.has_pending_confirmation:
            if await self._resolve_pending_confirmation(transcript):
                await self.ws_server.emit({"type": "ConversationFinished"})
                return True

        # 4. Generate intent
        await self.ws_server.emit({"type": "UnderstandingStarted", "label": "Understanding request\u2026"})
        log.info("Calling LLM for intent generation...")
        try:
            async with trace.stage("intent_llm"):
                intent_json = await self.gemma.generate_intent(
                    user_message=transcript,
                    system_prompt=self.intent_prompt,
                )
        except RateLimitError as e:
            log.error(f"Intent generation rate limited: {e}")
            await self.ws_server.emit({"type": "UnderstandingFinished"})
            await self._speak_response(
                "I've hit my request limit for now. Please try again in a moment."
            )
            await self.ws_server.emit({"type": "ConversationFinished"})
            return True

        intent = None
        if intent_json:
            log.info(f"LLM output: {intent_json[:200]}")
            try:
                intent = self.intent_parser.parse(intent_json)
                log.info(f"Intent: {intent}")
            except Exception as e:
                log.error(f"Intent parse failed: {e}")
        else:
            log.warning("No single-turn intent JSON generated, evaluating CapabilityRouter for agentic execution...")

        await self.ws_server.emit({"type": "UnderstandingFinished"})

        # 5. Route & Execute (Dual-Path Architecture)
        complexity = CapabilityRouter.classify(transcript, intent)
        trace.mark_event(f"route_{complexity.value}")

        if complexity == TaskComplexity.AGENTIC_GOAL:
            log.info(f"Routing goal '{transcript}' to Agentic ReAct Loop...")
            if intent and intent.tool in ("open_app", "cua_launch"):
                log.info(f"Pre-executing setup intent before AgentLoop: {intent.tool}({intent.params})")
                async with trace.stage("pre_exec"):
                    await self.dispatcher.execute(intent)
                    await asyncio.sleep(2.0)

            async with trace.stage("agent_loop") as agent_stage:
                agent_res = await self.agent_loop.run(user_goal=transcript)
                agent_stage.detail(f"{len(agent_res.get('steps') or [])} steps")

            log.info(f"AgentLoop Result: {agent_res}")

            response_text = self._agent_response_text(agent_res)
            if response_text:
                async with trace.stage("speak"):
                    await self._speak_response(response_text)
        elif (intent is not None and intent.is_conversation) or complexity == TaskComplexity.CONVERSATION:
            # Use the response from intent generation directly (already generated by the intent LLM call)
            response_text = intent.response if intent and intent.response else "I'm here to help! What can I do for you today?"
            log.info(f"Speaking conversational response: '{response_text[:100]}...'")
            async with trace.stage("speak"):
                await self._speak_response(response_text)
        else:
            log.info(f"Fast-path executing tool '{intent.tool}' with params={intent.params}")
            async with trace.stage("dispatch"):
                result = await self.dispatcher.execute(intent)
            log.info(f"Result: {result}")

            if await self._park_if_confirmation_required(intent, result):
                await self.ws_server.emit({"type": "ConversationFinished"})
                return True

            response_text = None
            if intent.is_conversation and intent.response:
                response_text = intent.response
            elif result.get("text"):
                response_text = result["text"]

            if response_text:
                log.info(f"Speaking tool result: '{response_text[:100]}...' ({len(response_text)} chars)")
                async with trace.stage("speak"):
                    await self._speak_response(response_text)

        await self.ws_server.emit({"type": "ConversationFinished"})
        return True

    async def _run_followup_window(self, timeout_seconds: int = 10):
        """Keep listening without a wake word, restarting the window after each turn.

        Handling a follow-up used to re-enter this method recursively, so a long
        back-and-forth grew the Python stack one frame per exchange until it
        overflowed. It is now an outer loop.
        """
        import time

        log.info(f"Entering active {timeout_seconds}s follow-up listening window...")
        self.wake_word.pause()
        FeedbackSounds.play_listening()

        while self._running:
            await self.ws_server.emit({"type": "FollowupListeningStarted", "timeout": timeout_seconds})
            self.vad.reset()
            self.whisper.reset_buffer()
            self.pump.drain()

            start_time = time.time()
            audio_buffer = bytearray()
            handled = False

            while self._running and (time.time() - start_time < timeout_seconds):
                try:
                    chunk = await self.pump.get(timeout=0.5)
                    if chunk is None:
                        continue
                    audio_buffer.extend(chunk)

                    if not self.vad.process_chunk(chunk, self.capture):
                        continue

                    log.info(f"Follow-up VAD natural speech boundary detected after {len(audio_buffer)} bytes")
                    raw_pcm = bytes(audio_buffer)
                    transcript = await asyncio.to_thread(self.whisper.transcribe_bytes, raw_pcm)

                    if transcript.strip():
                        handled = await self._handle_followup_transcript(transcript)
                        if handled:
                            break

                    audio_buffer.clear()
                    self.vad.reset()
                except Exception as e:
                    log.error(f"Error in follow-up listening window: {e}")
                    handled = False
                    break

            if not handled:
                break

        log.info("Follow-up listening window timed out. Returning to idle.")
        if self.agent_loop.has_pending_confirmation:
            # A parked delete or app-close that nobody answered must not still
            # be approvable once we're back to idle and a new "yes" could be
            # about something else entirely.
            log.info("Cancelling a confirmation that timed out unanswered.")
            self.agent_loop.cancel_pending()
        self.wake_word.resume()
        self._flush_mic_buffer()
        await self.ws_server.emit({"type": "Idle"})

    async def _handle_followup_transcript(self, transcript: str) -> bool:
        """Act on one follow-up utterance. Returns True if the window should restart."""
        log.info(f"Follow-up transcribed: '{transcript}'")
        await self.ws_server.emit({"type": "FinalTranscript", "text": transcript})

        # Cancel any ongoing TTS output if user barged in
        self.response_gen.stop_speaking()

        clean_transcript = re.sub(r"[^\w\s]", "", transcript.strip().lower()).strip()

        # Immediate voice stop reflex in follow-up window
        if clean_transcript in self._STOP_COMMANDS or clean_transcript.startswith("stop ") or clean_transcript == "stop":
            log.info(f"Follow-up stop command detected: '{transcript}'")
            self.response_gen.stop_speaking()
            if self.agent_loop.has_pending_confirmation:
                self.agent_loop.cancel_pending()
            FeedbackSounds.play_cancel()
            return False

        # Immediate voice repeat reflex in follow-up window
        if clean_transcript in self._REPEAT_COMMANDS:
            log.info(f"Follow-up repeat command detected: '{transcript}'")
            if self._last_spoken_text:
                await self._speak_response(self._last_spoken_text)
            else:
                await self._speak_response("I haven't said anything yet.")
            return True

        if self.agent_loop.has_pending_confirmation:
            if await self._resolve_pending_confirmation(transcript):
                return True

        try:
            async with stage("intent_llm"):
                intent_json = await self.gemma.generate_intent(
                    user_message=transcript,
                    system_prompt=self.intent_prompt,
                )
        except RateLimitError as e:
            log.error(f"Follow-up intent generation rate limited: {e}")
            await self._speak_response(
                "I've hit my request limit for now. Please try again in a moment."
            )
            return True

        intent = None
        if intent_json:
            try:
                intent = self.intent_parser.parse(intent_json)
            except Exception as e:
                log.debug(f"Follow-up intent parse failed: {e}")

        complexity = CapabilityRouter.classify(transcript, intent)
        if complexity == TaskComplexity.AGENTIC_GOAL:
            if intent and intent.tool in ("open_app", "cua_launch"):
                log.info(f"Pre-executing setup intent before AgentLoop: {intent.tool}({intent.params})")
                await self.dispatcher.execute(intent)
                await asyncio.sleep(2.0)

            agent_res = await self.agent_loop.run(user_goal=transcript)
            response_text = self._agent_response_text(agent_res)
            if response_text:
                await self._speak_response(response_text)
            return True

        if intent and not intent.is_conversation:
            result = await self.dispatcher.execute(intent)
            if await self._park_if_confirmation_required(intent, result):
                return True
            response_text = result.get("text")
            if response_text:
                await self._speak_response(response_text)
            return True

        if intent and intent.response:
            await self._speak_response(intent.response)
            return True

        return False

    def _flush_mic_buffer(self) -> None:
        """Drain microphone queue and Vosk queue to prevent audio feedback / stale wake word detection."""
        if self._audio_queue:
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except Exception:
                    break
        self.pump.drain()
        self.wake_word.reset()

    async def shutdown(self):
        """Gracefully shut down all components."""
        self._running = False
        log.info("Shutting down Grace...")

        log.info("Shutdown: closing LLM client...")
        await self.gemma.close()

        log.info("Shutdown: stopping computer use...")
        self.computer_use.stop()

        log.info("Shutdown: stopping WebSocket server...")
        await self.ws_server.stop()

        log.info("Shutdown: stopping audio pump...")
        self.pump.stop()

        log.info("Shutdown: stopping audio capture...")
        self.capture.stop()
        self.capture.close()

        log.info("Shutdown: stopping wake word detector...")
        self.wake_word.stop()

        log.info("Shutdown: stopping TTS player...")
        self.tts_player.stop()
        self.tts_player.close()

        log.info("Shutdown: stopping Kokoro workers...")
        self.kokoro.shutdown()

        log.info("Shutdown: stopping llama-server...")
        if self._model_swap is not None:
            await self._model_swap.shutdown()
        else:
            self._stop_llama_server()

        # Last, so anything the shutdown path itself emits still lands on the tape.
        recorder = get_recorder()
        if recorder is not None:
            log.info(f"Shutdown: closing session tape {recorder.dir}")
            recorder.close()

        log.info("Grace stopped.")


def main():
    """Entry point for Grace."""
    app = GraceApp()
    try:
        asyncio.run(app.start())
    except KeyboardInterrupt:
        log.info("Interrupted.")


if __name__ == "__main__":
    main()
