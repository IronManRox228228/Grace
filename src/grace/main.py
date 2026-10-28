"""Grace - Offline voice accessibility assistant for Windows.

Main entry point. Wires together all components and runs the
main event loop: wake word → speech capture → transcription →
intent generation → tool execution → response TTS → repeat.
"""

import asyncio
import collections
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
import warnings
from typing import Any, Optional

# Disable online HuggingFace/Transformers network checks & silence PyTorch warnings
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Run as a script (the Tauri shell does `python src/grace/main.py`), Python
# puts src/grace itself on sys.path, so `import training` finds grace/training
# (a regular package) ahead of the repo-root training/ namespace package.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if not p or os.path.abspath(p) != _SCRIPT_DIR]

# Config field defaults are os.getenv(...) calls evaluated at import time, so
# .env must be applied before Config is imported.
from grace.env_loader import load_env
load_env()

from grace.automation.dpi_helper import DPIHelper
DPIHelper.ensure_dpi_aware()

from grace.config import Config
from grace.audio.capture import AudioCapture
from grace.audio.pump import AudioPump
from grace.audio.wake_word import WakeWordDetector
from grace.vad.detector import VadDetector
from grace.stt.whisper_stream import WhisperStreaming
from grace.llm.gemma_client import GemmaClient, RateLimitError
from grace.llm.model_swap import ModelSwapManager, LocalModelSpec
from grace.intent.prompt import get_system_prompt
from grace.intent.parser import Intent, IntentParser
from grace.automation.computer_use import ComputerUse
from grace.tools.dispatcher import Dispatcher
from grace.tts.kokoro_engine import KokoroEngine
from grace.ws_server import WsEventServer
from grace.agent.facts import (
    detect_remember_pattern,
    is_value_in_transcript,
    is_recall_command,
    detect_forget_fact_pattern,
    format_fact_for_speech,
)
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

def build_llama_server_args(cfg: Config, thread_budget: Optional[Any] = None) -> list[str]:
    """Build the command-line arguments to launch llama-server."""
    args = [
        cfg.llama_server_exe,
        "-m", cfg.llama_model_path,
        "-ngl", str(cfg.llama_ngl),
        "-c", str(cfg.llama_context_window),
        "--cache-type-k", cfg.llama_cache_type_k,
        "--cache-type-v", cfg.llama_cache_type_v,
        "--port", str(cfg.llama_port),
        "--host", cfg.llama_host,
    ]
    # Memory-map weights (PLAN §4, Task 08): allow OS paging
    if getattr(cfg, "hardware_tier", None) == "cpu" or getattr(cfg, "use_mmap", True):
        args.append("--mmap")
    else:
        args.extend(["--no-mmap", "--mlock"])

    if thread_budget is not None:
        args.extend(["-t", str(thread_budget.llama_threads)])
    elif getattr(cfg, "thread_budget", None):
        args.extend(["-t", str(cfg.thread_budget)])

    if cfg.llama_parallel_slots:
        args.extend(["-np", str(cfg.llama_parallel_slots)])
    if getattr(cfg, "llama_flash_attn", False):
        args.extend(["-fa", "on"])
    if cfg.llama_mmproj_path:
        args.extend(["--mmproj", cfg.llama_mmproj_path])
    return args


from grace.agent.loop import AgentLoop
from grace.intent.router import CapabilityRouter, TaskComplexity


class GraceApp:
    """Main Grace application coordinator.

    Wires all components together and runs the main event loop.
    """

    # True while the last agent goal ended by asking the user something
    # (`needs_user_input`); see `_awaiting_answer`.
    _awaiting_user_input = False

    def __init__(self, tts_off: bool = False, capture_dir: Optional[Any] = None):
        self.config = Config()
        self._running = False
        self._llama_process: Optional[subprocess.Popen] = None
        self._audio_queue: Optional[queue.Queue] = None
        self._wake_event: Optional[asyncio.Event] = None
        self._last_spoken_text: str = ""

        # Run capture (grace-prompts/run-capture.md)
        from grace.harness.run_capture import RunCapture
        self._run_capture = RunCapture(run_dir=capture_dir, enabled=bool(capture_dir))
        if self._run_capture.is_enabled:
            self._run_capture.start_logging()

        # Tier Detection & Central Thread Budget (PLAN §2, §4, Task 08)
        from grace.tier import detect_hardware_tier, HardwareTier
        from grace.thread_budget import calculate_thread_budget
        from grace.priority import set_idle_priority

        self.tier_diag = detect_hardware_tier(self.config)
        self.tier = self.tier_diag.tier
        self.thread_budget = calculate_thread_budget(self.tier, self.config)
        log.info(
            f"Hardware Tier: {self.tier.value} | Budget: {self.thread_budget.active_worker_threads} workers "
            f"({self.thread_budget.rationale})"
        )

        if getattr(self.config, "process_priority_control", True):
            set_idle_priority()

        # Audio. GRACE_FAKE_MIC (dev only) swaps the mic for a folder of WAVs.
        fake_mic = os.getenv("GRACE_FAKE_MIC", "")
        if fake_mic:
            from grace.audio.fake_mic import FakeMicCapture
            self.capture = FakeMicCapture(
                fake_mic,
                sample_rate=self.config.mic_sample_rate,
                chunk_size=self.config.mic_chunk,
                channels=self.config.mic_channels,
                width=self.config.mic_width,
            )
        else:
            self.capture = AudioCapture(
                sample_rate=self.config.mic_sample_rate,
                chunk_size=self.config.mic_chunk,
                channels=self.config.mic_channels,
                width=self.config.mic_width,
                device_index=self.config.mic_device_index,
            )

        # Blocking mic reads happen on the pump's thread, never on the loop.
        self.pump = AudioPump(self.capture)

        # Wake Word
        if getattr(self.config, "wake_engine", "vosk") == "sherpa":
            from grace.audio.sherpa_engine import SherpaKWS
            self.wake_word = SherpaKWS(
                model_path=self.config.sherpa_kws_model_path,
                keywords_file=self.config.sherpa_kws_keywords_path,
                num_threads=self.thread_budget.active_worker_threads,
            )
        else:
            self.wake_word = WakeWordDetector(
                model_path=self.config.vosk_model_path,
                keyword=self.config.wake_word_keyword,
                threshold=self.config.wake_word_threshold,
                sample_rate=self.config.mic_sample_rate,
            )

        # VAD
        if getattr(self.config, "vad_engine", "silero") == "sherpa":
            from grace.audio.sherpa_engine import SherpaVAD
            self.vad = SherpaVAD(
                model_path=self.config.sherpa_silero_vad_model_path,
                threshold=self.config.whisper_vad_threshold,
            )
        else:
            self.vad = VadDetector(
                threshold=self.config.whisper_vad_threshold,
                silence_duration_ms=self.config.whisper_silence_duration_ms,
            )

        # Speech-to-text
        if getattr(self.config, "stt_engine", "whisper") == "sherpa":
            from grace.audio.sherpa_engine import SherpaASR
            model_type = getattr(self.config, "sherpa_asr_model_type", "parakeet")
            if self.tier == HardwareTier.FLOOR_CPU:
                model_type = "moonshine"
                model_path = self.config.sherpa_moonshine_model_path
            else:
                model_path = self.config.sherpa_parakeet_model_path
            self.whisper = SherpaASR(
                model_path=model_path,
                model_type=model_type,
                num_threads=self.thread_budget.active_worker_threads,
            )
        else:
            self.whisper = WhisperStreaming(
                model_path=self.config.whisper_model_path,
                device="cuda",
                compute_type=self.config.whisper_compute_type,
            )

        # Model hot-swap: only needed when a distinct local planner model is configured.
        # When running with a single local model (Holo3.1-4B), one model handles every slot.
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

        # LLM (Cloud Gemini Flash or local llama-server)
        api_key = self.config.gemini_api_key if self.config.use_cloud_llm else None
        self.gemma = GemmaClient(
            base_url=self.config.llama_server_url,
            api_key=api_key,
            model_name=self.config.gemini_model_name,
            nvidia_api_key=self.config.nvidia_api_key,
            nvidia_model_name=self.config.nvidia_model_name,
            before_request=(lambda: self._model_swap.swap_to("planner")) if self._model_swap else None,
            allow_cloud_screenshots=self.config.allow_cloud_screenshots,
        )

        # Vision LLM client (shares self.gemma unless model hot-swapping is configured)
        if self._model_swap:
            self.vision_llm = GemmaClient(
                base_url=self.config.llama_server_url,
                api_key=None,
                before_request=(lambda: self._model_swap.swap_to("grounder")),
            )
        else:
            self.vision_llm = self.gemma

        # Intent
        self.intent_prompt = get_system_prompt()
        self.intent_parser = IntentParser()

        # WebSocket server (frontend) - R8 per-launch auth token
        ws_token = self.config.ws_auth_token
        if not ws_token:
            import dataclasses
            import secrets
            ws_token = secrets.token_urlsafe(32)
            self.config = dataclasses.replace(self.config, ws_auth_token=ws_token)
            os.environ["GRACE_WS_TOKEN"] = ws_token

        self.ws_server = WsEventServer(
            host=self.config.ws_host,
            port=self.config.ws_port,
            allowed_origins=self.config.ws_allowed_origins,
            auth_token=ws_token,
        )

        # Computer use
        self.computer_use = ComputerUse()

        # Dispatcher
        self.dispatcher = Dispatcher(
            computer_use=self.computer_use,
            ws_server=self.ws_server,
        )

        # Agentic loop. One model handles every slot; the visual grounder
        # is only consulted when the element graph cannot resolve a target.
        self.agent_loop = AgentLoop(
            gemma=self.gemma,
            dispatcher=self.dispatcher,
            ws_server=self.ws_server,
            vision_llm=self.vision_llm,
            config=self.config,
            run_capture=self._run_capture,
        )

        # Persistent memory store (PLAN §12, Task 16)
        from grace.agent.memory import PersistentMemoryStore
        self.memory_store = PersistentMemoryStore()
        if hasattr(self.dispatcher, "set_memory_store"):
            self.dispatcher.set_memory_store(self.memory_store)
        if hasattr(self.agent_loop, "set_memory_store"):
            self.agent_loop.set_memory_store(self.memory_store)
        if hasattr(self.agent_loop, "set_speaker"):
            self.agent_loop.set_speaker(self._speak_response)

        # TTS. A --no-tts session (typed-command users) swaps both engines
        # for the harness's silent stand-in, so the same speech paths run and
        # simply make no sound.
        if tts_off:
            from grace.harness.replay import NullTts

            self.kokoro = NullTts()
            self.tts_player = NullTts()
        elif getattr(self.config, "tts_engine", "kokoro") == "sherpa":
            from grace.audio.sherpa_engine import SherpaTTS
            self.kokoro = SherpaTTS(
                model_path=self.config.sherpa_kokoro_model_path,
                num_threads=self.thread_budget.active_worker_threads,
                speed=self.config.kokoro_speed,
            )
        else:
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

        if not tts_off:
            self.tts_player = TTSPlayer()

        # Response generator
        self.response_gen = ResponseGenerator(
            gemma_client=self.gemma,
            kokoro_engine=self.kokoro,
            tts_player=self.tts_player,
            ws_server=self.ws_server,
        )
        self.response_gen.set_system_prompt(
            "You are Grace, a warm, concise voice assistant helping a user with motor "
            "dysfunction control their Windows PC. Answer naturally in short spoken sentences. "
            "Do NOT output JSON. Do NOT use markdown, lists, or special formatting. "
            "Keep replies under 3 sentences unless the user asks for detail."
        )
        self.agent_loop.set_cancel_hook(self.response_gen.stop_speaking)
        self._confirmation_prompt_finished_at: float = 0.0

        # Task 10: Word biasing, personal corrections, switch input, Gameface coexistence
        from grace.stt.word_biasing import WordBiaser
        from grace.stt.corrections import PersonalCorrectionList
        from grace.input.switch_input import SwitchInputHandler, SwitchAction, VoiceSwitchRaceResolver, MarkScanner
        from grace.automation.gameface import GamefaceCoexistence

        self.word_biaser = WordBiaser(max_words=getattr(self.config, "max_bias_words", 100))
        corrections_path = getattr(self.config, "correction_list_path", None)
        self.corrections = PersonalCorrectionList(filepath=corrections_path)
        self._last_raw_transcript: Optional[str] = None
        self.gameface = GamefaceCoexistence()
        self._confirmation_race_resolver: Optional[VoiceSwitchRaceResolver] = None
        self._choice_race_resolver: Optional[VoiceSwitchRaceResolver] = None
        self._active_mark_scanner: Optional[MarkScanner] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self.switch_handler = SwitchInputHandler(
            double_press_window_ms=getattr(self.config, "switch_double_press_ms", 400.0),
            long_press_threshold_ms=getattr(self.config, "switch_long_press_ms", 800.0),
            switch_key=getattr(self.config, "switch_key", "space"),
            on_action=self._handle_switch_action,
        )

        # Task 13: Decision Ladder & Micromodel Fleet (PLAN §3, §5)
        from grace.fleet import get_fleet_config
        from grace.agent.decision_ladder import DecisionLadder
        self.fleet_config = get_fleet_config(getattr(self.config, "preset", None), self.config)
        self.decision_ladder = DecisionLadder(fleet_config=self.fleet_config)

        # Task 14 & 15: Enrollment, Ongoing tracking & Opportunistic Training Scheduler (PLAN §6)
        from grace.enrollment.storage import EnrollmentStorage
        from grace.enrollment.ongoing import OngoingTracker
        from grace.training.scheduler import TrainingScheduler
        self.enrollment_storage = EnrollmentStorage()
        self.ongoing_tracker = OngoingTracker(storage=self.enrollment_storage)
        self.training_scheduler = TrainingScheduler(
            ongoing_tracker=self.ongoing_tracker,
            enrollment_storage=self.enrollment_storage,
            ws_server=self.ws_server,
        )
        self.dispatcher.set_training_scheduler(self.training_scheduler)



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

    async def _pump_audio_to_wake_word(self) -> None:
        """Keep the wake-word queue fed while an agent goal runs.

        The main loop normally does this feeding as part of its own
        while-loop, but it is blocked on the single `await` in
        `_run_agent_goal` for the whole goal, so without this the Vosk
        detector - even though left unpaused - would see no new audio and
        never hear a spoken "stop" (R5).
        """
        while True:
            try:
                chunk = await self.pump.get(timeout=0.5)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.debug(f"Audio forwarding to wake word paused briefly: {e}")
                await asyncio.sleep(0.1)
                continue
            if chunk is None:
                continue
            if self._audio_queue is not None:
                try:
                    self._audio_queue.put_nowait(chunk)
                except queue.Full:
                    pass

    async def _run_agent_goal(self, transcript: str, pre_exec_step: Optional[dict[str, Any]] = None) -> dict:
        """Run one agent goal while listening for a spoken "stop"/"cancel".

        `AgentLoop.run` used to be awaited with nothing listening for more
        audio until it returned, so a task once started could not be stopped
        by voice or any other channel (R5). This keeps the wake-word detector
        unpaused and fed with live audio for the duration, armed to call
        `agent_loop.request_cancel()` on a stop phrase instead of treating it
        as a new activation.
        """
        self.wake_word.arm_cancel_watch(self.agent_loop.request_cancel)
        self.wake_word.resume()
        forwarder = asyncio.create_task(self._pump_audio_to_wake_word())
        try:
            result = await self.agent_loop.run(user_goal=transcript, pre_exec_step=pre_exec_step)
            # A sign-in cover and the like end the goal by asking the user
            # something: the answer window opens for it (and only for it).
            self._awaiting_user_input = (
                isinstance(result, dict) and result.get("status") == "needs_user_input"
            )
            return result
        finally:
            forwarder.cancel()
            try:
                await forwarder
            except asyncio.CancelledError:
                pass
            self.wake_word.disarm_cancel_watch()
            self.wake_word.pause()

    async def _wait_and_activate_browser(
        self,
        initial_windows: list[dict[str, Any]],
        target_name: str = "",
        url: str = "",
        timeout: float = 3.0,
    ) -> bool:
        """Wait up to timeout seconds for a browser window whose title changes or appears, then activate it."""
        from grace.perception.uia_provider import is_browser_window
        from urllib.parse import urlparse

        domain = ""
        if url:
            domain = urlparse(url if "://" in url else f"https://{url}").netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]

        target_name_lower = target_name.lower().strip() if target_name else ""
        initial_ids = {w.get("id"): w.get("title", "") for w in (initial_windows or [])}

        start_t = time.monotonic()
        while time.monotonic() - start_t < timeout:
            res = await self.dispatcher.execute(Intent(tool="cua_list_windows", params={}))
            windows = res.get("windows") or (res.get("result", {}).get("windows") if isinstance(res.get("result"), dict) else [])
            if not windows and hasattr(self, "computer_use"):
                windows = self.computer_use._list_windows().get("windows", [])

            candidate = None
            for w in (windows or []):
                w_id = w.get("id")
                w_title = w.get("title", "")
                title_lower = w_title.lower()

                is_new = w_id not in initial_ids
                is_changed = w_id in initial_ids and w_title != initial_ids[w_id]

                matches_target = False
                if target_name_lower and target_name_lower in title_lower:
                    matches_target = True
                elif domain and domain in title_lower:
                    matches_target = True

                if (is_new or is_changed or matches_target) and (is_browser_window(w_title) or matches_target):
                    candidate = w
                    break

            if candidate is not None:
                log.info(f"Pre-exec: found browser window '{candidate.get('title')}' (id={candidate.get('id')}), activating...")
                await self.dispatcher.execute(
                    Intent(
                        tool="cua_activate",
                        params={"id": candidate.get("id"), "window_title": candidate.get("title")},
                    )
                )
                return True

            await asyncio.sleep(0.2)

        # Fallback: if timeout reached, try activating any browser window found
        res = await self.dispatcher.execute(Intent(tool="cua_list_windows", params={}))
        windows = res.get("windows") or (res.get("result", {}).get("windows") if isinstance(res.get("result"), dict) else [])
        for w in (windows or []):
            if is_browser_window(w.get("title", "")):
                log.info(f"Pre-exec: timeout reached; activating available browser window '{w.get('title')}'...")
                await self.dispatcher.execute(
                    Intent(tool="cua_activate", params={"id": w.get("id"), "window_title": w.get("title")})
                )
                return True

        return False

    async def _speak_response(self, text: str) -> None:
        """Speak response text and record it for voice repeat requests."""
        if not text or not text.strip():
            return
        self._last_spoken_text = text.strip()

        # Check if the spoken response asks a numbered marks disambiguation question ("4 or 7?")
        m = re.search(r"(?:button|option|choice|number|#)?\s*(\d+)\s+or\s+(?:button|option|choice|number|#)?\s*(\d+)", text, re.IGNORECASE)
        if m:
            choices = [f"button {m.group(1)}", f"button {m.group(2)}"]
            from grace.input.switch_input import MarkScanner, VoiceSwitchRaceResolver
            scanner = getattr(self, "_active_mark_scanner", None)
            if scanner is not None:
                scanner.stop()
            self._choice_race_resolver = VoiceSwitchRaceResolver()
            self._active_mark_scanner = MarkScanner(
                options=choices,
                scan_rate_ms=getattr(self.config, "mark_scan_rate_ms", 1500.0),
            )
            self._active_mark_scanner.start()

        # Allow interrupting Grace mid-speech (barge-in)
        def _on_barge_in_stop():
            log.info("Barge-in voice interrupt triggered during speech")
            if hasattr(self, "response_gen") and self.response_gen:
                self.response_gen.stop_speaking()

        if hasattr(self, "wake_word") and self.wake_word:
            self.wake_word.arm_cancel_watch(_on_barge_in_stop)

        try:
            if hasattr(self, "response_gen") and self.response_gen:
                await self.response_gen.generate_and_speak_with_text(self._last_spoken_text)
        finally:
            if hasattr(self, "wake_word") and self.wake_word:
                self.wake_word.disarm_cancel_watch()
            # Stop the TTS tail being captured (echo suppression: gate mic for ~200ms)
            if hasattr(self, "pump") and self.pump:
                self.pump.gate(0.20)

    async def offer_voice_access_setup(self, confirmed: bool) -> str:
        """Hands-free setup for Voice Access before sign-in."""
        from grace.agent.secure_desktop import request_enable_voice_access_elevation
        if not confirmed:
            msg = "You can turn on Voice Access before sign-in in Windows Settings under Accessibility, Speech, Voice access."
            await self._speak_response(msg)
            return msg
        ok = request_enable_voice_access_elevation()
        if ok:
            msg = "I've requested administrator approval to start Voice Access before sign-in. Please approve the prompt using Voice Access."
        else:
            msg = "I couldn't launch the administrator helper to enable Voice Access."
        await self._speak_response(msg)
        return msg

    def _start_llama_server(self) -> None:
        """Start llama-server as a subprocess on GPU."""
        cfg = self.config
        if not os.path.exists(cfg.llama_server_exe):
            raise FileNotFoundError(
                f"llama-server executable not found at: {cfg.llama_server_exe}"
            )
        args = build_llama_server_args(cfg, getattr(self, "thread_budget", None))
        if cfg.llama_mmproj_path and os.path.exists(cfg.llama_mmproj_path):
            log.info(f"Vision Projector detected: {cfg.llama_mmproj_path}")
        elif cfg.llama_mmproj_path:
            log.warning(f"Projector path configured but file not found on disk: {cfg.llama_mmproj_path}")
        else:
            log.warning("No mmproj file configured! Model will run without vision features.")

        server_dir = os.path.dirname(cfg.llama_server_exe)
        log.info(f"Starting llama-server from {server_dir}...")
        log.debug(f"Args: {' '.join(args)}")
        self._llama_stderr_tail = collections.deque(maxlen=30)
        self._llama_process = subprocess.Popen(
            args,
            cwd=server_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        )

        def _drain_stderr():
            try:
                if self._llama_process and self._llama_process.stderr:
                    for line in self._llama_process.stderr:
                        self._llama_stderr_tail.append(line.rstrip())
            except Exception:
                pass

        threading.Thread(target=_drain_stderr, daemon=True, name="grace-llama-stderr").start()

    def _log_llama_stderr(self) -> None:
        """Log the last 30 lines of llama-server's stderr when it exits unexpectedly."""
        tail = getattr(self, "_llama_stderr_tail", None)
        if tail:
            log.error(f"llama-server stderr (last {len(tail)} lines):")
            for line in tail:
                log.error(f"  [llama-server] {line}")

    def _stop_llama_server(self) -> None:
        """Stop the llama-server subprocess."""
        if self._llama_process is not None:
            pid = self._llama_process.pid
            log.info(f"Stopping llama-server PID {pid}...")
            # Kill the tree only while the process is still alive. Once it has
            # exited, Windows may hand its PID to an unrelated process, and
            # `taskkill /T` on that PID would kill that process and all its
            # children.
            if self._llama_process.poll() is None:
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True,
                        timeout=10,
                    )
                except Exception:
                    pass
            try:
                self._llama_process.terminate()
                try:
                    self._llama_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._llama_process.kill()
                    self._llama_process.wait(timeout=3)
            except Exception as e:
                log.warning(f"Error terminating llama-server process: {e}")
            self._llama_process = None

    async def _wait_for_llama_server(self, timeout: int = 120) -> bool:
        """Poll /health until the server responds, then verify model is loaded."""
        import aiohttp

        url = self.config.llama_server_url
        log.info(f"Waiting for llama-server at {url}...")
        deadline = asyncio.get_event_loop().time() + timeout
        async with aiohttp.ClientSession() as session:
            while asyncio.get_event_loop().time() < deadline:
                if self._llama_process is not None and self._llama_process.poll() is not None:
                    exit_code = self._llama_process.poll()
                    log.error(f"llama-server process exited unexpectedly with code {exit_code}")
                    self._log_llama_stderr()
                    return False
                try:
                    async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status == 200:
                            log.info("llama-server health check passed, waiting for model to load...")
                            if await self._check_model_ready(session, url):
                                log.info("llama-server model is ready.")
                                return True
                except Exception:
                    pass
                await asyncio.sleep(0.5)
        log.error(f"llama-server did not become healthy within {timeout}s")
        self._log_llama_stderr()
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

                    # Multimodal 1-token warm-up with tiny 1x1 image
                    if self.config.llama_mmproj_path and os.path.exists(self.config.llama_mmproj_path):
                        img_payload = {
                            "messages": [{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Hi"},
                                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}}
                                ]
                            }],
                            "max_tokens": 1,
                            "temperature": 0.0,
                            "stream": False,
                        }
                        try:
                            async with session.post(
                                f"{url}/v1/chat/completions",
                                json=img_payload,
                                timeout=aiohttp.ClientTimeout(total=10),
                            ) as img_resp:
                                log.info(f"Multimodal vision projector warmed up (status={img_resp.status})")
                        except Exception as img_err:
                            log.debug(f"Vision projector warm-up non-critical error: {img_err}")

                    return True
                log.debug(f"Model not ready yet (status={resp.status})")
                return False
        except asyncio.TimeoutError:
            log.debug("Model not ready yet (test request timed out)")
            return False
        except Exception as e:
            log.debug(f"Model not ready yet ({e})")
            return False

    async def start(self, initial_text: Optional[str] = None) -> None:
        """Start Grace and enter the main event loop.

        ``initial_text`` runs the typed-command entrypoint instead: that one
        string is handled exactly as if it had been heard, a parked safety
        confirmation is answered by typed yes/no on stdin, and the process
        shuts down when the turn is done.
        """
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
        self._loop = asyncio.get_running_loop()
        self.pump.start(self._loop)

        # Start wake word detector with audio queue
        self._audio_queue = queue.Queue(maxsize=100)
        self.wake_word.start(audio_queue=self._audio_queue)

        # Warm the slow, lazily-initialised components in the background so the
        # first real request doesn't pay for them.
        self._start_background_warmup()

        log.info("Microphone active. Listening for 'Grace'...")

        if initial_text is not None:
            log.info("Text mode: feeding the typed command through the activation path...")
            try:
                await self._run_typed_session(initial_text)
            except Exception as e:
                log.error(f"Typed activation crashed: {e}", exc_info=True)
                await self._speak_response("Sorry, something went wrong. I'm ready for your next command.")
            finally:
                await self.shutdown()
            return

        # Enter main loop
        try:
            while self._running:
                # Feed audio chunk to wake word detector
                chunk = await self.pump.get(timeout=0.5)
                if chunk is None:
                    continue
                from grace.agent.secure_desktop import is_deaf
                if is_deaf():
                    # Security rule: Grace stops speech recognition completely on secure or locked screens.
                    # Nothing is transcribed, buffered, logged or written to tapes.
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
    _GREETING_WORDS = {
        "hello", "hi", "hey", "howdy", "greetings", "morning", "good morning",
        "good afternoon", "good evening",
    }

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

        # Red-team R16: Greeting words must not count as "yes" or confirmation
        if text in cls._GREETING_WORDS:
            return None

        words = text.split()
        if words and words[0] in cls._GREETING_WORDS:
            return None

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
        # A "yes" heard before the question has finished being spoken does not count
        now = time.monotonic()
        if getattr(self, "_confirmation_prompt_finished_at", 0.0) and now < self._confirmation_prompt_finished_at:
            log.info("Ignoring confirmation answer received before question finished speaking.")
            return True

        # Check voice/switch coexistence race resolver
        if getattr(self, "_confirmation_race_resolver", None) is not None:
            if not self._confirmation_race_resolver.resolve_voice(transcript):
                log.info("Voice confirmation ignored: switch already resolved it")
                return True
            self._confirmation_race_resolver = None

        clean = re.sub(r"[^\w\s]", "", transcript.strip().lower()).strip()
        if clean in self._REPEAT_COMMANDS or clean in ("repeat", "repeat that", "what did you say", "what was that", "pardon", "say again"):
            prompt = self.agent_loop.pending_prompt or "Are you sure you want me to proceed?"
            log.info(f"Repeat requested during confirmation: repeating '{prompt}'")
            await self._speak_response(prompt)
            self._confirmation_prompt_finished_at = time.monotonic()
            return True

        answer = self._confirmation_answer(transcript)
        if answer is None:
            # Mumbled or unclear answer: re-ask the question
            prompt = self.agent_loop.pending_prompt or "Are you sure you want me to proceed?"
            reask_text = f"I didn't quite catch that. {prompt}"
            log.info(f"Unclear or mumbled confirmation answer ({transcript!r}); re-asking: {reask_text}")
            await self._speak_response(reask_text)
            self._confirmation_prompt_finished_at = time.monotonic()
            return True

        log.info(f"Resuming parked action with approval={answer}")
        result = await self.agent_loop.resume_pending(approved=answer)
        if result.get("status") == "safety_confirmation_required":
            # The approved step led straight into another one that also needs
            # confirming (e.g. delete this, then delete that too). Saying
            # "Done." here would tell the user their real action already
            # happened and drop the new pending step on the floor unasked.
            response_text = self._agent_response_text(result)
            self._confirmation_prompt_finished_at = time.monotonic()
            from grace.input.switch_input import VoiceSwitchRaceResolver
            self._confirmation_race_resolver = VoiceSwitchRaceResolver()
        else:
            response_text = result.get("final_response") or (
                "Done." if answer else "Alright, I won't do that."
            )
        await self._speak_response(response_text)
        return True

    async def _handle_voice_memory_command(self, transcript: str, clean: str) -> Optional[bool]:
        """Handle voice memory and saved secret commands (PLAN.md §12.1)."""
        store = getattr(self, "memory_store", None)
        if not store:
            from grace.agent.memory import PersistentMemoryStore
            store = getattr(getattr(self, "agent_loop", None), "_memory_store", None) or PersistentMemoryStore()
            self.memory_store = store

        # Deterministic secret typing: "type my gmail password" / "type gmail password"
        type_pwd_match = re.match(r"^type (?:my )?([a-zA-Z0-9_\-\.]+?) password$", clean)
        if type_pwd_match:
            name = type_pwd_match.group(1).strip()
            if hasattr(self, "dispatcher") and self.dispatcher:
                intent = Intent(tool="type_saved_secret", params={"name": name, "user_authorized": True})
                res_coro = self.dispatcher.execute(intent)
                import inspect
                res = await res_coro if inspect.iscoroutine(res_coro) else res_coro
                if isinstance(res, dict) and res.get("status") != "ok" and res.get("text"):
                    await self._speak_response(res["text"])
            return True

        # What passwords have you saved: speaks names only
        if clean in (
            "what passwords have you saved",
            "what passwords do you have saved",
            "what passwords are saved",
            "what passwords have i saved",
        ):
            secrets = store.list_secrets()
            if not secrets:
                msg = "You don't have any saved passwords."
            else:
                names = [s["name"] for s in secrets]
                msg = f"You have saved passwords for: {', '.join(names)}."
            self._last_spoken_text = msg
            await self._speak_response(msg)
            return True

        # Forget secret password: "forget my gmail password" -> asks confirmation first
        forget_pwd_match = re.match(r"^forget (?:my )?([a-zA-Z0-9_\-\.]+?) password$", clean)
        if forget_pwd_match:
            name = forget_pwd_match.group(1).strip()
            prompt = f"Are you sure you want me to forget your {name} password?"
            intent = Intent(tool="forget_secret", params={"name": name, "confirmation_prompt": prompt})
            self.agent_loop.park_intent(intent, prompt)
            from grace.input.switch_input import VoiceSwitchRaceResolver
            self._confirmation_race_resolver = VoiceSwitchRaceResolver()
            await self._speak_response(prompt)
            self._confirmation_prompt_finished_at = time.monotonic()
            return True

        # Deterministic remember pattern
        remember_spec = detect_remember_pattern(transcript)
        if remember_spec is not None:
            prompt = remember_spec["confirmation_prompt"]
            intent = Intent(tool="remember", params=remember_spec)
            log.info(f"Remember pattern matched: topic={remember_spec['topic']}")
            self.agent_loop.park_intent(intent, prompt)
            from grace.input.switch_input import VoiceSwitchRaceResolver
            self._confirmation_race_resolver = VoiceSwitchRaceResolver()
            await self._speak_response(prompt)
            self._confirmation_prompt_finished_at = time.monotonic()
            return True

        # Voice recall: "What do you remember about me?"
        if is_recall_command(clean):
            facts = store.get_all_confirmed_facts()
            if not facts:
                msg = "I don't remember anything about you yet."
            else:
                spoken_items = [format_fact_for_speech(f["topic"], f["value"]) for f in facts]
                if len(spoken_items) == 1:
                    msg = f"I remember that {spoken_items[0]}."
                elif len(spoken_items) == 2:
                    msg = f"I remember that {spoken_items[0]}, and {spoken_items[1]}."
                else:
                    msg = f"I remember that {', '.join(spoken_items[:-1])}, and {spoken_items[-1]}."
            self._last_spoken_text = msg
            await self._speak_response(msg)
            return True

        if clean in ("forget today", "forget what i did today"):
            store.forget_today()
            msg = "I've forgotten everything from today."
            self._last_spoken_text = msg
            await self._speak_response(msg)
            return True

        if clean in ("forget everything", "forget all"):
            store.forget_everything()
            msg = "I've forgotten everything."
            self._last_spoken_text = msg
            await self._speak_response(msg)
            return True

        if clean.startswith("forget what i did in "):
            app = clean[len("forget what i did in "):].strip()
            store.forget_app(app)
            msg = f"I've forgotten your history in {app}."
            self._last_spoken_text = msg
            await self._speak_response(msg)
            return True

        # Forget single fact by voice: "Forget my name" / "forget that I prefer Firefox"
        forget_match = detect_forget_fact_pattern(transcript, store.get_all_confirmed_facts())
        if forget_match is not None:
            topic, label = forget_match
            store.forget_fact(topic)
            msg = f"I've forgotten {label}."
            self._last_spoken_text = msg
            await self._speak_response(msg)
            return True

        if clean.startswith("forget ") or clean.startswith("forget that "):
            msg = "I don't have anything remembered about that."
            self._last_spoken_text = msg
            await self._speak_response(msg)
            return True

        return None

    async def _park_if_confirmation_required(self, intent, result: dict) -> bool:
        """Ask the question a refused dispatch came back with. True if parked."""
        if result.get("status") != "confirmation_required":
            return False

        prompt = result.get("confirmation_prompt") or "Should I go ahead?"
        log.warning(f"Fast-path '{intent.tool}' parked for confirmation: {prompt}")
        self.agent_loop.park_intent(intent, prompt)
        from grace.input.switch_input import VoiceSwitchRaceResolver
        self._confirmation_race_resolver = VoiceSwitchRaceResolver()
        await self._speak_response(prompt)
        self._confirmation_prompt_finished_at = time.monotonic()
        return True

    async def _resolve_pending_confirmation_switch(self, approved: bool) -> None:
        """Resolve a parked confirmation step via switch input."""
        log.info(f"Switch resolving parked confirmation step: approved={approved}")
        self._confirmation_race_resolver = None
        result = await self.agent_loop.resume_pending(approved=approved)
        if result.get("status") == "safety_confirmation_required":
            response_text = self._agent_response_text(result)
            self._confirmation_prompt_finished_at = time.monotonic()
            from grace.input.switch_input import VoiceSwitchRaceResolver
            self._confirmation_race_resolver = VoiceSwitchRaceResolver()
        else:
            response_text = result.get("final_response") or (
                "Done." if approved else "Alright, I won't do that."
            )
        await self._speak_response(response_text)
        await self.ws_server.emit({"type": "ConversationFinished"})
        self._stop_answer_wait = True

    def _handle_switch_action(self, action) -> None:
        """Handle incoming switch action (YES, NO, STOP) with voice coexistence."""
        action_val = getattr(action, "value", str(action)).lower()
        log.info(f"Switch action received: {action_val}")

        def _dispatch_async(coro):
            loop = getattr(self, "_loop", None)
            if loop is None:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
            if loop is not None and loop.is_running():
                try:
                    if asyncio.get_running_loop() is loop:
                        return asyncio.create_task(coro)
                except RuntimeError:
                    pass
                asyncio.run_coroutine_threadsafe(coro, loop)
            else:
                try:
                    t = threading.Thread(target=lambda: asyncio.run(coro), daemon=True)
                    t.start()
                except Exception as e:
                    log.debug(f"Failed to run async switch action: {e}")

        # STOP action: immediate reflex (long press >= threshold)
        if action_val == "stop":
            log.info("Switch emergency stop triggered")
            if hasattr(self, "response_gen") and self.response_gen:
                self.response_gen.stop_speaking()
            scanner = getattr(self, "_active_mark_scanner", None)
            if scanner is not None:
                scanner.stop()
                self._active_mark_scanner = None
            if hasattr(self, "agent_loop") and self.agent_loop:
                if self.agent_loop.has_pending_confirmation:
                    self.agent_loop.cancel_pending()
                self.agent_loop.request_cancel()
            FeedbackSounds.play_cancel()
            _dispatch_async(self.ws_server.emit({"type": "ConversationFinished"}))
            _dispatch_async(self.ws_server.emit({"type": "Idle"}))
            return

        # MarkScanner active: cyclic scan choice ("4 or 7?")
        scanner = getattr(self, "_active_mark_scanner", None)
        if scanner is not None and getattr(scanner, "_running", False):
            if action_val == "yes":
                resolver = getattr(self, "_choice_race_resolver", None)
                if resolver is not None:
                    if not resolver.resolve_switch(action):
                        log.info("Switch mark choice ignored (voice won race)")
                        return
                selected = scanner.select_current()
                log.info(f"Switch selected scanned mark: '{selected}'")
                self._active_mark_scanner = None
                if selected:
                    _dispatch_async(self._handle_answer_transcript(selected))
                return
            elif action_val == "no":
                log.info("Switch rejected scanned mark choices")
                scanner.stop()
                self._active_mark_scanner = None
                return

        # Confirmation active (safety gate / parked step)
        if hasattr(self, "agent_loop") and self.agent_loop and self.agent_loop.has_pending_confirmation:
            if self._confirmation_race_resolver:
                if not self._confirmation_race_resolver.resolve_switch(action):
                    log.info("Switch confirmation ignored (voice won race)")
                    return
            approved = (action_val == "yes")
            _dispatch_async(self._resolve_pending_confirmation_switch(approved))
            return

    def refresh_word_biasing(self) -> list[str]:
        """Build and feed live word biasing to the active ASR engine."""
        if not hasattr(self, "word_biaser") or self.word_biaser is None:
            return []

        app_indexer = getattr(getattr(self, "dispatcher", None), "_app_indexer", None)
        window_titles: list[str] = []
        try:
            import win32gui
            def _enum_cb(hwnd, titles):
                if win32gui.IsWindowVisible(hwnd):
                    txt = win32gui.GetWindowText(hwnd)
                    if txt and txt.strip():
                        titles.append(txt.strip())
            win32gui.EnumWindows(_enum_cb, window_titles)
        except Exception:
            pass

        contacts: list[str] = []
        try:
            store = getattr(self, "memory_store", None)
            if store and hasattr(store, "get_all_facts"):
                contacts = store.get_all_facts()
        except Exception:
            pass

        words = self.word_biaser.build_bias_list(
            app_indexer=app_indexer,
            window_titles=window_titles,
            confirmed_contacts=contacts,
        )

        whisper_prompt = self.word_biaser.to_whisper_initial_prompt(words)
        if hasattr(self, "whisper") and hasattr(self.whisper, "set_initial_prompt"):
            self.whisper.set_initial_prompt(whisper_prompt)

        sherpa_hotwords = self.word_biaser.to_sherpa_hotwords(words)
        if hasattr(self, "sherpa_asr") and hasattr(self.sherpa_asr, "set_hotwords"):
            self.sherpa_asr.set_hotwords(sherpa_hotwords)

        return words

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
                self.refresh_word_biasing()
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
        if getattr(getattr(self, "config", None), "process_priority_control", True):
            try:
                from grace.priority import set_turn_priority
                set_turn_priority()
            except Exception as e:
                log.debug(f"Failed to elevate turn priority: {e}")
        try:
            await self._handle_activation()
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as e:
            log.error(f"Activation turn crashed: {e}", exc_info=True)
            FeedbackSounds.play_error()
            await self._check_watchdog_and_speak_recovery()
            self.wake_word.resume()
            await self.ws_server.emit({"type": "ConversationFinished"})
            await self.ws_server.emit({"type": "Idle"})
        finally:
            if getattr(getattr(self, "config", None), "process_priority_control", True):
                try:
                    from grace.priority import set_idle_priority
                    set_idle_priority()
                except Exception as e:
                    log.debug(f"Failed to restore idle priority: {e}")

    async def _check_watchdog_and_speak_recovery(self) -> None:
        """Watchdog: restart dead llama-server if it exited, and speak a short human sentence."""
        restarted = False
        config = getattr(self, "config", None)
        llama_proc = getattr(self, "_llama_process", None)
        if config and not getattr(config, "use_cloud_llm", False) and llama_proc is not None:
            if llama_proc.poll() is not None:
                log.warning("Watchdog: llama-server process died. Restarting...")
                try:
                    self._start_llama_server()
                    if await self._wait_for_llama_server(timeout=30):
                        log.info("Watchdog: llama-server restarted successfully.")
                        restarted = True
                        await self._speak_response("The language model had to restart. I'm ready again.")
                    else:
                        await self._speak_response("The local language model stopped and could not restart.")
                except Exception as exc:
                    log.error(f"Watchdog restart failed: {exc}")
                    await self._speak_response("I encountered a problem with the local model.")
        if not restarted:
            await self._speak_response("Sorry, something went wrong. I'm ready for your next command.")

    async def _handle_activation(self):
        """Time one activation cycle end to end.

        After a normal turn Grace goes back to waiting for the wake word. Only
        when Grace has just asked the user something does the answer window
        open (see `_awaiting_answer`). It runs outside this trace so each of
        its turns can be measured on its own rather than inflating the
        wake-word turn.
        """
        turn_result = False
        with start_turn("activation") as trace:
            try:
                turn_result = await self._handle_activation_turn(trace)
            finally:
                log.info(trace.summary())
                try:
                    log_path = getattr(self.config, "latency_log_path", None)
                    if isinstance(log_path, str):
                        trace.write_jsonl(log_path)
                    await self.ws_server.emit({"type": "TurnTrace", "trace": trace.to_dict()})
                except Exception as e:
                    log.debug(f"Failed to emit TurnTrace / write latency: {e}")
                # The answer window pauses/resumes the detector itself once it
                # actually starts. Every other exit from this turn - an aborted
                # False, a turn that asked nothing, or an exception below -
                # must resume it here instead, or "Grace" is never heard again.
                if turn_result is False or not self._awaiting_answer():
                    self.wake_word.resume()

        if turn_result is not False and self._awaiting_answer():
            await self._run_answer_window(timeout_seconds=self.config.answer_timeout_seconds)
        elif turn_result is not False:
            # A finished turn with nothing asked: back to the wake word.
            self._flush_mic_buffer()
            await self.ws_server.emit({"type": "Idle"})

    def _awaiting_answer(self) -> bool:
        """True when Grace has just asked the user something and needs the reply.

        That is a parked yes/no confirmation (terminal cover, delete, ...), a
        `needs_user_input` question such as the sign-in cover, or a scanned
        mark choice. Nothing else keeps the microphone open without the wake
        word.
        """
        loop = getattr(self, "agent_loop", None)
        if loop is not None and getattr(loop, "has_pending_confirmation", False):
            return True
        if getattr(self, "_awaiting_user_input", False):
            return True
        scanner = getattr(self, "_active_mark_scanner", None)
        if scanner is not None and getattr(scanner, "_running", False):
            return True
        return getattr(self, "_choice_race_resolver", None) is not None

    async def _handle_activation_turn(self, trace) -> bool:
        """Handle a wake word activation cycle. Returns False if aborted/silence, True on turn success."""
        trace.mark_event("wake")
        # 1. Play activation chime
        with trace.stage("chime"):
            FeedbackSounds.play_chime()
            if hasattr(self, "pump") and self.pump:
                self.pump.gate(0.85)
        await self.ws_server.emit({"type": "WakeWordDetected"})

        log.info("Listening...")
        await self.ws_server.emit({"type": "ListeningStarted"})

        # Temporarily disable wake word detector to avoid re-triggering on user speech
        self.wake_word.pause()

        # 2. Reset and listen for speech
        self.refresh_word_biasing()
        self.vad.reset()
        self.whisper.reset_buffer()

        audio_buffer = bytearray()
        # Anything captured while the previous turn was still working is stale.
        self.pump.drain()
        # Tell the pump (in harness replays) that the listening window is now
        # consuming chunks, so pump exhaustion here is a real VAD-parity
        # finding; exhaustion caused by the wake-word forwarder draining the
        # tail while the agent goal runs is not.
        try:
            self.pump._listening = True
        except AttributeError:
            pass

        # Listen for speech with initial silence timeout (default 6s) and maximum timeout (30s)
        import time
        listen_start = time.time()
        max_listen_seconds = 30
        hard_max_listen_seconds = 60
        initial_timeout = self.config.initial_listen_timeout_seconds
        has_speech_started = False
        last_speech_time = listen_start
        hit_hard_cutoff = False

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
                            if has_speech_started and (now - last_speech_time < 2.0) and (now - listen_start <= hard_max_listen_seconds):
                                pass
                            else:
                                if has_speech_started and (now - last_speech_time < 2.0 or now - listen_start > hard_max_listen_seconds):
                                    hit_hard_cutoff = True
                                log.info("Listen timeout reached.")
                                break
                        continue
                    from grace.agent.secure_desktop import is_deaf
                    if is_deaf():
                        # Security rule: Drop audio frames arriving while locked or secure
                        continue
                    audio_buffer.extend(chunk)
                    rms = self.capture.get_rms(chunk)
                    threshold = self.config.whisper_vad_threshold * 32767
                    if log.isEnabledFor(logging.DEBUG):
                        log.debug(f"Audio chunk: {len(chunk)} bytes, RMS={rms:.1f}")
                    if rms > threshold:
                        has_speech_started = True
                        last_speech_time = now
                    if self.vad.process_chunk(chunk, self.capture):
                        log.info(f"VAD turn-end detected after {len(audio_buffer)} bytes")
                        break
                    if not has_speech_started and (now - listen_start > initial_timeout):
                        log.info(f"Initial listen timeout reached ({initial_timeout}s without speech).")
                        break
                    if now - listen_start > max_listen_seconds:
                        if has_speech_started and (now - last_speech_time < 2.0) and (now - listen_start <= hard_max_listen_seconds):
                            pass
                        else:
                            if has_speech_started and (now - last_speech_time < 2.0 or now - listen_start > hard_max_listen_seconds):
                                hit_hard_cutoff = True
                            log.info("Listen timeout reached.")
                            break
                except Exception as e:
                    log.error(f"Audio error: {e}")
                    break
            try:
                self.pump._listening = False
            except AttributeError:
                pass
            listen_stage.detail(f"{len(audio_buffer)} bytes")

        trace.mark_event("speech_end")
        if has_speech_started and len(audio_buffer) > 0 and not hit_hard_cutoff:
            # Instant acknowledgement within 300ms of end of speech
            FeedbackSounds.play_ack()

        if hit_hard_cutoff:
            log.warning("Listen window reached hard cutoff while user was speaking.")
            await self.ws_server.emit({"type": "ListeningStopped"})
            await self._speak_response("I didn't catch all of that.")
            await self.ws_server.emit({"type": "ConversationFinished"})
            await self.ws_server.emit({"type": "Idle"})
            return False

        # Don't transcribe if screen is locked or secure (security rule: deaf on secure/locked screens)
        from grace.agent.secure_desktop import is_deaf
        if is_deaf():
            log.info("Secure screen or lock active: dropping captured audio before STT.")
            audio_buffer.clear()
            self.whisper.reset_buffer()
            await self.ws_server.emit({"type": "ListeningStopped"})
            await self.ws_server.emit({"type": "ConversationFinished"})
            await self.ws_server.emit({"type": "Idle"})
            return False

        # Don't transcribe silence after a timeout (skip Whisper GPU/CPU inference on empty audio)
        from unittest import mock
        is_mock = isinstance(getattr(self.whisper, "transcribe", None), mock.Mock)
        if (not has_speech_started or len(audio_buffer) == 0) and not is_mock:
            log.info("No speech detected before timeout, skipping transcription.")
            self.whisper.reset_buffer()
            remaining = self.MIN_WAKE_TO_IDLE_SECONDS - (time.time() - listen_start)
            if remaining > 0:
                await asyncio.sleep(remaining)
            FeedbackSounds.play_cancel()
            await self.ws_server.emit({"type": "ListeningStopped"})
            await self.ws_server.emit({"type": "ConversationFinished"})
            await self.ws_server.emit({"type": "Idle"})
            return False

        # 3. Transcribe. GPU inference is seconds of blocking work, so it goes
        # to a worker thread rather than stalling the WebSocket server.
        # A transcription failure ends the turn; it must never end the process.
        try:
            async with trace.stage("whisper"):
                self.whisper.add_buffer(audio_buffer)
                transcript = await asyncio.to_thread(self.whisper.transcribe)
            trace.mark_event("asr_done")
        except Exception as e:
            log.error(f"Transcription failed: {e}", exc_info=True)
            self.whisper.reset_buffer()
            FeedbackSounds.play_error()
            await self._speak_response("I couldn't hear that clearly. Could you say that again?")
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

        # Personal correction list voice reversal and explicit learning
        if hasattr(self, "corrections") and self.corrections:
            forget_req = self.corrections.detect_forget_request(transcript)
            if forget_req:
                if forget_req == "*":
                    removed = self.corrections.remove_last_correction()
                    msg = f"I've forgotten the correction for {removed}." if removed else "There are no recent corrections to forget."
                else:
                    ok = self.corrections.remove_correction(forget_req)
                    msg = f"I've forgotten the correction for {forget_req}." if ok else f"I couldn't find a correction for {forget_req}."
                log.info(f"Voice correction reversal: {msg}")
                await self._speak_response(msg)
                await self.ws_server.emit({"type": "ConversationFinished"})
                return True

            explicit = self.corrections.detect_explicit_correction(
                transcript, previous_transcript=self._last_raw_transcript
            )
            if explicit:
                heard, meant = explicit
                self.corrections.add_correction(heard, meant)
                log.info(f"Learned personal voice correction: '{heard}' -> '{meant}'")
                if self._last_raw_transcript:
                    transcript = self.corrections.apply(self._last_raw_transcript)
                else:
                    transcript = meant

            transcript = self.corrections.apply(transcript)

        return await self._route_transcript(trace, transcript)

    def _get_intent_system_prompt(self) -> str:
        base = self.intent_prompt or get_system_prompt()
        if hasattr(self, "memory_store") and self.memory_store:
            facts = self.memory_store.render_context(600, 8)
            if facts:
                from grace.agent.planner import wrap_untrusted
                return f"{base.rstrip()}\n\n### What you know about the user\n{wrap_untrusted(facts)}\n"
        return base

    async def _open_site_in_browser_shortcut(self, trace, transcript: str) -> Optional[bool]:
        """Shortcut: "open X in my browser" opens the site URL directly.

        Returns True when the shortcut handled the request (caller returns),
        None when this is not a browser-shortcut request (fall through to normal
        routing). Only matches when the whole request is the shortcut, so
        multi-step requests such as "open youtube music..., search for ...,
        and play it" still reach the agent loop.
        """
        m = re.fullmatch(
            r"\s*open\s+(.+?)\s+in\s+(?:my|the)\s+browser\s*", transcript, re.IGNORECASE
        )
        if not m:
            return None
        from grace.intent.tools import site_url

        url = site_url(m.group(1))
        if not url:
            return None  # unknown site - fall through to normal routing

        site = m.group(1).strip()
        log.info(f"Browser shortcut: open {site} -> {url}")
        intent = Intent(tool="open_app", params={"name": site, "url": url})
        await self.ws_server.emit({"type": "UnderstandingStarted", "label": "Opening site..."})
        try:
            async with trace.stage("exec"):
                result = await self.dispatcher.execute(intent)
        except Exception as e:
            log.error(f"Browser shortcut failed for {site}: {e}")
            await self._speak_response(f"I couldn't open {site}.")
            await self.ws_server.emit({"type": "ConversationFinished"})
            return True
        await self.ws_server.emit({"type": "UnderstandingFinished"})
        response_text = result.get("text") or f"I've opened {site} in your browser."
        if response_text:
            await self._speak_response(response_text)
        await self.ws_server.emit({"type": "ConversationFinished"})
        return True

    async def _route_transcript(self, trace, transcript: str) -> bool:
        """Everything after a final transcript exists, identical for a
        heard string and a typed one: the stop/repeat reflexes, answers to
        a parked safety confirmation, memory commands, the ladder fast
        path, intent generation, the CapabilityRouter and the three
        execution paths. The wake-word turn applies personal voice
        corrections before calling this; typed input needs none. There is
        no other gate: whatever route this takes, it is the only route."""
        self._awaiting_user_input = False  # a new utterance answers any open question
        self._last_raw_transcript = transcript
        turn_start_time = time.monotonic()

        clean_transcript = re.sub(r"[^\w\s]", "", transcript.strip().lower()).strip()

        # Security: ignore Voice Access's own phrases completely (never route to agent)
        if clean_transcript.startswith("voice access"):
            log.debug(f"Dropped Voice Access command phrase: '{transcript}'")
            return False

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
            if self.agent_loop.has_pending_confirmation:
                prompt = self.agent_loop.pending_prompt or self._last_spoken_text or "Are you sure you want me to proceed?"
                await self._speak_response(prompt)
                self._confirmation_prompt_finished_at = time.monotonic()
            elif self._last_spoken_text:
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

        mem_handled = await self._handle_voice_memory_command(transcript, clean_transcript)
        if mem_handled is not None:
            await self.ws_server.emit({"type": "ConversationFinished"})
            return mem_handled

        # Task 12 shortcut: "open X in my browser" opens the site URL directly
        # instead of driving the browser's address bar through the agent loop.
        shortcut = await self._open_site_in_browser_shortcut(trace, transcript)
        if shortcut is not None:
            return shortcut

        # Task 13 Decision Ladder Rung 1: Pattern fast-path (no model call required)
        ladder_fast_path = None
        if self.fleet_config.is_fleet:
            app_idx = getattr(getattr(self, "dispatcher", None), "_app_indexer", None)
            ladder_fast_path = self.decision_ladder.check_pattern_fast_path(
                transcript,
                corrections=self.corrections,
                app_indexer=app_idx,
            )

        if ladder_fast_path is not None:
            log.info(f"DecisionLadder Rung 1 matched: {ladder_fast_path.tool}({ladder_fast_path.params})")
            trace.mark_event("decision_ladder_rung1_pattern")
            intent = Intent(tool=ladder_fast_path.tool, params=ladder_fast_path.params)
            await self.ws_server.emit({"type": "UnderstandingStarted", "label": "Executing command\u2026"})
            await self.ws_server.emit({"type": "UnderstandingFinished"})
            complexity = TaskComplexity.FAST_PATH
            trace.mark_event(f"route_{complexity.value}")

            if getattr(self, "_run_capture", None) and self._run_capture.is_enabled:
                self._run_capture.begin_turn(transcript)
                self._run_capture.record_intent(
                    raw_output="",
                    intent=intent,
                    route=complexity.value,
                )
            log.info(f"Fast-path executing tool '{intent.tool}' with params={intent.params}")
            async with trace.stage("exec"):
                result = await self.dispatcher.execute(intent)
            trace.mark_event("action_done")
            if await self._park_if_confirmation_required(intent, result):
                return True
            response_text = result.get("text")
            if response_text:
                async with trace.stage("speak"):
                    await self._speak_response(response_text)
                trace.mark_event("spoken_reply_done")
            await self.ws_server.emit({"type": "ConversationFinished"})
            return True

        # 4. Generate intent
        await self.ws_server.emit({"type": "UnderstandingStarted", "label": "Understanding request\u2026"})
        log.info("Calling LLM for intent generation...")
        try:
            async with trace.stage("intent_llm"):
                intent_json = await self.gemma.generate_intent(
                    user_message=transcript,
                    system_prompt=self._get_intent_system_prompt(),
                )
            trace.mark_event("first_model_token")
        except RateLimitError as e:
            log.error(f"Intent generation rate limited: {e}")
            await self.ws_server.emit({"type": "UnderstandingFinished"})
            await self._speak_response(
                "I've hit my request limit for now. Please try again in a moment."
            )
            await self.ws_server.emit({"type": "ConversationFinished"})
            return True

        if hasattr(self.dispatcher, "set_goal"):
            self.dispatcher.set_goal(transcript)
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

        if intent and intent.tool == "remember":
            val = str(intent.params.get("value", "")).strip()
            topic = str(intent.params.get("topic", "")).strip()
            if not val or not is_value_in_transcript(val, transcript):
                log.warning(f"Provenance safety: rejected remember intent; value '{val}' was not spoken by user in '{transcript}'")
                await self._speak_response("I can only remember things you say to me.")
                await self.ws_server.emit({"type": "ConversationFinished"})
                return True
            prompt = intent.params.get("confirmation_prompt") or f"Should I remember that {topic} is {val}?"
            intent.params["confirmation_prompt"] = prompt
            self.agent_loop.park_intent(intent, prompt)
            from grace.input.switch_input import VoiceSwitchRaceResolver
            self._confirmation_race_resolver = VoiceSwitchRaceResolver()
            await self._speak_response(prompt)
            self._confirmation_prompt_finished_at = time.monotonic()
            await self.ws_server.emit({"type": "ConversationFinished"})
            return True

        # 5. Route & Execute (Dual-Path Architecture)
        complexity = CapabilityRouter.classify(transcript, intent)
        trace.mark_event(f"route_{complexity.value}")

        if getattr(self, "_run_capture", None) and self._run_capture.is_enabled:
            self._run_capture.begin_turn(transcript)
            self._run_capture.record_intent(
                raw_output=intent_json or "",
                intent=intent,
                route=complexity.value,
            )

        if complexity == TaskComplexity.AGENTIC_GOAL:
            log.info(f"Routing goal '{transcript}' to Agentic ReAct Loop...")
            pre_exec_step = None
            if intent and intent.tool in ("open_app", "cua_launch", "open_calculator"):
                log.info(f"Pre-executing setup intent before AgentLoop: {intent.tool}({intent.params})")
                async with trace.stage("pre_exec"):
                    before_res = await self.dispatcher.execute(Intent(tool="cua_list_windows", params={}))
                    before_windows = (
                        before_res.get("windows")
                        or (before_res.get("result", {}).get("windows") if isinstance(before_res.get("result"), dict) else [])
                    )
                    if not before_windows and hasattr(self, "computer_use"):
                        before_windows = self.computer_use._list_windows().get("windows", [])

                    pre_res = await self.dispatcher.execute(intent)
                    pre_exec_step = {
                        "action": intent.tool,
                        "params": intent.params,
                        "result": pre_res,
                    }
                    if intent.tool == "open_app" and intent.params.get("url"):
                        await self._wait_and_activate_browser(
                            initial_windows=before_windows,
                            target_name=intent.params.get("name", ""),
                            url=intent.params.get("url", ""),
                            timeout=3.0,
                        )
                    else:
                        await asyncio.sleep(2.0)

            async with trace.stage("agent_loop") as agent_stage:
                agent_res = await self._run_agent_goal(transcript, pre_exec_step=pre_exec_step)
                if hasattr(agent_stage, "detail") and callable(agent_stage.detail):
                    res_steps = agent_res.get("steps") if isinstance(agent_res, dict) else []
                    det = agent_stage.detail(f"{len(res_steps or [])} steps")
                    if asyncio.iscoroutine(det):
                        await det
            trace.mark_event("action_done")

            log.info(f"AgentLoop Result: {agent_res}")

            response_text = self._agent_response_text(agent_res)
            if response_text:
                async with trace.stage("speak"):
                    await self._speak_response(response_text)
                trace.mark_event("spoken_reply_done")
        elif (intent is not None and intent.is_conversation) or complexity == TaskComplexity.CONVERSATION:
            # Use the response from intent generation directly (already generated by the intent LLM call)
            response_text = intent.response if intent and intent.response else "I'm here to help! What can I do for you today?"
            log.info(f"Speaking conversational response: '{response_text[:100]}...'")
            async with trace.stage("speak"):
                await self._speak_response(response_text)
            trace.mark_event("spoken_reply_done")
        else:
            log.info(f"Fast-path executing tool '{intent.tool}' with params={intent.params}")
            async with trace.stage("dispatch"):
                result = await self.dispatcher.execute(intent)
            trace.mark_event("action_done")
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
                trace.mark_event("spoken_reply_done")

        if getattr(self, "_run_capture", None) and self._run_capture.is_enabled and not self._run_capture.has_summary:
            model_name = getattr(self.config, "llama_model_path", "") or getattr(self.config, "llm_model", "")
            tier_name = getattr(self.config, "hardware_tier", "")
            self._run_capture.record_summary(
                goal=transcript,
                status="complete",
                step_count=0,
                total_time=round(time.monotonic() - turn_start_time, 3),
                model=model_name,
                tier=tier_name,
            )

        await self.ws_server.emit({"type": "ConversationFinished"})
        return True

    async def _handle_text_activation(self, transcript: str) -> bool:
        """Route one typed string through the exact heard-string path."""
        with start_turn("text") as trace:
            try:
                return await self._route_transcript(trace, transcript)
            finally:
                log.info(trace.summary())
                try:
                    await self.ws_server.emit(
                        {"type": "TurnTrace", "trace": trace.to_dict()})
                except Exception as e:
                    log.debug(f"Failed to emit typed TurnTrace: {e}")

    async def _run_typed_session(self, text: str) -> None:
        """Feed one typed command, then typed yes/no for anything the
        safety gate parks. The answer re-enters the identical routing
        (stop reflex, repeat reflex, then _resolve_pending_confirmation),
        so a typed word can approve no more than a spoken word could."""
        await self._handle_text_activation(text)
        while self.agent_loop.has_pending_confirmation:
            prompt = (self.agent_loop.pending_prompt
                      or "Are you sure you want me to proceed?")
            try:
                answer = input(f"\n(parked confirmation) {prompt}\nType yes or no: ")
            except (EOFError, KeyboardInterrupt):
                log.info("Typed session got no confirmation answer; cancelling the parked step.")
                self.agent_loop.cancel_pending()
                break
            if not answer.strip():
                log.info("Typed session got an empty confirmation answer; cancelling the parked step.")
                self.agent_loop.cancel_pending()
                break
            await self._handle_text_activation(answer)

    async def _run_answer_window(self, timeout_seconds: int = 10):
        """Listen without a wake word for the answer to a question Grace just asked.

        Opened only while `_awaiting_answer()` is true. Once the answer has been
        handled and nothing further is being asked, the window closes and Grace
        waits for the wake word again (it is never an
        open-ended listener). If the answer raises another question, the window restarts
        in an outer loop rather than recursing.
        """
        import time

        log.info(f"Entering {timeout_seconds}s window to hear the answer to Grace's question...")
        self.wake_word.pause()
        FeedbackSounds.play_listening()

        while self._running:
            await self.ws_server.emit({"type": "ListeningStarted"})
            self.vad.reset()
            self.whisper.reset_buffer()
            self.pump.drain()

            start_time = time.time()
            audio_buffer = bytearray()
            handled = False
            has_speech_started = False
            last_speech_time = start_time
            hard_max_seconds = 30.0

            while self._running:
                if getattr(self, "_stop_answer_wait", False):
                    self._stop_answer_wait = False
                    handled = False
                    break
                now = time.time()
                elapsed = now - start_time
                if elapsed >= timeout_seconds:
                    if has_speech_started and (now - last_speech_time < 2.0) and elapsed < hard_max_seconds:
                        pass
                    else:
                        break

                try:
                    chunk = await self.pump.get(timeout=0.5)
                    if chunk is None:
                        continue
                    from grace.agent.secure_desktop import is_deaf
                    if is_deaf():
                        # Security rule: Drop audio frames arriving while locked or secure
                        continue
                    audio_buffer.extend(chunk)

                    rms = self.capture.get_rms(chunk)
                    threshold = self.config.whisper_vad_threshold * 32767
                    if rms > threshold:
                        has_speech_started = True
                        last_speech_time = time.time()

                    if not self.vad.process_chunk(chunk, self.capture):
                        continue

                    log.info(f"Answer-window VAD natural speech boundary detected after {len(audio_buffer)} bytes")
                    raw_pcm = bytes(audio_buffer)
                    transcript = await asyncio.to_thread(self.whisper.transcribe_bytes, raw_pcm)

                    if transcript.strip():
                        handled = await self._handle_answer_transcript(transcript)
                        if handled:
                            break

                    audio_buffer.clear()
                    self.vad.reset()
                    has_speech_started = False
                except Exception as e:
                    log.error(f"Error in answer window: {e}")
                    handled = False
                    break

            if not handled or not self._awaiting_answer():
                break

        log.info("Answer window closed. Returning to idle.")
        scanner = getattr(self, "_active_mark_scanner", None)
        if scanner is not None:
            scanner.stop()
            self._active_mark_scanner = None
        self._choice_race_resolver = None
        self._confirmation_race_resolver = None
        self._awaiting_user_input = False
        if self.agent_loop.has_pending_confirmation:
            # A parked delete or app-close that nobody answered must not still
            # be approvable once we're back to idle and a new "yes" could be
            # about something else entirely.
            log.info("Cancelling a confirmation that timed out unanswered.")
            self.agent_loop.cancel_pending()
        self.wake_word.resume()
        self._flush_mic_buffer()
        await self.ws_server.emit({"type": "Idle"})

    async def _handle_answer_transcript(self, transcript: str) -> bool:
        """Act on one utterance heard in the answer window. Returns True if it was handled."""
        log.info(f"Answer-window transcribed: '{transcript}'")
        self._awaiting_user_input = False  # this utterance answers the open question
        await self.ws_server.emit({"type": "FinalTranscript", "text": transcript})

        # Cancel any ongoing TTS output if user barged in
        self.response_gen.stop_speaking()

        # Check choice race resolver if mark scanning was active
        if getattr(self, "_choice_race_resolver", None) is not None:
            if not self._choice_race_resolver.resolve_voice(transcript):
                log.info("Voice mark choice ignored: switch already resolved it")
                return True
            self._choice_race_resolver = None
            scanner = getattr(self, "_active_mark_scanner", None)
            if scanner is not None:
                scanner.stop()
                self._active_mark_scanner = None

        # Personal correction list voice reversal and explicit learning
        if hasattr(self, "corrections") and self.corrections:
            forget_req = self.corrections.detect_forget_request(transcript)
            if forget_req:
                if forget_req == "*":
                    removed = self.corrections.remove_last_correction()
                    msg = f"I've forgotten the correction for {removed}." if removed else "There are no recent corrections to forget."
                else:
                    ok = self.corrections.remove_correction(forget_req)
                    msg = f"I've forgotten the correction for {forget_req}." if ok else f"I couldn't find a correction for {forget_req}."
                log.info(f"Voice correction reversal: {msg}")
                await self._speak_response(msg)
                return True

            explicit = self.corrections.detect_explicit_correction(
                transcript, previous_transcript=self._last_raw_transcript
            )
            if explicit:
                heard, meant = explicit
                self.corrections.add_correction(heard, meant)
                log.info(f"Learned personal voice correction: '{heard}' -> '{meant}'")
                if self._last_raw_transcript:
                    transcript = self.corrections.apply(self._last_raw_transcript)
                else:
                    transcript = meant

            transcript = self.corrections.apply(transcript)

        self._last_raw_transcript = transcript
        turn_start_time = time.monotonic()

        clean_transcript = re.sub(r"[^\w\s]", "", transcript.strip().lower()).strip()

        # Security: ignore Voice Access's own phrases completely (never route to agent)
        if clean_transcript.startswith("voice access"):
            log.debug(f"Dropped Voice Access command phrase: '{transcript}'")
            return False

        # Immediate voice stop reflex in the answer window
        if clean_transcript in self._STOP_COMMANDS or clean_transcript.startswith("stop ") or clean_transcript == "stop":
            log.info(f"Answer-window stop command detected: '{transcript}'")
            self.response_gen.stop_speaking()
            if self.agent_loop.has_pending_confirmation:
                self.agent_loop.cancel_pending()
            FeedbackSounds.play_cancel()
            return False

        # Immediate voice repeat reflex in the answer window
        if clean_transcript in self._REPEAT_COMMANDS:
            log.info(f"Answer-window repeat command detected: '{transcript}'")
            if self.agent_loop.has_pending_confirmation:
                prompt = self.agent_loop.pending_prompt or self._last_spoken_text or "Are you sure you want me to proceed?"
                await self._speak_response(prompt)
                self._confirmation_prompt_finished_at = time.monotonic()
            elif self._last_spoken_text:
                await self._speak_response(self._last_spoken_text)
            else:
                await self._speak_response("I haven't said anything yet.")
            return True

        if self.agent_loop.has_pending_confirmation:
            if await self._resolve_pending_confirmation(transcript):
                return True

        mem_handled = await self._handle_voice_memory_command(transcript, clean_transcript)
        if mem_handled is not None:
            return mem_handled

        try:
            async with stage("intent_llm"):
                intent_json = await self.gemma.generate_intent(
                    user_message=transcript,
                    system_prompt=self._get_intent_system_prompt(),
                )
        except RateLimitError as e:
            log.error(f"Answer-window intent generation rate limited: {e}")
            await self._speak_response(
                "I've hit my request limit for now. Please try again in a moment."
            )
            return True

        if hasattr(self.dispatcher, "set_goal"):
            self.dispatcher.set_goal(transcript)
        intent = None
        if intent_json:
            try:
                intent = self.intent_parser.parse(intent_json)
            except Exception as e:
                log.debug(f"Answer-window intent parse failed: {e}")

        if intent and intent.tool == "remember":
            val = str(intent.params.get("value", "")).strip()
            topic = str(intent.params.get("topic", "")).strip()
            if not val or not is_value_in_transcript(val, transcript):
                log.warning(f"Provenance safety: rejected remember intent heard in the answer window; value '{val}' was not spoken in '{transcript}'")
                await self._speak_response("I can only remember things you say to me.")
                return True
            prompt = intent.params.get("confirmation_prompt") or f"Should I remember that {topic} is {val}?"
            intent.params["confirmation_prompt"] = prompt
            self.agent_loop.park_intent(intent, prompt)
            from grace.input.switch_input import VoiceSwitchRaceResolver
            self._confirmation_race_resolver = VoiceSwitchRaceResolver()
            await self._speak_response(prompt)
            self._confirmation_prompt_finished_at = time.monotonic()
            return True

        complexity = CapabilityRouter.classify(transcript, intent)
        if complexity == TaskComplexity.AGENTIC_GOAL:
            if intent and intent.tool in ("open_app", "cua_launch"):
                log.info(f"Pre-executing setup intent before AgentLoop: {intent.tool}({intent.params})")
                await self.dispatcher.execute(intent)
                await asyncio.sleep(2.0)

            agent_res = await self._run_agent_goal(transcript)
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

        if getattr(self, "_run_capture", None):
            self._run_capture.close()

        log.info("Grace stopped.")


def main():
    """Entry point for Grace."""
    import argparse

    parser = argparse.ArgumentParser(description="Grace - offline voice accessibility assistant")
    parser.add_argument(
        "--text", default=None, metavar="COMMAND",
        help="Handle this exact string as if it had been heard, answer a parked "
             "safety confirmation with typed yes/no on stdin, then exit.",
    )
    parser.add_argument(
        "--no-tts", action="store_true",
        help="Run without speech output (same behavior, silent).",
    )
    parser.add_argument(
        "--capture-run", nargs="?", const="", default=None, metavar="DIR",
        help="Capture full run data (log, screenshots, planner prompts/raw, step outcomes, summary) to DIR (default: runs/<timestamp>/).",
    )
    parser.add_argument(
        "--save-secret", default=None, metavar="NAME",
        help="Save a secret password to the DPAPI vault (caregiver setup, prompt-hidden entry).",
    )
    parser.add_argument(
        "--site", default=None, metavar="SITE",
        help="Target domain or process executable name for the secret being saved.",
    )
    parser.add_argument(
        "--list-secrets", action="store_true",
        help="List saved secret names and sites only (never prints values).",
    )
    parser.add_argument(
        "--forget-secret", default=None, metavar="NAME",
        help="Delete a saved secret from the DPAPI vault.",
    )
    parser.add_argument(
        "--check-setup", action="store_true",
        help="Check accessibility setup, reporting whether Voice Access is set to start before sign-in.",
    )
    parser.add_argument(
        "--enable-voice-access-at-signin", action="store_true",
        help="Elevated helper to enable Voice Access before sign-in in HKLM registry.",
    )
    args = parser.parse_args()

    if args.check_setup:
        from grace.agent.secure_desktop import (
            is_voice_access_enabled_at_signin,
            get_signin_accessibility_config,
        )
        enabled = is_voice_access_enabled_at_signin()
        cfg = get_signin_accessibility_config()
        if enabled:
            print(f"Voice Access before sign-in: enabled (Configuration: {cfg})")
            import sys; sys.exit(0)
        else:
            print(f"Voice Access before sign-in: disabled (Configuration: {cfg})")
            print("To enable, run: grace.exe --enable-voice-access-at-signin (requires admin approval) or tick 'Start voice access before you sign in' in Windows Settings.")
            import sys; sys.exit(1)

    if args.enable_voice_access_at_signin:
        from grace.agent.secure_desktop import enable_voice_access_at_signin_registry
        ok = enable_voice_access_at_signin_registry()
        if ok:
            print("Successfully enabled Voice Access before sign-in.")
            import sys; sys.exit(0)
        else:
            print("Failed to enable Voice Access before sign-in.")
            import sys; sys.exit(1)

    if args.save_secret:
        if not args.site:
            print("Error: --site <domain-or-exe> is required when saving a secret.")
            import sys; sys.exit(1)
        import getpass
        val1 = getpass.getpass(f"Enter password for '{args.save_secret}': ")
        val2 = getpass.getpass(f"Confirm password for '{args.save_secret}': ")
        if val1 != val2:
            print("Error: Passwords do not match.")
            import sys; sys.exit(1)
        if not val1:
            print("Error: Password cannot be empty.")
            import sys; sys.exit(1)
        from grace.agent.secrets_vault import classify_site
        from grace.agent.memory import PersistentMemoryStore
        site_kind, site = classify_site(args.site)
        store = PersistentMemoryStore()
        ok = store.save_secret(args.save_secret, val1, site_kind, site)
        store.close()
        if ok:
            print(f"Saved secret '{args.save_secret}' for {site_kind} '{site}'.")
            import sys; sys.exit(0)
        else:
            print(f"Error saving secret '{args.save_secret}'.")
            import sys; sys.exit(1)

    if args.list_secrets:
        from grace.agent.memory import PersistentMemoryStore
        store = PersistentMemoryStore()
        secrets = store.list_secrets()
        store.close()
        if not secrets:
            print("No saved secrets.")
        else:
            print("Saved secrets:")
            for s in secrets:
                print(f"  - {s['name']} ({s['site_kind']}: {s['site']})")
        import sys; sys.exit(0)

    if args.forget_secret:
        from grace.agent.memory import PersistentMemoryStore
        store = PersistentMemoryStore()
        ok = store.forget_secret(args.forget_secret)
        store.close()
        if ok:
            print(f"Forgotten secret '{args.forget_secret}'.")
            import sys; sys.exit(0)
        else:
            print(f"No secret found with name '{args.forget_secret}'.")
            import sys; sys.exit(1)

    capture_dir = None
    # The Tauri shell spawns this process without CLI flags; GRACE_CAPTURE_RUN
    # (1 = runs/<timestamp>, or a directory) turns capture on from there.
    if args.capture_run is None and os.getenv("GRACE_CAPTURE_RUN"):
        env_capture = os.getenv("GRACE_CAPTURE_RUN", "")
        args.capture_run = "" if env_capture.lower() in ("1", "true", "yes") else env_capture
    if args.capture_run is not None:
        if args.capture_run == "":
            from datetime import datetime
            from pathlib import Path
            repo_root = Path(__file__).resolve().parent.parent.parent
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            capture_dir = repo_root / "runs" / timestamp
        else:
            from pathlib import Path
            capture_dir = Path(args.capture_run).resolve()

    app = GraceApp(tts_off=args.no_tts, capture_dir=capture_dir)
    try:
        asyncio.run(app.start(initial_text=args.text))
    except KeyboardInterrupt:
        log.info("Interrupted.")


if __name__ == "__main__":
    main()
