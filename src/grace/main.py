"""Grace - Offline voice accessibility assistant for Windows.

Main entry point. Wires together all components and runs the
main event loop: wake word → speech capture → transcription →
intent generation → tool execution → response TTS → repeat.
"""

import asyncio
import logging
import os
import queue
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

from grace.config import Config
from grace.audio.capture import AudioCapture
from grace.audio.wake_word import WakeWordDetector
from grace.vad.detector import VadDetector
from grace.stt.whisper_stream import WhisperStreaming
from grace.llm.gemma_client import GemmaClient
from grace.intent.prompt import get_system_prompt
from grace.intent.parser import IntentParser
from grace.automation.computer_use import ComputerUse
from grace.tools.dispatcher import Dispatcher
from grace.tts.kokoro_engine import KokoroEngine
from grace.ws_server import WsEventServer
from grace.tts.player import TTSPlayer
from grace.response.generator import ResponseGenerator
from grace.response.feedback import FeedbackSounds

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
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

        # Audio
        self.capture = AudioCapture(
            sample_rate=self.config.mic_sample_rate,
            chunk_size=self.config.mic_chunk,
            channels=self.config.mic_channels,
            width=self.config.mic_width,
            device_index=self.config.mic_device_index,
        )

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

        # LLM (Cloud Gemini 3.6 Flash or local llama-server)
        api_key = self.config.gemini_api_key if self.config.use_cloud_llm else None
        self.gemma = GemmaClient(
            base_url=self.config.llama_server_url,
            api_key=api_key,
            model_name=self.config.gemini_model_name,
        )

        # Dedicated UI-TARS Vision LLM client for AgentLoop CUA steps (local llama-server)
        self.ui_tars_client = GemmaClient(
            base_url=self.config.llama_server_url,
            api_key=None,  # Forces local llama-server for UI-TARS GUI vision
        ) if self.config.use_ui_tars_local else self.gemma

        # Intent
        self.intent_prompt = get_system_prompt()
        self.intent_parser = IntentParser()

        # WebSocket server (frontend)
        self.ws_server = WsEventServer(
            host=self.config.ws_host,
            port=self.config.ws_port,
        )

        # Computer use
        self.computer_use = ComputerUse()

        # Dispatcher
        self.dispatcher = Dispatcher(
            computer_use=self.computer_use,
            ws_server=self.ws_server,
        )

        # Agentic Loop Engine (passes UI-TARS for CUA vision steps)
        self.agent_loop = AgentLoop(
            gemma=self.gemma,
            dispatcher=self.dispatcher,
            ws_server=self.ws_server,
            vision_llm=self.ui_tars_client,
        )

        # TTS
        self.kokoro = KokoroEngine(
            model_path=self.config.kokoro_model_path,
            voices_path=self.config.kokoro_voices_path,
            num_workers=self.config.kokoro_workers,
            device=self.config.kokoro_device,
            dtype=self.config.kokoro_dtype,
        )

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

    def _start_llama_server(self) -> None:
        """Start llama-server as a subprocess with UI-TARS 1.5-7B on GPU."""
        cfg = self.config
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
            log.info(f"Stopping llama-server PID {self._llama_process.pid}...")
            try:
                self._llama_process.kill()
                self._llama_process.wait(timeout=3)
            except Exception as e:
                log.warning(f"Error terminating llama-server process: {e}")
            self._llama_process = None

        try:
            subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
            log.info("Force killed residual llama-server.exe processes.")
        except Exception:
            pass

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

        # Start llama-server for UI-TARS local GUI vision execution
        if not self.config.use_cloud_llm or self.config.use_ui_tars_local:
            log.info("Initializing local llama-server for UI-TARS 1.5-7B vision agent...")
            self._start_llama_server()
            if not await self._wait_for_llama_server(timeout=120):
                if not self.config.use_cloud_llm:
                    log.error("Failed to start llama-server. Exiting.")
                    await self.shutdown()
                    return
                else:
                    log.warning("llama-server for UI-TARS did not become healthy; falling back to Cloud LLM for agent steps.")
                    self.ui_tars_client = self.gemma
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

        # Start wake word detector with audio queue
        self._audio_queue = queue.Queue(maxsize=100)
        self.wake_word.start(audio_queue=self._audio_queue)

        log.info("Microphone active. Listening for 'Grace'...")

        # Enter main loop
        try:
            while self._running:
                # Feed audio chunk to wake word detector
                chunk = self.capture.get_chunk()
                if self._audio_queue is not None:
                    try:
                        self._audio_queue.put(chunk, block=False)
                    except queue.Full:
                        pass

                if self.wake_word.detected:
                    log.info("Wake word detected!")
                    self.wake_word.reset()
                    await self._handle_activation()
                elif self._wake_event and self._wake_event.is_set():
                    log.info("Wake command received from frontend!")
                    self._wake_event.clear()
                    await self._handle_activation()
                else:
                    await asyncio.sleep(0)
        except KeyboardInterrupt:
            log.info("Shutdown requested.")
        finally:
            await self.shutdown()

    async def _handle_activation(self):
        """Handle a wake word activation cycle."""
        # 1. Play activation chime
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

        # Listen for speech (max 30s to avoid hanging forever)
        import time
        listen_start = time.time()
        max_listen_seconds = 30
        while self._running:
            try:
                chunk = self.capture.get_chunk()
                audio_buffer.extend(chunk)
                rms = self.capture.get_rms(chunk)
                log.debug(f"Audio chunk: {len(chunk)} bytes, RMS={rms:.1f}")
                if self.vad.process_chunk(chunk, self.capture):
                    log.info(f"VAD turn-end detected after {len(audio_buffer)} bytes")
                    break
                if time.time() - listen_start > max_listen_seconds:
                    log.info("Listen timeout reached.")
                    break
            except Exception as e:
                log.error(f"Audio error: {e}")
                break

        # 3. Transcribe
        self.whisper.add_buffer(audio_buffer)
        transcript = self.whisper.transcribe()

        if not transcript.strip():
            log.info("No speech detected.")
            await self.ws_server.emit({"type": "ConversationFinished"})
            await asyncio.sleep(2.6)
            await self.ws_server.emit({"type": "Idle"})
            return

        log.info(f"Transcribed: '{transcript}'")
        await self.ws_server.emit({"type": "FinalTranscript", "text": transcript})
        await self.ws_server.emit({"type": "ListeningStopped"})

        # 4. Generate intent
        await self.ws_server.emit({"type": "UnderstandingStarted", "label": "Understanding request\u2026"})
        log.info("Calling LLM for intent generation...")
        intent_json = await self.gemma.generate_intent(
            user_message=transcript,
            system_prompt=self.intent_prompt,
        )

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

        if complexity == TaskComplexity.AGENTIC_GOAL:
            log.info(f"Routing goal '{transcript}' to Agentic ReAct Loop...")
            if intent and intent.tool in ("open_app", "cua_launch"):
                log.info(f"Pre-executing setup intent before AgentLoop: {intent.tool}({intent.params})")
                await self.dispatcher.execute(intent)
                await asyncio.sleep(2.0)

            agent_res = await self.agent_loop.run(user_goal=transcript)

            log.info(f"AgentLoop Result: {agent_res}")

            if agent_res.get("status") == "safety_confirmation_required":
                response_text = agent_res.get("confirmation_prompt")
            else:
                response_text = agent_res.get("final_response") or "Goal executed."
            if response_text:
                await self.response_gen.generate_and_speak_with_text(response_text)
        elif intent.is_conversation or complexity == TaskComplexity.CONVERSATION:
            # Use the response from intent generation directly (already generated by the intent LLM call)
            response_text = intent.response if intent and intent.response else "I'm here to help! What can I do for you today?"
            log.info(f"Speaking conversational response: '{response_text[:100]}...'")
            await self.response_gen.generate_and_speak_with_text(response_text)
        else:
            log.info(f"Fast-path executing tool '{intent.tool}' with params={intent.params}")
            result = await self.dispatcher.execute(intent)
            log.info(f"Result: {result}")

            response_text = None
            if intent.is_conversation and intent.response:
                response_text = intent.response
            elif result.get("text"):
                response_text = result["text"]

            if response_text:
                log.info(f"Speaking tool result: '{response_text[:100]}...' ({len(response_text)} chars)")
                await self.response_gen.generate_and_speak_with_text(response_text)

        await self.ws_server.emit({"type": "ConversationFinished"})
        await self._run_followup_window(timeout_seconds=self.config.followup_timeout_seconds)

    async def _run_followup_window(self, timeout_seconds: int = 10):
        """Maintain active follow-up listening window for 10s without requiring wake word."""
        log.info(f"Entering active {timeout_seconds}s follow-up listening window...")
        await self.ws_server.emit({"type": "FollowupListeningStarted", "timeout": timeout_seconds})

        self.wake_word.pause()
        self.vad.reset()
        self.whisper.reset_buffer()

        import time
        start_time = time.time()
        audio_buffer = bytearray()

        while self._running and (time.time() - start_time < timeout_seconds):
            try:
                chunk = self.capture.get_chunk()
                audio_buffer.extend(chunk)

                if self.vad.process_chunk(chunk, self.capture):
                    log.info(f"Follow-up VAD natural speech boundary detected after {len(audio_buffer)} bytes")
                    raw_pcm = bytes(audio_buffer)
                    transcript = self.whisper.transcribe_bytes(raw_pcm)

                    if transcript.strip():
                        log.info(f"Follow-up transcribed: '{transcript}'")
                        await self.ws_server.emit({"type": "FinalTranscript", "text": transcript})

                        # Cancel any ongoing TTS output if user barged in
                        self.response_gen.tts_player.stop()

                        intent_json = await self.gemma.generate_intent(
                            user_message=transcript,
                            system_prompt=self.intent_prompt,
                        )
                        intent = None
                        if intent_json:
                            try:
                                intent = self.intent_parser.parse(intent_json)
                            except Exception:
                                pass

                        complexity = CapabilityRouter.classify(transcript, intent)
                        if complexity == TaskComplexity.AGENTIC_GOAL:
                            if intent and intent.tool in ("open_app", "cua_launch"):
                                log.info(f"Pre-executing setup intent before AgentLoop: {intent.tool}({intent.params})")
                                await self.dispatcher.execute(intent)
                                await asyncio.sleep(2.0)

                            agent_res = await self.agent_loop.run(user_goal=transcript)

                            response_text = agent_res.get("final_response") or "Goal executed."
                            if response_text:
                                await self.response_gen.generate_and_speak_with_text(response_text)
                            return await self._run_followup_window(timeout_seconds=timeout_seconds)
                        elif intent and not intent.is_conversation:
                            result = await self.dispatcher.execute(intent)
                            response_text = result.get("text")
                            if response_text:
                                await self.response_gen.generate_and_speak_with_text(response_text)
                            return await self._run_followup_window(timeout_seconds=timeout_seconds)
                        else:
                            if intent and intent.response:
                                await self.response_gen.generate_and_speak_with_text(intent.response)
                                return await self._run_followup_window(timeout_seconds=timeout_seconds)

                    audio_buffer.clear()
                    self.vad.reset()

                await asyncio.sleep(0.01)
            except Exception as e:
                log.error(f"Error in follow-up listening window: {e}")
                break

        log.info("Follow-up listening window timed out. Returning to idle.")
        self.wake_word.resume()
        self._flush_mic_buffer()
        await self.ws_server.emit({"type": "Idle"})

    def _flush_mic_buffer(self) -> None:
        """Drain microphone queue and Vosk queue to prevent audio feedback / stale wake word detection."""
        if self._audio_queue:
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except Exception:
                    break
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

        log.info("Shutdown: stopping audio capture...")
        self.capture.stop()
        self.capture.close()

        log.info("Shutdown: stopping wake word detector...")
        self.wake_word.stop()

        log.info("Shutdown: stopping TTS player...")
        self.tts_player.stop()

        log.info("Shutdown: stopping llama-server...")
        self._stop_llama_server()

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
