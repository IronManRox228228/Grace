"""Streaming response generator.

Manages the pipeline from Gemma text generation through Kokoro TTS
to audio playback. Implements sentence-level streaming so that
TTS synthesis and playback begin before the LLM finishes generating.
"""

import asyncio
import logging
import re
from typing import Optional, AsyncIterator

from grace.tts.kokoro_engine import KokoroEngine
from grace.tts.player import TTSPlayer
from grace.llm.gemma_client import GemmaClient
from grace.ws_server import WsEventServer
from grace.text.sentence_split import split_sentences
from grace.util.timing import mark_event, stage

logger = logging.getLogger("grace.response")

# Sentinel: the engine has no pipelined submit(), so synthesize inline instead.
_SYNTHESIZE_INLINE = object()


class ResponseGenerator:
    """Generates and speaks text responses with streaming overlap.

    Pipeline:
    Gemma (tokens) -> sentence boundary detection -> Kokoro worker -> playback queue

    Key property: TTS begins on the first complete sentence while
    Gemma is still generating subsequent sentences. This gives the
    perception of immediate responsiveness.
    """

    def __init__(
        self,
        gemma_client: GemmaClient,
        kokoro_engine: KokoroEngine,
        tts_player: TTSPlayer,
        ws_server: Optional[WsEventServer] = None,
    ):
        self._gemma = gemma_client
        self._kokoro = kokoro_engine
        self._player = tts_player
        self._ws = ws_server
        self._system_prompt: Optional[str] = None

    @property
    def tts_player(self) -> TTSPlayer:
        """Access the underlying TTS player."""
        return self._player

    def stop_speaking(self):
        """Immediately stop current TTS audio playback."""
        self._player.stop()

    def set_system_prompt(self, prompt: str):
        self._system_prompt = prompt

    async def generate_and_speak(
        self,
        user_message: str,
        voice: str = "af_bella",
    ) -> bool:
        """Generate a response to the user's message and speak it.

        Streams tokens from Gemma, detects completed sentences on the fly,
        and routes each sentence immediately to Kokoro for synthesis & playback.
        The first sentence begins playing while subsequent sentences are being generated.
        """
        if not self._system_prompt:
            logger.warning("No system prompt set for response generation")
            return False

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_message},
        ]

        logger.info("generate_and_speak: streaming tokens from LLM and synthesizing sentences...")
        token_stream = await self._gemma.chat(
            messages,
            temperature=0.8,
            max_tokens=2048,
            stream=True,
        )

        if token_stream is None:
            logger.warning("generate_and_speak: LLM token stream returned None")
            return False

        if self._ws:
            await self._ws.emit({"type": "SpeechStarted"})

        sentence_buffer = ""
        spoken_any = False
        sentence_count = 0

        async for token in token_stream:
            sentence_buffer += token

            # Cheap regex gate, then confirm with the abbreviation-aware
            # splitter so "Dr. Smith" is not cut in half mid-stream.
            if not re.search(r"(?<=[.!?])\s", sentence_buffer):
                continue

            parts = split_sentences(sentence_buffer)
            if len(parts) < 2:
                continue

            # The trailing part may still be mid-sentence, so hold it back.
            complete, sentence_buffer = parts[:-1], parts[-1]

            for complete_sentence in complete:
                sentence_count += 1
                logger.info(f"Streaming TTS sentence #{sentence_count}: '{complete_sentence[:60]}'")
                async with stage(f"synth#{sentence_count}"):
                    wav = await asyncio.to_thread(self._kokoro.synthesize, complete_sentence, voice)
                if wav:
                    self._player.play(wav)
                    mark_event("first_audio_out")
                    spoken_any = True
                    if self._ws:
                        chunk_text = f" {complete_sentence}" if sentence_count > 1 else complete_sentence
                        await self._ws.emit({"type": "ResponseChunk", "text": chunk_text})
                        await self._ws.emit({"type": "SpeechChunk"})

        final_sentence = sentence_buffer.strip()
        if final_sentence:
            sentence_count += 1
            logger.info(f"Streaming TTS final sentence #{sentence_count}: '{final_sentence[:60]}'")
            async with stage(f"synth#{sentence_count}"):
                wav = await asyncio.to_thread(self._kokoro.synthesize, final_sentence, voice)
            if wav:
                self._player.play(wav)
                mark_event("first_audio_out")
                spoken_any = True
                if self._ws:
                    chunk_text = f" {final_sentence}" if sentence_count > 1 else final_sentence
                    await self._ws.emit({"type": "ResponseChunk", "text": chunk_text})
                    await self._ws.emit({"type": "SpeechChunk"})

        # Wait for TTS player to finish playing audio out loud
        async with stage("playback_drain"):
            await asyncio.to_thread(self._player.wait)

        if self._ws:
            await self._ws.emit({"type": "SpeechFinished"})

        logger.info(f"Streaming TTS complete: {sentence_count} sentence(s) processed")
        return spoken_any

    async def generate_and_speak_with_text(
        self,
        text: str,
        voice: str = "af_bella",
    ) -> bool:
        """Speak pre-generated text (e.g., PDF content).

        Used for read_pdf and summarize_pdf responses.
        """
        logger.info(f"generate_and_speak_with_text: {len(text)} chars, {voice}")
        return await self._speak_text(text, voice)

    async def _speak_text(self, text: str, voice: str) -> bool:
        """Split text into sentences, synthesize, and play."""
        sentences = self._split_sentences(text)
        if not sentences:
            logger.warning("No sentences to speak")
            return False

        logger.info(f"Speaking {len(sentences)} sentences via {voice}")
        success = await self._synthesize_and_play(sentences, voice)
        return success

    async def _synthesize_and_play(
        self,
        sentences: list[str],
        voice: str,
    ) -> bool:
        """Synthesize sentences and queue for playback.

        Each sentence is sent to the Kokoro engine immediately.
        The TTS player queues them sequentially.
        """
        success = False
        if self._ws:
            await self._ws.emit({"type": "SpeechStarted"})

        # One-deep lookahead: sentence N+1 is queued on a second worker while
        # sentence N is still being resolved and handed to the player, so
        # synthesis hides behind playback instead of serialising after it.
        pending = self._submit(sentences[0], voice)

        for i, sentence in enumerate(sentences):
            logger.debug(f"TTS sentence {i+1}/{len(sentences)}: '{sentence[:80]}'")

            next_pending = (
                self._submit(sentences[i + 1], voice) if i + 1 < len(sentences) else None
            )

            async with stage(f"synth#{i+1}") as synth_stage:
                wav = await asyncio.to_thread(self._resolve, pending, sentence, voice)
                synth_stage.detail(f"{len(wav) if wav else 0} bytes")
            pending = next_pending

            if wav:
                self._player.play(wav)
                # Time-to-first-audio is the number that matters most to the
                # user: how long after they stopped speaking Grace starts to.
                mark_event("first_audio_out")
                success = True
                logger.debug(f"TTS sentence {i+1}/{len(sentences)} synthesized ({len(wav)} bytes)")
                if self._ws:
                    chunk_text = f" {sentence}" if i > 0 else sentence
                    await self._ws.emit({"type": "ResponseChunk", "text": chunk_text})
                    await self._ws.emit({"type": "SpeechChunk"})
            else:
                logger.warning(f"TTS sentence {i+1}/{len(sentences)} FAILED: '{sentence[:50]}'")

        # Wait for TTS player to finish playing audio out loud
        async with stage("playback_drain"):
            await asyncio.to_thread(self._player.wait)

        if self._ws:
            await self._ws.emit({"type": "SpeechFinished"})

        logger.info(f"TTS complete: {sum(1 for s in sentences if s)}/{len(sentences)} sentences spoken")
        return success

    def _submit(self, sentence: str, voice: str):
        """Queue a sentence for synthesis without blocking.

        Falls back to a marker for engines that predate the pipelined API (or
        test doubles), so behaviour degrades to the old synchronous path rather
        than breaking.
        """
        submit = getattr(self._kokoro, "submit", None)
        if submit is None:
            return _SYNTHESIZE_INLINE
        try:
            return submit(sentence, voice)
        except Exception as e:
            logger.debug(f"Kokoro submit unavailable ({e}); falling back to inline synthesis")
            return _SYNTHESIZE_INLINE

    def _resolve(self, pending, sentence: str, voice: str) -> Optional[bytes]:
        """Turn whatever _submit produced into WAV bytes. Runs off the event loop."""
        if pending is _SYNTHESIZE_INLINE:
            return self._kokoro.synthesize(sentence, voice)
        resolve = getattr(self._kokoro, "resolve", None)
        if resolve is None:
            return self._kokoro.synthesize(sentence, voice)
        return resolve(pending, sentence, voice)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        return split_sentences(text)
