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

logger = logging.getLogger("grace.response")


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

            m = re.search(r"(?<=[.!?])\s+", sentence_buffer)
            if m:
                split_idx = m.end()
                complete_sentence = sentence_buffer[:split_idx].strip()
                sentence_buffer = sentence_buffer[split_idx:]

                if complete_sentence:
                    sentence_count += 1
                    logger.info(f"Streaming TTS sentence #{sentence_count}: '{complete_sentence[:60]}'")
                    logger.debug(f"[Boundary] Entering asyncio.to_thread(kokoro.synthesize) for sentence #{sentence_count}")
                    wav = await asyncio.to_thread(self._kokoro.synthesize, complete_sentence, voice)
                    logger.debug(f"[Boundary] Returned from kokoro.synthesize for sentence #{sentence_count} ({len(wav) if wav else 0} bytes)")
                    if wav:
                        logger.debug(f"[Boundary] Entering tts_player.play for sentence #{sentence_count}")
                        self._player.play(wav)
                        spoken_any = True
                        if self._ws:
                            chunk_text = f" {complete_sentence}" if sentence_count > 1 else complete_sentence
                            await self._ws.emit({"type": "ResponseChunk", "text": chunk_text})
                            await self._ws.emit({"type": "SpeechChunk"})
                            logger.debug(f"[Boundary] Emitted ResponseChunk to WS for sentence #{sentence_count}")

        final_sentence = sentence_buffer.strip()
        if final_sentence:
            sentence_count += 1
            logger.info(f"Streaming TTS final sentence #{sentence_count}: '{final_sentence[:60]}'")
            logger.debug(f"[Boundary] Entering asyncio.to_thread(kokoro.synthesize) for final sentence #{sentence_count}")
            wav = await asyncio.to_thread(self._kokoro.synthesize, final_sentence, voice)
            logger.debug(f"[Boundary] Returned from kokoro.synthesize for final sentence #{sentence_count} ({len(wav) if wav else 0} bytes)")
            if wav:
                logger.debug(f"[Boundary] Entering tts_player.play for final sentence #{sentence_count}")
                self._player.play(wav)
                spoken_any = True
                if self._ws:
                    chunk_text = f" {final_sentence}" if sentence_count > 1 else final_sentence
                    await self._ws.emit({"type": "ResponseChunk", "text": chunk_text})
                    await self._ws.emit({"type": "SpeechChunk"})
                    logger.debug(f"[Boundary] Emitted ResponseChunk to WS for final sentence #{sentence_count}")

        # Wait for TTS player to finish playing audio out loud
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

        for i, sentence in enumerate(sentences):
            logger.debug(f"TTS sentence {i+1}/{len(sentences)}: '{sentence[:80]}'")
            logger.debug(f"[Boundary] Entering asyncio.to_thread(kokoro.synthesize) for sentence {i+1}/{len(sentences)}")
            wav = await asyncio.to_thread(self._kokoro.synthesize, sentence, voice)
            logger.debug(f"[Boundary] Returned from kokoro.synthesize for sentence {i+1}/{len(sentences)} ({len(wav) if wav else 0} bytes)")
            if wav:
                logger.debug(f"[Boundary] Entering tts_player.play for sentence {i+1}/{len(sentences)}")
                self._player.play(wav)
                success = True
                logger.debug(f"TTS sentence {i+1}/{len(sentences)} synthesized ({len(wav)} bytes)")
                if self._ws:
                    chunk_text = f" {sentence}" if i > 0 else sentence
                    await self._ws.emit({"type": "ResponseChunk", "text": chunk_text})
                    await self._ws.emit({"type": "SpeechChunk"})
                    logger.debug(f"[Boundary] Emitted ResponseChunk to WS for sentence {i+1}/{len(sentences)}")
            else:
                logger.warning(f"TTS sentence {i+1}/{len(sentences)} FAILED: '{sentence[:50]}'")

        # Wait for TTS player to finish playing audio out loud
        await asyncio.to_thread(self._player.wait)

        if self._ws:
            await self._ws.emit({"type": "SpeechFinished"})

        logger.info(f"TTS complete: {sum(1 for s in sentences if s)}/{len(sentences)} sentences spoken")
        return success

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        return split_sentences(text)
