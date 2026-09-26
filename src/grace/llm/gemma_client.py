"""Client for Cloud Gemini 3.6 Flash API and local llama.cpp server.

Connects to Google Gemini API (gemini-3.6-flash) when API key is provided,
or falls back to a locally running llama-server instance.
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional, AsyncIterator, Any

import aiohttp

from ..harness import get_recorder

logger = logging.getLogger("grace.gemma")

# Retries for a 429. Kept small and explicit: the user's key is rate-limited
# quickly, and silently retrying forever just moves the stall somewhere else.
RATE_LIMIT_RETRIES = 2
RATE_LIMIT_BACKOFF_SECONDS = (1.0, 4.0)


def _redact_images(messages: list[dict]) -> list[dict]:
    """Copy *messages* with base64 image payloads replaced by their size.

    A grounding turn's screenshot is megabytes of base64. Keeping the length
    preserves the one property a replay cares about - that an image was present
    and how large it was - without making tapes unreadable or unstoreable.
    """
    redacted = []
    for message in messages:
        copy = dict(message)
        image = copy.get("image_b64")
        if image:
            copy["image_b64"] = f"<image {len(image)} b64 chars>"
        redacted.append(copy)
    return redacted


async def _record_stream(recorder, kind: str, request: Any, stream: AsyncIterator[str]):
    """Pass *stream* through unchanged while taping each chunk as it arrives.

    Chunks are recorded individually and never joined. response/generator.py
    splits sentences off the token stream *as it arrives*, so the emitted
    ResponseChunk/SpeechChunk pairs depend on where those boundaries land. A
    Rust SSE reader that buffers differently would produce identical final text
    but a different chunk sequence - a real UI and TTS regression that a joined
    recording would destroy the evidence of.
    """
    started = time.perf_counter()
    chunks: list[str] = []
    try:
        async for chunk in stream:
            chunks.append(chunk)
            yield chunk
    except Exception as exc:
        recorder.record_llm(
            kind=kind, request=request, chunks=chunks, error=f"{type(exc).__name__}: {exc}",
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
        raise
    else:
        recorder.record_llm(
            kind=kind, request=request, chunks=chunks,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )


class RateLimitError(RuntimeError):
    """Raised when the cloud model returns HTTP 429 after backoff.

    Surfaced rather than swallowed so the caller can tell the user their quota
    is exhausted, instead of silently degrading to a different model.
    """


class GemmaClient:
    """HTTP client supporting Cloud Gemini 3.6 Flash API and local llama-server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.1-flash-lite",
        before_request=None,
    ):
        self._base_url = base_url
        self._api_key = api_key
        self._model_name = model_name or "gemini-3.1-flash-lite"
        self._http = None
        # Called before every local request. Used to hot-swap the model
        # resident on the GPU when the planner and grounder share one card -
        # see grace.llm.model_swap. None (the default) means no swap is needed.
        self._before_request = before_request

    async def _prepare(self) -> None:
        if self._before_request is None:
            return
        result = self._before_request()
        if asyncio.iscoroutine(result):
            await result

    async def _get_http(self):
        if self._http is None or self._http.closed:
            connector = aiohttp.TCPConnector(limit=10, keepalive_timeout=60, enable_cleanup_closed=True)
            self._http = aiohttp.ClientSession(connector=connector)
        return self._http

    async def close(self):
        """Close the underlying HTTP session."""
        if self._http and not self._http.closed:
            await self._http.close()
            self._http = None

    async def health_check(self) -> bool:
        """Check if LLM backend (Gemini Cloud API or local llama-server) is reachable."""
        if self._api_key:
            return True
        try:
            http = await self._get_http()
            async with http.get(f"{self._base_url}/health") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 8192,
        stream: bool = True,
        model: Optional[str] = None,
    ):
        """Send a chat completion request.

        `model` overrides the configured one for this call only. It exists for
        the escalation ladder: when a goal is stuck, the last thing worth trying
        before giving up is a more capable planner, and paying for that on every
        step to have it available on the rare one would be the wrong trade. It
        applies to the Gemini backend; a local llama-server serves whatever
        model it was started with, so there is nothing to switch there.
        """
        await self._prepare()
        effective_model = model or self._model_name
        logger.info(f"LLM request ({effective_model if self._api_key else 'llama.cpp'}): {len(messages)} messages, max_tokens={max_tokens}, temperature={temperature}")

        recorder = get_recorder()
        request = None
        if recorder is not None:
            request = {
                "backend": "gemini" if self._api_key else "llama.cpp",
                "model": effective_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
                "messages": _redact_images(messages),
            }

        if self._api_key:
            gemini = self._stream_gemini_response(messages, temperature=temperature,
                                                  max_tokens=max_tokens, model=effective_model)
            if recorder is None:
                return gemini
            return _record_stream(recorder, "chat", request, gemini)

        formatted_messages = []
        for msg in messages:
            msg_copy = dict(msg)
            img_b64 = msg_copy.pop("image_b64", None)
            if img_b64 and not self._api_key:
                content_str = msg_copy.get("content", "")
                msg_copy["content"] = [
                    {"type": "text", "text": content_str},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ]
            formatted_messages.append(msg_copy)

        payload = {
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "cache_prompt": True,
        }


        if stream:
            local = self._stream_response(payload)
            if recorder is None:
                return local
            return _record_stream(recorder, "chat", request, local)
        else:
            started = time.perf_counter()
            http = await self._get_http()
            try:
                async with http.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        choices = data.get("choices", [])
                        content = None
                        if choices and isinstance(choices, list):
                            content = choices[0].get("message", {}).get("content")
                        if recorder is not None:
                            recorder.record_llm(
                                kind="chat", request=request, response=content,
                                duration_ms=(time.perf_counter() - started) * 1000.0,
                            )
                        return content
                    else:
                        logger.error(f"LLM request failed with status {resp.status}")
                        if recorder is not None:
                            recorder.record_llm(
                                kind="chat", request=request, error=f"HTTP {resp.status}",
                                duration_ms=(time.perf_counter() - started) * 1000.0,
                            )
                        return None
            except Exception as e:
                logger.error(f"LLM request error: {e}")
                if recorder is not None:
                    recorder.record_llm(
                        kind="chat", request=request, error=str(e),
                        duration_ms=(time.perf_counter() - started) * 1000.0,
                    )
                return None

    async def _stream_gemini_response(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream response tokens from the configured Gemini model.

        Exactly one model is called. The old fallback chain
        (gemini-2.0-flash-lite, gemini-1.5-flash-latest) turned a single
        rate-limited request into three requests against three separate
        quotas, which is the fastest possible way to exhaust all of them.
        """
        system_instruction = None
        contents = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = {"parts": [{"text": content}]}
            elif role == "user":
                parts = [{"text": content}]
                if msg.get("image_b64"):
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": msg["image_b64"]
                        }
                    })
                contents.append({"role": "user", "parts": parts})
            elif role in ("assistant", "model"):
                contents.append({"role": "model", "parts": [{"text": content}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_instruction:
            payload["system_instruction"] = system_instruction

        model = model or self._model_name
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
            f":streamGenerateContent?key={self._api_key}&alt=sse"
        )

        last_error = None
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                # Reuse the pooled session. Opening a fresh ClientSession per
                # call threw away the connection pool and paid a full TCP+TLS
                # handshake on every single turn.
                session = await self._get_http()
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status == 429:
                        body = await resp.text()
                        last_error = RateLimitError(
                            f"{model} rate limited (HTTP 429): {body[:150]}"
                        )
                        if attempt < RATE_LIMIT_RETRIES:
                            delay = RATE_LIMIT_BACKOFF_SECONDS[
                                min(attempt, len(RATE_LIMIT_BACKOFF_SECONDS) - 1)
                            ]
                            logger.warning(
                                f"Gemini '{model}' rate limited; retrying in {delay:.0f}s "
                                f"({attempt + 1}/{RATE_LIMIT_RETRIES})"
                            )
                            await asyncio.sleep(delay)
                            continue
                        raise last_error

                    if resp.status != 200:
                        body = await resp.text()
                        logger.error(f"Gemini '{model}' failed with status {resp.status}: {body[:150]}")
                        raise RuntimeError(f"HTTP {resp.status}: {body[:100]}")

                    async for line_bytes in resp.content:
                        line_str = line_bytes.decode("utf-8", errors="replace").strip()
                        if line_str.startswith("data: "):
                            data_json = line_str[6:].strip()
                            try:
                                chunk = json.loads(data_json)
                                candidates = chunk.get("candidates", [])
                                if candidates and isinstance(candidates, list):
                                    parts = candidates[0].get("content", {}).get("parts", [])
                                    for part in parts:
                                        text = part.get("text")
                                        if text:
                                            yield text
                            except Exception as ex:
                                logger.debug(f"Failed to decode Gemini stream chunk: {ex}")
                    return
            except RateLimitError:
                raise
            except aiohttp.ClientError as e:
                logger.warning(f"Gemini streaming transport error for '{model}': {e}")
                last_error = e
                if attempt >= RATE_LIMIT_RETRIES:
                    break
                await asyncio.sleep(RATE_LIMIT_BACKOFF_SECONDS[0])

        if last_error:
            raise last_error

    async def _stream_response(self, payload: dict) -> AsyncIterator[str]:
        """Parse SSE stream and yield token chunks from local llama-server."""
        try:
            # Pooled session, same reasoning as the Gemini path.
            session = await self._get_http()
            async with session.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"LLM SSE stream failed with status {resp.status}: {body}")
                    raise RuntimeError(f"LLM request failed: HTTP {resp.status}: {body}")

                async for line_bytes in resp.content:
                    line_str = line_bytes.decode("utf-8", errors="replace").strip()
                    if not line_str or line_str.startswith(":"):
                        continue
                    if line_str == "data: [DONE]":
                        break
                    if line_str.startswith("data: "):
                        data_json = line_str[6:].strip()
                        try:
                            chunk_data = json.loads(data_json)
                            choices = chunk_data.get("choices", [])
                            if choices and isinstance(choices, list):
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    yield content
                        except json.JSONDecodeError as ex:
                            logger.debug(f"Failed to decode LLM stream chunk: {ex}")
                            continue
        except Exception as e:
            logger.error(f"SSE streaming exception: {e}")
            raise

    async def generate_intent(self, user_message: str, system_prompt: str) -> Optional[str]:
        """Generate structured intent JSON for a user message."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        tokens = []
        try:
            stream = await self.chat(messages, temperature=0.1, max_tokens=4096, stream=True)
            if stream is not None:
                async for token in stream:
                    tokens.append(token)
        except RateLimitError:
            # Must reach the caller: quota exhaustion is something the user
            # needs told, not something to paper over with an empty result.
            raise
        except Exception as e:
            logger.error(f"Failed to generate intent: {e}")

        return "".join(tokens) if tokens else None

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        messages: Optional[list[dict]] = None,
        image_b64: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Optional[str]:
        """Generate text completion non-streaming."""
        if not messages:
            user_msg = {"role": "user", "content": prompt}
            if image_b64:
                user_msg["image_b64"] = image_b64
            messages = [
                {"role": "system", "content": system_prompt},
                user_msg,
            ]
        tokens = []
        try:
            stream = await self.chat(messages, temperature=temperature, max_tokens=max_tokens,
                                     stream=True, model=model)
            if stream is not None:
                async for token in stream:
                    tokens.append(token)
        except RateLimitError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")

        return "".join(tokens) if tokens else None