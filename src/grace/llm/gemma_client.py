"""Client for Cloud Gemini 3.6 Flash API and local llama.cpp server.

Connects to Google Gemini API (gemini-3.6-flash) when API key is provided,
or falls back to a locally running llama-server instance.
"""

import asyncio
import json
import logging
import os
from typing import Optional, AsyncIterator, Any

import aiohttp

logger = logging.getLogger("grace.gemma")


class GemmaClient:
    """HTTP client supporting Cloud Gemini 3.6 Flash API and local llama-server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.1-flash-lite",
    ):
        self._base_url = base_url
        self._api_key = api_key
        self._model_name = model_name or "gemini-3.1-flash-lite"
        self._http = None

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
        max_tokens: int = 1024,
        stream: bool = True,
    ):
        """Send a chat completion request."""
        logger.info(f"LLM request ({self._model_name if self._api_key else 'llama.cpp'}): {len(messages)} messages, max_tokens={max_tokens}, temperature={temperature}")

        if self._api_key:
            return self._stream_gemini_response(messages, temperature=temperature, max_tokens=max_tokens)

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
            return self._stream_response(payload)
        else:
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
                        if choices and isinstance(choices, list):
                            return choices[0].get("message", {}).get("content")
                        return None
                    else:
                        logger.error(f"LLM request failed with status {resp.status}")
                        return None
            except Exception as e:
                logger.error(f"LLM request error: {e}")
                return None

    async def _stream_gemini_response(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Stream response tokens from Google Gemini REST API with automatic model fallback."""
        candidate_models = [self._model_name]
        for fallback in ["gemini-3.1-flash-lite", "gemini-2.0-flash-lite", "gemini-1.5-flash-latest"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

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

        last_error = None
        for model in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?key={self._api_key}&alt=sse"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                        if resp.status != 200:
                            body = await resp.text()
                            logger.warning(f"Gemini model '{model}' failed with status {resp.status}: {body[:150]}")
                            last_error = RuntimeError(f"HTTP {resp.status}: {body[:100]}")
                            if resp.status == 429:
                                await asyncio.sleep(1.0)
                            continue

                        self._model_name = model  # Remember working model
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
                                except Exception:
                                    pass
                        return  # Successfully streamed content
            except Exception as e:
                logger.warning(f"Gemini streaming exception for model '{model}': {e}")
                last_error = e

        if last_error:
            raise last_error

    async def _stream_response(self, payload: dict) -> AsyncIterator[str]:
        """Parse SSE stream and yield token chunks from local llama-server."""
        try:
            async with aiohttp.ClientSession() as session:
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
                            except json.JSONDecodeError:
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
            stream = await self.chat(messages, temperature=0.1, max_tokens=256, stream=True)
            if stream is not None:
                async for token in stream:
                    tokens.append(token)
        except Exception as e:
            logger.error(f"Failed to generate intent: {e}")

        return "".join(tokens) if tokens else None

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        messages: Optional[list[dict]] = None,
        image_b64: Optional[str] = None,
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
            stream = await self.chat(messages, temperature=temperature, max_tokens=max_tokens, stream=True)
            if stream is not None:
                async for token in stream:
                    tokens.append(token)
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")

        return "".join(tokens) if tokens else None