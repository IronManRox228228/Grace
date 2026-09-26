import asyncio
import json
import logging
import os
from typing import Optional, Callable

from aiohttp import web

from .harness import ContractViolation, get_recorder, validate_event

logger = logging.getLogger("grace.ws")

# Raise on a contract violation instead of logging it. On in dev and CI; off in
# production, where a malformed diagnostic event must never abort a user's turn.
_CONTRACT_STRICT = os.getenv("GRACE_CONTRACT_STRICT", "").lower() in ("1", "true", "yes")

# Origins the Tauri renderer actually connects from. WebView2 reports the
# packaged app's own page as one of the tauri.localhost/tauri://localhost
# forms depending on config; the dev server is the fixed port declared in both
# frontend/renderer/vite.config.ts and src-tauri/tauri.conf.json's devUrl.
# Anything else talking to this socket is not the renderer Grace ships with.
DEFAULT_ALLOWED_ORIGINS = frozenset({
    "http://tauri.localhost",
    "https://tauri.localhost",
    "tauri://localhost",
    "http://localhost:5173",
})


def _parse_extra_origins(raw: str) -> frozenset:
    """Split the comma-separated GRACE_WS_ALLOWED_ORIGINS override."""
    return frozenset(o.strip() for o in (raw or "").split(",") if o.strip())


def is_origin_allowed(origin: Optional[str], allowed: frozenset) -> bool:
    """Whether a WebSocket handshake with this Origin header should proceed.

    No Origin header at all is allowed: only browsers send one, so a
    non-browser local client (a test script, a health check) would otherwise
    be rejected for doing nothing wrong. A browser-origin connection from
    anywhere not in the allowlist is what this guards against - without it,
    any page open in the user's browser could open this socket, read live
    transcripts, and inject a fake wake event.
    """
    return origin is None or origin in allowed


class WsEventServer:
    """WebSocket server sending GraceEvent messages to the frontend.

    Accepts client connections and provides an ``emit()`` method
    that the rest of the backend pipeline calls to push events to the UI.
    Incoming messages from the frontend (e.g. ``{"type": "wake"}``) are
    forwarded through the *on_wake* callback.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, allowed_origins: str = ""):
        self._host = host
        self._port = port
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._clients: set[web.WebSocketResponse] = set()
        self._on_wake: Optional[Callable[[], None]] = None
        self._allowed_origins = DEFAULT_ALLOWED_ORIGINS | _parse_extra_origins(allowed_origins)

    @property
    def is_connected(self) -> bool:
        return len(self._clients) > 0

    def set_on_wake(self, callback: Callable[[], None]) -> None:
        self._on_wake = callback

    async def _handler(self, request: web.Request) -> web.WebSocketResponse:
        origin = request.headers.get("Origin")
        if not is_origin_allowed(origin, self._allowed_origins):
            logger.warning(f"Rejected WebSocket handshake from disallowed origin: {origin}")
            raise web.HTTPForbidden(text="Origin not allowed")

        ws = web.WebSocketResponse()
        try:
            await ws.prepare(request)
        except Exception:
            logger.debug("WebSocket handshake failed (client may have disconnected)")
            return ws

        self._clients.add(ws)
        logger.info(f"Frontend connected via WebSocket (active clients: {len(self._clients)})")

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        if data.get("type") == "wake" and self._on_wake:
                            self._on_wake()
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON from frontend WS")
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"WS error: {ws.exception()}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"WS handler error: {e}")
        finally:
            self._clients.discard(ws)
            logger.info(f"Frontend disconnected (active clients: {len(self._clients)})")

        return ws

    async def start(self) -> None:
        self._app = web.Application()
        self._app.router.add_get("/", self._handler)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        logger.info(f"WebSocket server listening on ws://{self._host}:{self._port}")

    async def stop(self) -> None:
        for ws in list(self._clients):
            if not ws.closed:
                await ws.close()
        self._clients.clear()

        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        logger.info("WebSocket server stopped")

    async def emit(self, event: dict) -> None:
        """Send a GraceEvent dict to all connected frontends.

        This is the single seam between the backend and the UI, so it is also
        where the frozen contract is enforced and where session tapes are
        recorded. Both taps run before the connected-clients check: an event is
        part of the contract whether or not anyone is listening, and a tape
        recorded with no frontend attached must still be complete.
        """
        event_type = event.get("type", "unknown")
        logger.debug(f"WS emit: {event_type}")

        try:
            validate_event(event)
        except ContractViolation as exc:
            if _CONTRACT_STRICT:
                raise
            logger.error(f"Contract violation on emit: {exc}")

        recorder = get_recorder()
        if recorder is not None:
            recorder.record_event(event)

        if not self._clients:
            return

        to_remove = set()
        for ws in list(self._clients):
            if ws.closed:
                to_remove.add(ws)
                continue
            try:
                await ws.send_json(event)
                logger.debug(f"WS sent: {event_type}")
            except Exception as e:
                logger.warning(f"Failed to send WS event ({event_type}): {e}")
                to_remove.add(ws)

        self._clients -= to_remove
