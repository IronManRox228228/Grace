import asyncio
import json
import logging
from typing import Optional, Callable

from aiohttp import web

logger = logging.getLogger("grace.ws")


class WsEventServer:
    """WebSocket server sending GraceEvent messages to the frontend.

    Accepts client connections and provides an ``emit()`` method
    that the rest of the backend pipeline calls to push events to the UI.
    Incoming messages from the frontend (e.g. ``{"type": "wake"}``) are
    forwarded through the *on_wake* callback.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self._host = host
        self._port = port
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._clients: set[web.WebSocketResponse] = set()
        self._on_wake: Optional[Callable[[], None]] = None

    @property
    def is_connected(self) -> bool:
        return len(self._clients) > 0

    def set_on_wake(self, callback: Callable[[], None]) -> None:
        self._on_wake = callback

    async def _handler(self, request: web.Request) -> web.WebSocketResponse:
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
        """Send a GraceEvent dict to all connected frontends."""
        event_type = event.get("type", "unknown")
        logger.debug(f"WS emit: {event_type}")
        if not self._clients:
            return

        to_remove = set()
        for ws in self._clients:
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
