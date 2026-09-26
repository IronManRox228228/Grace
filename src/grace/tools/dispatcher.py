"""Intent dispatcher - routes parsed intents to CUA or hardcoded tools.

The dispatcher validates intents and delegates execution to the
appropriate tool handler. CUA tools go through the local ComputerUse
backend, system tools execute directly via Python Windows APIs.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Optional

from grace.agent.safety import SafetyGuard
from grace.intent.parser import Intent
from grace.automation.computer_use import ComputerUse
from grace.harness import get_recorder
from grace.ws_server import WsEventServer

logger = logging.getLogger("grace.dispatcher")

_TOOL_LABELS: dict[str, str] = {
    "open_app": "Opening application\u2026",
    "close_app": "Closing application\u2026",
    "search_files": "Searching files\u2026",
    "open_file": "Opening file\u2026",
    "read_pdf": "Reading document\u2026",
    "summarize_pdf": "Summarizing document\u2026",
    "adjust_volume": "Adjusting volume\u2026",
    "lock_computer": "Locking computer\u2026",
    "open_calculator": "Opening calculator\u2026",
    "delete_file": "Moving file to recycle bin\u2026",
    "undo": "Undoing action\u2026",
    "describe_screen": "Inspecting screen\u2026",
    "set_speech_rate": "Adjusting speech rate\u2026",
}

# One entry per handler ComputerUse actually implements. The removed keys
# (double_click, right_click, move_mouse, focus_window, get_cursor_position)
# had no corresponding _<name> method, so they could never be dispatched.
_CUA_LABELS: dict[str, str] = {
    "click": "Clicking\u2026",
    "type_text": "Typing\u2026",
    "press_key": "Pressing key\u2026",
    "drag": "Dragging\u2026",
    "scroll": "Scrolling\u2026",
    "screenshot": "Taking screenshot\u2026",
    "list_windows": "Listing windows\u2026",
    "list_apps": "Listing applications\u2026",
    "get_window": "Reading window\u2026",
    "activate": "Switching window\u2026",
    "launch": "Launching application\u2026",
    "text": "Reading screen\u2026",
    "set_value": "Setting value\u2026",
    "secondary_action": "Right-clicking\u2026",
}


_APP_ALIASES: dict[str, str] = {
    "edge": "msedge",
    "microsoft edge": "msedge",
    "browser": "msedge",
    "my browser": "msedge",
    "chrome": "chrome",
    "google chrome": "chrome",
    "firefox": "firefox",
    "notepad": "notepad",
    "calculator": "calc",
    "calc": "calc",
    "word": "winword",
    "excel": "excel",
    "powerpoint": "powerpnt",
    "paint": "mspaint",
    "cmd": "cmd",
    "command prompt": "cmd",
    "terminal": "wt",
    "windows terminal": "wt",
    "explorer": "explorer",
    "file explorer": "explorer",
    "my computer": "explorer",
    "settings": "ms-settings:",
    "vlc": "vlc",
    "spotify": "spotify",
}


_WEBSITE_ALIASES: dict[str, str] = {
    "youtube": "https://www.youtube.com",
    "youtube.com": "https://www.youtube.com",
    "google": "https://www.google.com",
    "google.com": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "reddit": "https://www.reddit.com",
    "reddit.com": "https://www.reddit.com",
    "github": "https://github.com",
    "github.com": "https://github.com",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "x.com": "https://x.com",
    "amazon": "https://www.amazon.com",
    "amazon.com": "https://www.amazon.com",
    "wikipedia": "https://www.wikipedia.org",
    "netflix": "https://www.netflix.com",
    "chatgpt": "https://chatgpt.com",
}


class Dispatcher:
    """Routes intents to CUA or hardcoded tool implementations."""

    def __init__(
        self,
        computer_use: Optional[ComputerUse] = None,
        ws_server: Optional[WsEventServer] = None,
        kokoro_engine: Optional[Any] = None,
    ):
        self._cua = computer_use
        self._ws = ws_server
        self._kokoro_engine = kokoro_engine
        self._last_search_results: list[str] = []
        self._app_indexer = None
        self._app_indexer_lock = asyncio.Lock()

    def set_kokoro_engine(self, engine: Any) -> None:
        """Install Kokoro engine for runtime speech parameter adjustments."""
        self._kokoro_engine = engine

    def set_app_indexer(self, indexer) -> None:
        """Install a pre-built index (main.py builds one during startup warmup)."""
        self._app_indexer = indexer

    async def execute(self, intent: Intent, confirmed: bool = False) -> dict[str, Any]:
        """Execute a parsed intent, taping the call when recording is on.

        This is the outermost side-effecting boundary in the backend. In a
        replay it is asserted rather than performed: what must match is that
        the Rust port asked for the same tool with the same parameters, in the
        same order - "Grace did the right thing on screen" is not enough if it
        took a different route to get there.

        Being the outermost boundary is also why the safety check belongs here.
        It used to live inside `AgentLoop`, which meant it covered the agentic
        path and nothing else - and `delete_file`, `close_app` and
        `lock_computer` are all in `CapabilityRouter.FAST_PATH_TOOLS`, so
        "delete my tax return" reached the desktop unconfirmed precisely because
        it was simple enough to route directly. `AgentLoop` still asks first, so
        it can park the whole step and resume it; this is the backstop that
        makes the guarantee independent of which path was taken.

        `confirmed=True` is the caller saying the user has already answered yes.
        """
        refusal = self.confirmation_required(intent, confirmed)
        if refusal is not None:
            return refusal

        recorder = get_recorder()
        result = await self._execute(intent)
        if recorder is not None:
            recorder.record_dispatch(intent.tool, intent.params, result)
        return result

    @staticmethod
    def confirmation_required(intent: Intent, confirmed: bool = False) -> Optional[dict[str, Any]]:
        """The refusal to return, or None to go ahead.

        Split out of `execute` so the replay harness can apply the same decision
        without a second copy of it. `TapeDispatcher` replaces `execute`
        wholesale, so a guard living only inside that method would be silently
        absent from every replayed run - and a safety rule that the regression
        oracle cannot see is a safety rule that can be deleted without any test
        going red.
        """
        if confirmed:
            return None
        is_safe, prompt = SafetyGuard.evaluate(intent.tool, intent.params)
        if is_safe:
            return None

        logger.warning(f"Dispatcher: '{intent.tool}' needs confirmation before it runs")
        return {
            "status": "confirmation_required",
            "confirmation_prompt": prompt,
            "text": prompt,
            "tool": intent.tool,
            "params": intent.params,
        }

    async def _execute(self, intent: Intent) -> dict[str, Any]:
        """Route to CUA for cua_* tools, or to a hardcoded system tool."""
        tool = intent.tool
        params = intent.params

        if tool.startswith("cua_"):
            return await self._execute_cua(tool, params)

        handler_map = {
            "converse": self._converse,
            "open_app": self._open_app,
            "close_app": self._close_app,
            "search_files": self._search_files,
            "open_file": self._open_file,
            "read_pdf": self._read_pdf,
            "summarize_pdf": self._summarize_pdf,
            "adjust_volume": self._adjust_volume,
            "lock_computer": self._lock_computer,
            "open_calculator": self._open_calculator,
            "delete_file": self._delete_file,
            "undo": self._undo,
            "describe_screen": self._describe_screen,
            "set_speech_rate": self._set_speech_rate,
        }

        handler = handler_map.get(tool)
        if handler:
            try:
                await self._emit_tool_started(tool, params)
                result = await handler(params)
                await self._emit_tool_finished()
                return result
            except Exception as e:
                logger.error(f"Tool {tool} failed: {e}")
                await self._emit_tool_finished()
                return {"status": "error", "error": str(e), "text": f"Sorry, I couldn't {tool}. {e}"}

        return {"status": "error", "error": f"Unknown tool: {tool}"}

    async def _emit_tool_started(self, tool: str, params: dict) -> None:
        if not self._ws:
            return
        label = _TOOL_LABELS.get(tool, f"{tool}\u2026")
        if tool == "open_app":
            name = params.get("name", "")
            if name:
                label = f"Opening {name}\u2026"
        elif tool == "close_app":
            name = params.get("name", "")
            if name:
                label = f"Closing {name}\u2026"
        elif tool == "search_files":
            query = params.get("query", "")
            if query:
                label = f"Searching for {query}\u2026"
        elif tool == "open_file":
            name = params.get("name", "")
            if name:
                label = f"Opening {name}\u2026"
        elif tool == "delete_file":
            name = params.get("name", "")
            if name:
                label = f"Moving {name} to recycle bin\u2026"
        await self._ws.emit({"type": "ToolExecutionStarted", "label": label})

    async def _emit_tool_finished(self) -> None:
        if self._ws:
            await self._ws.emit({"type": "ToolExecutionFinished"})

    async def _execute_cua(self, tool: str, params: dict) -> dict:
        """Execute a CUA tool via the local ComputerUse backend."""
        if not self._cua or not self._cua.is_ready:
            return {"status": "error", "error": "Computer use not available"}

        action = tool[4:]
        await self._emit_cua_started(action)
        try:
            result = await asyncio.to_thread(self._cua.perform, action, params)
            await self._emit_tool_finished()
            if result and result.get("error"):
                # The handler's own envelope is carried through, not replaced.
                # Flattening it to `{status, error, text}` threw away `status:
                # "wrong_focus"`, `status: "element_not_found"` and every A2
                # field - so a precondition that refused precisely, naming both
                # the window it wanted and the one in front, reached the planner
                # as an undifferentiated "Action error". Everything that could
                # be acted on was in the part discarded.
                return {
                    "status": "error",
                    "error": result["error"],
                    "text": f"Action error: {result['error']}",
                    "result": result,
                }
            return {"status": "ok", "result": result}
        except Exception as e:
            await self._emit_tool_finished()
            return {"status": "error", "error": str(e), "text": f"Action error: {e}"}

    async def _emit_cua_started(self, action: str) -> None:
        if not self._ws:
            return
        label = _CUA_LABELS.get(action, f"{action}\u2026")
        await self._ws.emit({"type": "ToolExecutionStarted", "label": label})

    async def _converse(self, params: dict) -> dict:
        response = params.get("response", "")
        return {"status": "ok", "text": response}

    async def _open_app(self, params: dict) -> dict:
        name = params.get("name", "").strip()
        url = params.get("url", "").strip()
        if not name and not url:
            return {"status": "error", "error": "Missing 'name' or 'url' parameter"}

        # 1. Check if url or name is a website/URL
        target_url = url
        lower_name = name.lower()
        if not target_url:
            if lower_name in _WEBSITE_ALIASES:
                target_url = _WEBSITE_ALIASES[lower_name]
            elif lower_name.startswith(("http://", "https://", "www.")) or any(
                ext in lower_name for ext in [".com", ".org", ".net", ".io", ".edu", ".gov"]
            ):
                target_url = name if name.startswith("http") else f"https://{name}"

        if target_url:
            import webbrowser
            webbrowser.open(target_url)
            display_name = name or target_url
            return {"status": "ok", "text": f"I've opened {display_name} in your browser."}

        # 2. Launch installed application via AppIndexer. Building the index
        # globs Program Files recursively, so it must never run on the loop.
        # The lock stops two concurrent "open X" requests building it twice.
        if self._app_indexer is None:
            async with self._app_indexer_lock:
                if self._app_indexer is None:
                    from grace.automation.app_indexer import AppIndexer

                    self._app_indexer = await asyncio.to_thread(AppIndexer)

        return await asyncio.to_thread(self._app_indexer.launch, name)

    async def _close_app(self, params: dict) -> dict:
        name = params.get("name", "")
        if not name:
            return {"status": "error", "error": "Missing 'name' parameter"}

        try:
            # taskkill blocks for up to 10s; keep it off the event loop.
            result = await asyncio.to_thread(
                subprocess.run,
                ["taskkill", "/F", "/IM", f"{name}.exe"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return {"status": "ok", "text": f"I've closed {name}."}
            else:
                result2 = await asyncio.to_thread(
                    subprocess.run,
                    ["taskkill", "/F", "/FI", f"WINDOWTITLE eq {name}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result2.returncode == 0:
                    return {"status": "ok", "text": f"I've closed {name}."}
                return {"status": "error", "error": result2.stderr, "text": f"I couldn't find {name} to close it."}
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "Timed out closing app", "text": "I couldn't close that in time."}
        except Exception as e:
            return {"status": "error", "error": str(e), "text": f"I couldn't close {name}. {e}"}

    async def _search_files(self, params: dict) -> dict:
        query = params.get("query", "")
        if not query:
            return {"status": "error", "error": "Missing 'query' parameter"}

        # Sanitize: strip shell-dangerous / control characters
        query = re.sub(r'[\"\'`;$|&<>(){}!\[\]\\]', '', query).strip()
        if not query:
            return {"status": "error", "error": "Invalid query after sanitization"}

        self._last_search_results = []

        try:
            user_profile = os.environ.get("USERPROFILE", os.path.expanduser("~"))
            docs_dir = os.path.join(user_profile, "Documents")
            extensions = {'.pdf', '.docx', '.txt', '.xlsx'}
            q_lower = query.lower()

            def _search():
                if not os.path.exists(docs_dir):
                    return []
                matches = []
                for p in Path(docs_dir).rglob("*"):
                    try:
                        if p.suffix.lower() in extensions and q_lower in p.name.lower():
                            matches.append(str(p))
                            if len(matches) >= 10:
                                break
                    except (PermissionError, OSError):
                        continue
                return matches

            self._last_search_results = await asyncio.to_thread(_search)

            if self._last_search_results:
                result_list = "\n".join(f"  {i+1}. {path}" for i, path in enumerate(self._last_search_results))
                return {
                    "status": "ok",
                    "text": f"Here are the matching files:\n{result_list}",
                    "files": self._last_search_results,
                }
            else:
                return {
                    "status": "ok",
                    "text": f"I couldn't find any files matching '{query}'.",
                    "files": [],
                }
        except Exception as e:
            return {"status": "error", "error": str(e), "text": f"Search failed: {e}"}

    async def _open_file(self, params: dict) -> dict:
        name = params.get("name", "")
        if not name:
            return {"status": "error", "error": "Missing 'name' parameter"}

        try:
            if name in self._last_search_results:
                os.startfile(name)
                return {"status": "ok", "text": f"I've opened {name}."}

            import glob

            user_profile = os.environ.get("USERPROFILE", os.path.expanduser("~"))
            search_dirs = [
                os.path.join(user_profile, "Documents"),
                os.path.join(user_profile, "Desktop"),
                os.path.join(user_profile, "Downloads"),
            ]

            for search_dir in search_dirs:
                matches = glob.glob(os.path.join(search_dir, f"*{name}*"))
                if matches:
                    path = matches[0]
                    os.startfile(path)
                    return {"status": "ok", "text": f"I've opened {name}."}

            os.startfile(name)
            return {"status": "ok", "text": f"I've opened {name}."}
        except Exception as e:
            return {"status": "error", "error": str(e), "text": f"I couldn't open {name}. {e}"}

    async def _read_pdf(self, params: dict) -> dict:
        path = params.get("path", "")
        query = params.get("query", "")
        if not path:
            return {"status": "error", "error": "Missing 'path' parameter"}

        def _do_read():
            from pypdf import PdfReader
            from grace.rag import LocalRagIndex

            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"

            if not text.strip():
                return None, "No extractable text in PDF"

            index = LocalRagIndex()
            doc_id = os.path.basename(path)
            index.index_text(doc_id, text)

            if query:
                chunks = index.query(query, top_k=3)
                rag_text = "\n\n".join(f"[Excerpt {c['index']+1}]: {c['chunk']}" for c in chunks)
            else:
                chunks = index.get_summary_chunks(top_k=3)
                rag_text = "\n\n".join(f"[Excerpt {i+1}]: {c}" for i, c in enumerate(chunks))

            return rag_text or text.strip()[:2000], None

        try:
            rag_text, err = await asyncio.to_thread(_do_read)
            if err:
                return {"status": "error", "error": err, "text": "The PDF doesn't contain extractable text."}

            return {
                "status": "ok",
                "text": rag_text,
                "action": "read_pdf",
            }
        except FileNotFoundError:
            return {"status": "error", "error": f"File not found: {path}", "text": f"I couldn't find that PDF."}
        except Exception as e:
            return {"status": "error", "error": str(e), "text": f"Error reading PDF: {e}"}

    async def _summarize_pdf(self, params: dict) -> dict:
        path = params.get("path", "")
        if not path:
            return {"status": "error", "error": "Missing 'path' parameter"}

        def _do_summarize():
            from pypdf import PdfReader
            from grace.rag import LocalRagIndex

            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"

            if not text.strip():
                return None, "No extractable text in PDF"

            index = LocalRagIndex()
            doc_id = os.path.basename(path)
            index.index_text(doc_id, text)

            chunks = index.get_summary_chunks(top_k=4)
            summary_context = "\n\n".join(f"[Section {i+1}]: {c}" for i, c in enumerate(chunks))
            return summary_context, None

        try:
            summary_context, err = await asyncio.to_thread(_do_summarize)
            if err:
                return {"status": "error", "error": err, "text": "The PDF doesn't contain extractable text."}

            return {
                "status": "ok",
                "text": summary_context,
                "action": "summarize_pdf",
            }
        except FileNotFoundError:
            return {"status": "error", "error": f"File not found: {path}", "text": f"I couldn't find that PDF."}
        except Exception as e:
            return {"status": "error", "error": str(e), "text": f"Error summarizing PDF: {e}"}

    async def _adjust_volume(self, params: dict) -> dict:
        amount = params.get("amount", 0)
        mode = params.get("mode", "increase")

        def _do_adjust():
            try:
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                from comtypes import CLSCTX_ALL
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume_obj = interface.QueryInterface(IAudioEndpointVolume)
                current_vol = volume_obj.GetMasterVolumeLevelScalar()
                if mode == "increase":
                    new_vol = min(1.0, current_vol + (amount / 100))
                elif mode == "decrease":
                    new_vol = max(0.0, current_vol - (amount / 100))
                elif mode in ("set", "percent"):
                    new_vol = max(0.0, min(1.0, amount / 100))
                else:
                    new_vol = current_vol

                volume_obj.SetMasterVolumeLevelScalar(new_vol, None)
                new_percent = int(new_vol * 100)
                return {
                    "status": "ok",
                    "text": f"Volume set to {new_percent} percent.",
                    "volume": new_percent,
                }
            except Exception as ex:
                logger.warning(f"pycaw volume adjustment fallback: {ex}")
                return {
                    "status": "ok",
                    "text": f"Volume {mode} by {amount}.",
                    "volume": amount,
                }

        try:
            return await asyncio.to_thread(_do_adjust)
        except Exception as e:
            return {"status": "error", "error": str(e), "text": f"Could not adjust volume: {e}"}

    async def _lock_computer(self, params: dict) -> dict:
        try:
            ctypes = __import__("ctypes")
            ctypes.windll.user32.LockWorkStation()
            return {"status": "ok", "text": "I've locked your computer."}
        except Exception as e:
            return {"status": "error", "error": str(e), "text": "I couldn't lock your computer."}

    async def _open_calculator(self, params: dict) -> dict:
        try:
            os.startfile("calc")
            return {"status": "ok", "text": "I've opened the calculator."}
        except Exception as e:
            return {"status": "error", "error": str(e), "text": "I couldn't open the calculator."}

    async def _delete_file(self, params: dict) -> dict:
        name = params.get("name", "")
        if not name:
            return {"status": "error", "error": "Missing 'name' parameter"}

        def _do_delete():
            user_profile = os.environ.get("USERPROFILE", os.path.expanduser("~"))
            paths_to_try = [
                name,
                os.path.join(user_profile, "Documents", name),
                os.path.join(user_profile, "Desktop", name),
                os.path.join(user_profile, "Downloads", name),
            ]

            target_path = None
            for path in paths_to_try:
                if os.path.exists(path):
                    target_path = path
                    break

            if not target_path:
                return {"status": "error", "error": f"File not found: {name}", "text": f"I couldn't find '{name}' to delete."}

            try:
                from send2trash import send2trash
                send2trash(target_path)
            except Exception:
                import win32com.shell.shell as shell
                import win32com.shell.shellcon as shellcon
                shell.SHFileOperation(
                    (0, shellcon.FO_DELETE, target_path, None, shellcon.FOF_ALLOWUNDO | shellcon.FOF_NOCONFIRMATION, None, None)
                )

            return {
                "status": "ok",
                "text": f"I've moved '{os.path.basename(target_path)}' to the Recycle Bin.",
                "action": "delete_file",
            }

        try:
            return await asyncio.to_thread(_do_delete)
        except Exception as e:
            return {"status": "error", "error": str(e), "text": f"I couldn't delete '{name}'. {e}"}

    async def _undo(self, params: dict) -> dict:
        """Undo the last action using Ctrl+Z hotkey."""
        from grace.response.feedback import FeedbackSounds

        def _do_undo():
            import pyautogui
            pyautogui.hotkey("ctrl", "z")

        try:
            await asyncio.to_thread(_do_undo)
            FeedbackSounds.play_success()
            return {"status": "ok", "action": "undo", "text": "Undone."}
        except Exception as e:
            logger.error(f"Undo action failed: {e}")
            FeedbackSounds.play_error()
            return {"status": "error", "error": str(e), "text": f"Could not undo: {e}"}

    async def _describe_screen(self, params: dict) -> dict:
        """Describe what is currently visible on the screen."""
        title = ""
        elements_summary = []
        try:
            import win32gui
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd) or "Desktop"
        except Exception:
            title = "Current screen"

        try:
            from grace.agent.perception import PerceptionEngine
            graph = PerceptionEngine.get_graph_builder().get()
            if graph and graph.elements:
                named = [
                    e.name for e in graph.elements
                    if e.name and e.role in ("button", "edit", "link", "menuitem", "tabitem")
                ][:5]
                if named:
                    elements_summary = named
        except Exception as e:
            logger.debug(f"Perception graph unavailable for describe_screen: {e}")

        if elements_summary:
            controls_str = ", ".join(f"'{name}'" for name in elements_summary)
            text = f"You are looking at {title}. Key controls: {controls_str}."
        elif title:
            text = f"The active window is {title}."
        else:
            text = "I cannot detect an active application on screen."

        return {
            "status": "ok",
            "action": "describe_screen",
            "text": text,
            "window_title": title,
            "controls": elements_summary,
        }

    async def _set_speech_rate(self, params: dict) -> dict:
        """Set the Kokoro TTS speech speed multiplier."""
        from grace.response.feedback import FeedbackSounds
        rate_val = params.get("rate", "1.0")
        try:
            if isinstance(rate_val, str):
                rate_lower = rate_val.lower().strip()
                if "slow" in rate_lower:
                    speed = 0.8
                elif "fast" in rate_lower or "quick" in rate_lower:
                    speed = 1.25
                elif "normal" in rate_lower or "default" in rate_lower or "reset" in rate_lower:
                    speed = 1.0
                else:
                    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", rate_lower)
                    speed = float(nums[0]) if nums else 1.0
            else:
                speed = float(rate_val)

            speed = max(0.5, min(2.5, speed))

            if self._kokoro_engine is not None:
                self._kokoro_engine.set_speed(speed)

            FeedbackSounds.play_success()
            return {
                "status": "ok",
                "action": "set_speech_rate",
                "text": f"Speech rate set to {speed:.2f}x.",
                "speed": speed,
            }
        except Exception as e:
            logger.error(f"Set speech rate failed: {e}")
            FeedbackSounds.play_error()
            return {"status": "error", "error": str(e), "text": f"Could not adjust speech rate: {e}"}
