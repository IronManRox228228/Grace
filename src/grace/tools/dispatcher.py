"""Intent dispatcher - routes parsed intents to CUA or hardcoded tools.

The dispatcher validates intents and delegates execution to the
appropriate tool handler. CUA tools go through the local ComputerUse
backend, system tools execute directly via Python Windows APIs.
"""

import asyncio
import json
import logging
import os
import subprocess
from typing import Any, Optional

from grace.intent.parser import Intent
from grace.automation.computer_use import ComputerUse
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
}

_CUA_LABELS: dict[str, str] = {
    "click": "Clicking\u2026",
    "double_click": "Double-clicking\u2026",
    "right_click": "Right-clicking\u2026",
    "type_text": "Typing\u2026",
    "press_key": "Pressing key\u2026",
    "move_mouse": "Moving mouse\u2026",
    "drag": "Dragging\u2026",
    "scroll": "Scrolling\u2026",
    "screenshot": "Taking screenshot\u2026",
    "list_windows": "Listing windows\u2026",
    "focus_window": "Focusing window\u2026",
    "get_cursor_position": "Getting cursor position\u2026",
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

    def __init__(self, computer_use: Optional[ComputerUse] = None, ws_server: Optional[WsEventServer] = None):
        self._cua = computer_use
        self._ws = ws_server
        self._last_search_results: list[str] = []

    async def execute(self, intent: Intent) -> dict[str, Any]:
        """Execute a parsed intent.

        Routes to CUA for cua_* tools, or to hardcoded tools for
        system tools. Returns a result dict with status and output.
        """
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
                return {"status": "error", "error": result["error"], "text": f"Action error: {result['error']}"}
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

        # 2. Launch installed application via AppIndexer
        from grace.automation.app_indexer import AppIndexer
        if not hasattr(self, "_app_indexer") or self._app_indexer is None:
            self._app_indexer = AppIndexer()

        return self._app_indexer.launch(name)

    async def _close_app(self, params: dict) -> dict:
        name = params.get("name", "")
        if not name:
            return {"status": "error", "error": "Missing 'name' parameter"}

        try:
            result = subprocess.run(
                ["taskkill", "/F", "/IM", f"{name}.exe"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return {"status": "ok", "text": f"I've closed {name}."}
            else:
                result2 = subprocess.run(
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

        self._last_search_results = []

        try:
            user_profile = os.environ.get("USERPROFILE", os.path.expanduser("~"))
            docs_dir = os.path.join(user_profile, "Documents")
            cmd = f'$searchResults = Get-ChildItem -Path "{docs_dir}" -Recurse -Include *.pdf,*.docx,*.txt,*.xlsx -ErrorAction SilentlyContinue | Where-Object {{ $_.Name -like "*{query}*" }} | Select-Object -First 10 FullName; $searchResults | ForEach-Object {{ $_.FullName }}'

            result = subprocess.run(
                ["powershell", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.stdout.strip():
                self._last_search_results = [
                    line.strip() for line in result.stdout.strip().split("\n") if line.strip()
                ]
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
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "Search timed out", "text": "I couldn't search in time."}
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

        try:
            from pypdf import PdfReader
            from grace.rag import LocalRagIndex

            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"

            if not text.strip():
                return {"status": "error", "error": "No extractable text in PDF", "text": "The PDF doesn't contain extractable text."}

            index = LocalRagIndex()
            doc_id = os.path.basename(path)
            index.index_text(doc_id, text)

            if query:
                chunks = index.query(query, top_k=3)
                rag_text = "\n\n".join(f"[Excerpt {c['index']+1}]: {c['chunk']}" for c in chunks)
            else:
                chunks = index.get_summary_chunks(top_k=3)
                rag_text = "\n\n".join(f"[Excerpt {i+1}]: {c}" for i, c in enumerate(chunks))

            return {
                "status": "ok",
                "text": rag_text or text.strip()[:2000],
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

        try:
            from pypdf import PdfReader
            from grace.rag import LocalRagIndex

            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"

            if not text.strip():
                return {"status": "error", "error": "No extractable text in PDF", "text": "The PDF doesn't contain extractable text."}

            index = LocalRagIndex()
            doc_id = os.path.basename(path)
            index.index_text(doc_id, text)

            chunks = index.get_summary_chunks(top_k=4)
            summary_context = "\n\n".join(f"[Section {i+1}]: {c}" for i, c in enumerate(chunks))

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

        try:
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

        try:
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
        except Exception as e:
            return {"status": "error", "error": str(e), "text": f"I couldn't delete '{name}'. {e}"}
