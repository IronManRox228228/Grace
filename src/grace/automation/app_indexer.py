"""Windows Installed Application Indexer & Launcher for Grace.

Indexes installed Win32 applications, Start Menu shortcuts (.lnk),
and Microsoft Store UWP apps (shell:AppsFolder) to resolve voice
app launch requests deterministically without falling back to browsers.
"""

import glob
import logging
import os
import subprocess
import threading
from typing import Optional, Dict, Any

logger = logging.getLogger("grace.automation.app_indexer")


class AppIndexer:
    """Indexes and launches installed Windows applications."""

    def __init__(self):
        self._apps: Dict[str, str] = {}
        self._uwp_apps: Dict[str, str] = {}
        self._indexed = False
        self._lock = threading.Lock()
        self._index_apps()

    def _index_apps(self) -> None:
        """Scan Start Menu shortcuts and local user programs."""
        with self._lock:
            user_profile = os.environ.get("USERPROFILE", os.path.expanduser("~"))
            search_paths = [
                r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\**\*.lnk",
                os.path.join(user_profile, r"AppData\Roaming\Microsoft\Windows\Start Menu\Programs\**\*.lnk"),
                os.path.join(user_profile, r"AppData\Local\Programs\**\*.exe"),
                r"C:\Program Files\**\*.exe",
                r"C:\Program Files (x86)\**\*.exe",
            ]

            for path_pattern in search_paths:  # Index all search paths
                try:
                    for filepath in glob.glob(path_pattern, recursive=True):
                        basename = os.path.splitext(os.path.basename(filepath))[0].lower().strip()
                        if basename and basename not in self._apps:
                            self._apps[basename] = filepath
                except Exception as e:
                    logger.debug(f"Error scanning pattern {path_pattern}: {e}")

            self._indexed = True
            logger.info(f"Indexed {len(self._apps)} Windows desktop applications")

        # Synchronously scan for UWP / Store Apps via PowerShell Get-StartApps
        self._scan_uwp_apps()

    def _scan_uwp_apps(self) -> None:
        """Query Get-StartApps for UWP AppIDs (e.g. WhatsApp, Calculator)."""
        try:
            cmd = "powershell -Command \"Get-StartApps | Select-Object Name, AppID | ConvertTo-Json\""
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5, shell=True)
            if res.returncode == 0 and res.stdout.strip():
                import json
                data = json.loads(res.stdout)
                if isinstance(data, dict):
                    data = [data]
                if isinstance(data, list):
                    with self._lock:
                        for item in data:
                            name = item.get("Name", "").lower().strip()
                            appid = item.get("AppID", "").strip()
                            if name and appid:
                                self._uwp_apps[name] = appid
                        logger.info(f"Indexed {len(self._uwp_apps)} Windows UWP Apps (Get-StartApps)")
        except Exception as e:
            logger.debug(f"UWP App index query skipped: {e}")

    def find_app(self, query: str) -> Optional[str]:
        """Find matching application target path or AppID."""
        if not query:
            return None

        q = query.lower().strip()

        with self._lock:
            # 1. Exact match in indexed shortcuts / EXEs
            if q in self._apps:
                return self._apps[q]

            # 2. Exact match in UWP apps
            if q in self._uwp_apps:
                return f"uwp:{self._uwp_apps[q]}"

            # 3. Partial match in indexed shortcuts
            for app_name, app_path in self._apps.items():
                if q in app_name or app_name in q:
                    return app_path

            # 4. Partial match in UWP apps
            for uwp_name, appid in self._uwp_apps.items():
                if q in uwp_name or uwp_name in q:
                    return f"uwp:{appid}"

        return None

    def launch(self, name: str) -> dict[str, Any]:
        """Launch an application by name or search query."""
        if not name:
            return {"status": "error", "error": "Missing application name"}

        clean_name = name.lower().strip()

        # Direct Windows Protocol URI Handlers (WhatsApp, Settings, Calculator)
        PROTOCOL_MAP = {
            "whatsapp": "whatsapp://",
            "settings": "ms-settings:",
            "calculator": "calc:",
            "calc": "calc:",
        }
        if clean_name in PROTOCOL_MAP:
            try:
                os.startfile(PROTOCOL_MAP[clean_name])
                return {"status": "ok", "text": f"I've opened {name}."}
            except Exception as pe:
                logger.debug(f"Protocol launch for {clean_name} fallback to UWP index: {pe}")

        target = self.find_app(clean_name)

        if target:
            try:
                if target.startswith("uwp:"):
                    appid = target[4:]
                    os.system(f'start "" "shell:AppsFolder\\{appid}"')
                    return {"status": "ok", "text": f"I've opened {name}."}
                else:
                    os.startfile(target)
                    return {"status": "ok", "text": f"I've opened {name}."}
            except Exception as e:
                logger.error(f"Failed to launch {name} via target '{target}': {e}")
                return {"status": "error", "error": str(e), "text": f"Failed to open {name}. {e}"}

        # Direct shell start fallback
        try:
            os.system(f'start "" "{clean_name}"')
            return {"status": "ok", "text": f"I've opened {name}."}
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
                "text": f"Sorry, I couldn't find or open '{name}' on your computer."
            }
