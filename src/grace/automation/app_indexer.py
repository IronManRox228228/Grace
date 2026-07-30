"""Windows Installed Application Indexer & Launcher for Grace.

Indexes installed Win32 applications, Start Menu shortcuts (.lnk),
and Microsoft Store UWP apps (shell:AppsFolder) to resolve voice
app launch requests deterministically without falling back to browsers.
"""

import glob
import json
import logging
import os
import subprocess
import threading
import time
from typing import Optional, Dict, Any

logger = logging.getLogger("grace.automation.app_indexer")

# Recursively globbing Program Files takes tens of seconds. The result barely
# changes between runs, so it is cached on disk and only rebuilt when stale.
CACHE_PATH = os.path.join(os.path.expanduser("~"), ".grace", "app_index.json")
CACHE_MAX_AGE_SECONDS = 7 * 24 * 3600
CACHE_VERSION = 1


class AppIndexer:
    """Indexes and launches installed Windows applications."""

    def __init__(self, use_cache: bool = True):
        self._apps: Dict[str, str] = {}
        self._uwp_apps: Dict[str, str] = {}
        self._indexed = False
        self._lock = threading.Lock()
        if use_cache and self._load_cache():
            return
        self._index_apps()
        if use_cache:
            self._save_cache()

    # -- disk cache ------------------------------------------------------

    def _load_cache(self) -> bool:
        """Populate from the on-disk index. Returns False if absent or stale."""
        try:
            stat = os.stat(CACHE_PATH)
        except OSError:
            return False
        if time.time() - stat.st_mtime > CACHE_MAX_AGE_SECONDS:
            logger.info("App index cache is stale; rebuilding")
            return False
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            logger.debug(f"App index cache unreadable ({e}); rebuilding")
            return False
        if data.get("version") != CACHE_VERSION:
            return False
        apps = data.get("apps") or {}
        uwp = data.get("uwp") or {}
        if not apps and not uwp:
            return False
        with self._lock:
            self._apps = dict(apps)
            self._uwp_apps = dict(uwp)
            self._indexed = True
        logger.info(f"Loaded app index from cache ({len(apps)} apps, {len(uwp)} UWP)")
        return True

    def _save_cache(self) -> None:
        try:
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            with self._lock:
                payload = {
                    "version": CACHE_VERSION,
                    "apps": self._apps,
                    "uwp": self._uwp_apps,
                }
            # Write-then-rename so a crash mid-write cannot leave a truncated
            # cache that would be loaded as authoritative next launch.
            tmp = CACHE_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, CACHE_PATH)
        except Exception as e:
            logger.debug(f"Could not persist app index: {e}")

    def refresh(self) -> None:
        """Force a full rescan and rewrite the cache."""
        with self._lock:
            self._apps = {}
            self._uwp_apps = {}
        self._index_apps()
        self._save_cache()

    def _index_apps(self) -> None:
        """Scan Start Menu shortcuts and local user programs."""
        started = time.perf_counter()
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
            logger.info(
                f"Indexed {len(self._apps)} Windows desktop applications "
                f"in {time.perf_counter() - started:.1f}s"
            )

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
