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
        # lower-case index key -> name as the system spells it (for messages)
        self._display: Dict[str, str] = {}
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
            self._display = dict(data.get("display") or {})
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
                    "display": self._display,
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
            self._display = {}
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
                        original = os.path.splitext(os.path.basename(filepath))[0].strip()
                        basename = original.lower()
                        if basename and basename not in self._apps:
                            self._apps[basename] = filepath
                            self._display[basename] = original
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
                            original = item.get("Name", "").strip()
                            name = original.lower()
                            appid = item.get("AppID", "").strip()
                            if name and appid:
                                self._uwp_apps[name] = appid
                                self._display[name] = original
                        logger.info(f"Indexed {len(self._uwp_apps)} Windows UWP Apps (Get-StartApps)")
        except Exception as e:
            logger.debug(f"UWP App index query skipped: {e}")

    @staticmethod
    def _is_uninstaller(name: str, path: str = "") -> bool:
        combined = f"{name} {path}".lower()
        return any(kw in combined for kw in ("uninstall", "uninst", "setup", "installer"))

    def find_app(self, query: str) -> Optional[str]:
        """Find matching application target path or AppID."""
        if not query:
            return None

        import re

        q = query.lower().strip()
        query_wants_uninstaller = any(kw in q for kw in ("uninstall", "uninst", "setup", "installer"))

        with self._lock:
            # 1. Exact match in indexed shortcuts / EXEs
            if q in self._apps:
                return self._apps[q]

            # 2. Exact match in UWP apps
            if q in self._uwp_apps:
                return f"uwp:{self._uwp_apps[q]}"

            # 3. Scored match in indexed shortcuts
            # (demote_uninstaller, match_tier, name_len, path)
            candidates = []
            for app_name, app_path in self._apps.items():
                is_uninst = self._is_uninstaller(app_name, app_path)
                demote = 1 if (is_uninst and not query_wants_uninstaller) else 0

                if app_name.startswith(q):
                    tier = 0
                elif re.search(r"\b" + re.escape(q) + r"\b", app_name):
                    tier = 1
                elif q in app_name:
                    tier = 2
                else:
                    continue
                candidates.append((demote, tier, len(app_name), app_path))

            if candidates:
                candidates.sort(key=lambda c: (c[0], c[1], c[2]))
                return candidates[0][3]

            # 4. Scored match in UWP apps
            uwp_candidates = []
            for uwp_name, appid in self._uwp_apps.items():
                is_uninst = self._is_uninstaller(uwp_name)
                demote = 1 if (is_uninst and not query_wants_uninstaller) else 0

                if uwp_name.startswith(q):
                    tier = 0
                elif re.search(r"\b" + re.escape(q) + r"\b", uwp_name):
                    tier = 1
                elif q in uwp_name:
                    tier = 2
                else:
                    continue
                uwp_candidates.append((demote, tier, len(uwp_name), appid))

            if uwp_candidates:
                uwp_candidates.sort(key=lambda c: (c[0], c[1], c[2]))
                return f"uwp:{uwp_candidates[0][3]}"

        return None

    @staticmethod
    def _words(text: str) -> set[str]:
        import re

        return set(re.findall(r"[a-z0-9]+", (text or "").lower()))

    def nearest_apps(self, query: str, limit: int = 5) -> list[str]:
        """Up to `limit` installed app names closest to `query`, best first.

        Scored by word overlap in both directions (the query containing the
        app's words, and the app's name containing the query's words),
        case-insensitive. Apps sharing no word with the query are never listed.
        """
        q_words = self._words(query)
        if not q_words:
            return []
        scored: list[tuple[float, int, str]] = []
        seen: set[str] = set()
        with self._lock:
            names = list(self._apps) + list(self._uwp_apps)
            display = dict(getattr(self, "_display", {}) or {})
        for key in names:
            if key in seen:
                continue
            seen.add(key)
            a_words = self._words(key)
            if not a_words or self._is_uninstaller(key):
                continue
            common = len(q_words & a_words)
            if not common:
                continue
            score = common / len(a_words) + common / len(q_words)
            scored.append((-score, len(key), key))
        scored.sort()
        return [display.get(key, key) for _, _, key in scored[: max(0, limit)]]

    # Characters cmd.exe (or a future shell path) would re-parse as its own
    # syntax. `name` can be model-generated text, so a launch request must
    # never carry one through to a process launch.
    _UNSAFE_NAME_CHARS = set('&|<>^"%')

    PROTOCOL_MAP = {
        "whatsapp": "whatsapp://",
        "settings": "ms-settings:",
        "calculator": "calc:",
        "calc": "calc:",
    }

    def is_installed(self, name: str) -> bool:
        """True when `launch(name)` would open an installed app (index or protocol handler)."""
        clean = (name or "").lower().strip()
        if not clean or any(ch in clean for ch in self._UNSAFE_NAME_CHARS):
            return False
        return (
            clean in self.PROTOCOL_MAP
            or self.find_app(clean) is not None
            or self._is_system_command(clean)
        )

    @staticmethod
    def _is_system_command(clean: str) -> bool:
        """True for a bare program name Windows itself resolves (PATH or App Paths).

        `launch` ends in `os.startfile(name)`, which opens things like notepad,
        cmd or regedit that are in no Start Menu index. Paths are excluded: only
        a bare name counts.
        """
        import shutil

        if any(ch in clean for ch in "\\/:"):
            return False
        if shutil.which(clean, path=os.environ.get("PATH", "")):
            return True
        try:
            import winreg

            exe = clean if clean.endswith(".exe") else clean + ".exe"
            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                try:
                    with winreg.OpenKey(
                        hive, rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe}"
                    ):
                        return True
                except OSError:
                    continue
        except ImportError:
            pass
        return False

    def launch(self, name: str) -> dict[str, Any]:
        """Launch an application by name or search query."""
        if not name:
            return {"status": "error", "error": "Missing application name"}
        if any(ch in name for ch in self._UNSAFE_NAME_CHARS):
            return {
                "status": "error",
                "error": f"Refusing to launch '{name}': contains unsafe characters.",
            }

        clean_name = name.lower().strip()

        # Direct Windows Protocol URI Handlers (WhatsApp, Settings, Calculator)
        PROTOCOL_MAP = self.PROTOCOL_MAP
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
                    try:
                        os.startfile(f"shell:AppsFolder\\{appid}")
                    except Exception:
                        # No cmd.exe: it re-parses the argument for &|<>^, and
                        # explorer.exe accepts a shell: URI directly with no
                        # shell involved.
                        subprocess.run(["explorer.exe", f"shell:AppsFolder\\{appid}"])
                    return {"status": "ok", "text": f"I've opened {name}."}
                else:
                    os.startfile(target)
                    return {"status": "ok", "text": f"I've opened {name}."}
            except Exception as e:
                logger.error(f"Failed to launch {name} via target '{target}': {e}")
                return {"status": "error", "error": str(e), "text": f"Failed to open {name}. {e}"}

        # No cmd.exe fallback here either: `clean_name` is unresolved,
        # arbitrary text at this point, and cmd would re-parse it as shell
        # syntax. If Windows can't open it directly, say so honestly.
        try:
            os.startfile(clean_name)
            return {"status": "ok", "text": f"I've opened {name}."}
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
                "text": f"Sorry, I couldn't find or open '{name}' on your computer."
            }
