"""Load the project ``.env`` into ``os.environ`` before Config is imported.

``grace.config.Config`` evaluates every ``os.getenv(...)`` call at class-definition
time, so this must run *before* ``from grace.config import Config``.

Two rules matter here, and both exist because of real breakage:

1. **Empty values are skipped.** The project ``.env`` has historically carried
   keys like ``WHISPER_MODEL_PATH=`` and ``LLAMA_SERVER_EXE=``. Exporting those
   as empty strings would blank out working defaults in ``config.py`` and break
   startup, so an empty value means "not set" rather than "set to nothing".

2. **The real environment wins.** A value already present in ``os.environ`` is
   never overwritten, so a shell export or a parent process (the Electron
   launcher) can still override the file.

Deliberately dependency-free: the format used here is plain ``KEY=VALUE`` and a
30-line parser avoids making startup depend on ``python-dotenv`` being installed.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger("grace.env")


def parse_env_file(text: str) -> dict[str, str]:
    """Parse ``.env`` text into a dict, dropping comments, blanks and empty values."""
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue

        value = value.strip()
        # Strip matched surrounding quotes, but leave the contents alone -
        # Windows paths are stored raw and must not be unescaped.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]

        if not value:
            continue  # Rule 1: empty means "not set", never "set to nothing".
        values[key] = value
    return values


def find_env_file(start: Optional[str] = None) -> Optional[str]:
    """Locate the project ``.env`` by walking up from this file."""
    here = start or os.path.dirname(os.path.abspath(__file__))
    current = here
    for _ in range(5):
        candidate = os.path.join(current, ".env")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


def load_env(path: Optional[str] = None) -> dict[str, str]:
    """Load ``.env`` into ``os.environ``. Returns the keys actually applied."""
    env_path = path or find_env_file()
    if not env_path or not os.path.isfile(env_path):
        logger.debug("No .env file found; using config.py defaults")
        return {}

    try:
        with open(env_path, "r", encoding="utf-8") as fh:
            parsed = parse_env_file(fh.read())
    except Exception as e:
        logger.warning(f"Failed to read {env_path}: {e}; using config.py defaults")
        return {}

    applied = {}
    for key, value in parsed.items():
        if key in os.environ:
            continue  # Rule 2: the real environment wins.
        os.environ[key] = value
        applied[key] = value

    if applied:
        # Never log values - GEMINI_API_KEY lives in here.
        logger.info(f"Loaded {len(applied)} setting(s) from {env_path}: {', '.join(sorted(applied))}")
    return applied
