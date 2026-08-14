"""Grace's package root.

The two statements below are ordered, not incidental. ``Config``'s field
defaults are ``os.getenv(...)`` calls evaluated when the class body runs, so
whatever is in ``os.environ`` at *import* time is what Grace is configured with
for the rest of the process.

Importing anything under ``grace`` runs this file first, which means every
entry point - ``main.py``, the tests, the replay harness - reaches
``grace.config`` through here. Loading ``.env`` anywhere else is therefore too
late: ``main.py`` calling ``load_env()`` as its first statement still imports
``grace`` to get at it, and by then the defaults are already frozen. That is
not hypothetical. It silently blanked ``GEMINI_API_KEY``, which sent every
planner call to a local llama-server that cloud mode never starts, and the
agent loop retried the resulting empty plan forever.
"""

from grace.env_loader import load_env

load_env()

from grace.config import Config  # noqa: E402  - must follow load_env(); see above.

config = Config()
