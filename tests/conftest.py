"""Make the test suite import the project's `grace`, not a lookalike.

An unrelated PyPI distribution named `grace` (a build helper) can be present in
site-packages. Most test modules do their own `sys.path.insert(0, ../src)`, but
a few do not, and whichever module is collected first decides which `grace`
lands in sys.modules for the whole session. Pinning the path here removes that
ordering dependency.
"""

import os
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC in sys.path:
    sys.path.remove(SRC)
sys.path.insert(0, SRC)
