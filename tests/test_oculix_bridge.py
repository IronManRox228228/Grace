"""Unit tests for OculiX Java Bridge (and graceful fallback behavior)."""

import os
import unittest
from grace.automation.oculix_bridge import OculixBridge


class TestOculixBridge(unittest.TestCase):

    def test_get_classpath(self):
        cp = OculixBridge.get_classpath()
        self.assertIsInstance(cp, list)

    def test_bridge_initialization_or_fallback(self):
        # Initializing OculiX bridge should either succeed or fail gracefully without unhandled exception
        is_ready = OculixBridge.initialize()
        self.assertIn(is_ready, (True, False))


if __name__ == "__main__":
    unittest.main()
