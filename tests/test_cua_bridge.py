"""Test local ComputerUse backend."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grace.automation.computer_use import ComputerUse


class TestComputerUse:
    """Test suite for ComputerUse."""

    def test_init(self):
        cu = ComputerUse()
        assert not cu.is_ready

    def test_start_stop(self):
        cu = ComputerUse()
        cu.start()
        assert cu.is_ready
        cu.stop()
        assert not cu.is_ready

    def test_perform_screenshot(self):
        cu = ComputerUse()
        cu.start()
        result = cu.perform("screenshot", {})
        assert result.get("ok") is True
        assert "png_b64" in result
        assert result.get("width", 0) > 0
        cu.stop()
