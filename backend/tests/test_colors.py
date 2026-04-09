"""Tests that team colors use real jersey colors, not hardcoded values."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline.analysis_core import bgr_to_hex
import numpy as np


def test_bgr_to_hex_black():
    assert bgr_to_hex([0, 0, 0]) == "#000000"


def test_bgr_to_hex_white():
    assert bgr_to_hex([255, 255, 255]) == "#ffffff"


def test_bgr_to_hex_red():
    # Pure red in BGR = (0, 0, 255)
    assert bgr_to_hex([0, 0, 255]) == "#ff0000"


def test_bgr_to_hex_green():
    # Pure green in BGR = (0, 255, 0)
    assert bgr_to_hex([0, 255, 0]) == "#00ff00"


def test_real_jersey_color_is_not_hardcoded_blue():
    """A dark-green jersey should NOT produce #00BFFF."""
    dark_green_bgr = np.array([40, 100, 20])  # dark green in BGR
    result = bgr_to_hex(dark_green_bgr)
    assert result != "#00BFFF", f"Expected real jersey color, got hardcoded blue: {result}"
    assert result != "#FF1493", f"Expected real jersey color, got hardcoded pink: {result}"


def test_tasks_module_has_no_hardcoded_display_hex():
    """tasks.py must not define TEAM1_DISPLAY_HEX or TEAM2_DISPLAY_HEX."""
    import pathlib
    tasks_path = pathlib.Path(__file__).parent.parent / "app" / "pipeline" / "tasks.py"
    source = tasks_path.read_text()
    assert "TEAM1_DISPLAY_HEX" not in source, \
        "tasks.py still has TEAM1_DISPLAY_HEX hardcoded constant — remove it"
    assert "TEAM2_DISPLAY_HEX" not in source, \
        "tasks.py still has TEAM2_DISPLAY_HEX hardcoded constant — remove it"
    assert "TEAM1_DISPLAY_BGR" not in source, \
        "tasks.py still has TEAM1_DISPLAY_BGR hardcoded constant — remove it"
    assert "TEAM2_DISPLAY_BGR" not in source, \
        "tasks.py still has TEAM2_DISPLAY_BGR hardcoded constant — remove it"
