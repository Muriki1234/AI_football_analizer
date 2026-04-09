"""Tests for pipeline speed optimizations."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_read_video_importable_from_analysis_core():
    """read_video must be importable — needed for short-video fast path."""
    from app.pipeline.analysis_core import read_video
    assert callable(read_video)


def test_short_video_threshold_constant():
    """SHORT_VIDEO_FRAMES threshold must be defined in tasks.py."""
    import pathlib
    tasks_path = pathlib.Path(__file__).parent.parent / "app" / "pipeline" / "tasks.py"
    source = tasks_path.read_text()
    assert "SHORT_VIDEO_FRAMES" in source, \
        "tasks.py must define SHORT_VIDEO_FRAMES constant for fast-path branching"


def test_short_video_threshold_value():
    """SHORT_VIDEO_FRAMES should be between 1800 and 4000."""
    import pathlib, re
    tasks_path = pathlib.Path(__file__).parent.parent / "app" / "pipeline" / "tasks.py"
    source = tasks_path.read_text()
    match = re.search(r"SHORT_VIDEO_FRAMES\s*=\s*(\d+)", source)
    assert match, "Could not find SHORT_VIDEO_FRAMES = <number> in tasks.py"
    value = int(match.group(1))
    assert 1800 <= value <= 4000, \
        f"SHORT_VIDEO_FRAMES={value} is outside expected range [1800, 4000]"
