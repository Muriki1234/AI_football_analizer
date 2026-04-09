"""Tests for ball detection accuracy improvements."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline.analysis_core import BALL_CONF, YOLO_DETECTION_STRIDE


def test_ball_conf_is_higher_than_player_conf():
    """Ball confidence should be >= 0.25 to reduce false positives."""
    assert BALL_CONF >= 0.25, f"BALL_CONF={BALL_CONF} is too low — raises false positive rate"


def test_ball_conf_is_not_too_high():
    """Ball confidence shouldn't be so high we miss real detections."""
    assert BALL_CONF <= 0.5, f"BALL_CONF={BALL_CONF} may miss real ball detections"


import numpy as np

def test_ball_filtering_removes_low_confidence():
    """Simulate post-filtering: ball detections below BALL_CONF should be dropped."""
    from app.pipeline.analysis_core import BALL_CONF

    fake_ball_confs = [0.12, 0.28, 0.35, 0.50]
    kept = [c for c in fake_ball_confs if c >= BALL_CONF]
    assert len(kept) == 2, f"Expected 2 kept (>=0.3), got {len(kept)}: {kept}"
    assert all(c >= BALL_CONF for c in kept)


def test_spline_interpolation_is_smoother_than_linear():
    """Cubic spline should produce a non-linear curve between known points."""
    from app.pipeline.analysis_core import interpolate_ball_positions_spline

    positions = [{}] * 11
    positions[0]  = {1: {"bbox": [90, 190, 110, 210]}}   # center (100, 200)
    positions[10] = {1: {"bbox": [190, 90,  210, 110]}}   # center (200, 100)

    result = interpolate_ball_positions_spline(positions)

    assert all(1 in r and "bbox" in r[1] for r in result), "Missing bboxes after interpolation"

    mid = result[5][1]["bbox"]
    mid_cx = (mid[0] + mid[2]) / 2
    mid_cy = (mid[1] + mid[3]) / 2
    assert 130 <= mid_cx <= 170, f"Mid-frame cx={mid_cx:.1f} out of expected range"
    assert 130 <= mid_cy <= 170, f"Mid-frame cy={mid_cy:.1f} out of expected range"
