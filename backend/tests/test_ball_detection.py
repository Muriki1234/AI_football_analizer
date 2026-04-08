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
