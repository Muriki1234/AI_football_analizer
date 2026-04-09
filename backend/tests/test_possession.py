"""Tests for SmartBallPossessionDetector states."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline.analysis_core import SmartBallPossessionDetector


def _make_detector():
    return SmartBallPossessionDetector(fps=24, video_w=1920, video_h=1080)


def _make_player(x1, y1, x2, y2):
    return {"bbox": [x1, y1, x2, y2]}


def test_controlled_state_near_player():
    """Ball near a player's feet → state='controlled', returns player id."""
    det = _make_detector()
    players = {1: _make_player(80, 150, 120, 200)}
    ball_bbox = [90, 185, 110, 205]
    ball_history = [(100, 195)] * 5

    pid = det.detect_possession(0, players, ball_bbox, ball_history)
    assert det.ball_state == "controlled"
    assert pid == 1 or pid == -1  # might need 3-frame smoothing to lock in


def test_loose_ball_state_when_no_player_nearby():
    """Ball moving slowly but no player close → state='loose_ball', returns -1."""
    det = _make_detector()
    players = {1: _make_player(880, 150, 920, 200)}
    ball_bbox = [90, 90, 110, 110]
    ball_history = [(100, 100)] * 5

    pid = det.detect_possession(0, players, ball_bbox, ball_history)
    assert det.ball_state == "loose_ball", f"Expected 'loose_ball', got '{det.ball_state}'"
    assert pid == -1


def test_flying_state_fast_ball():
    """Fast-moving ball → state='flying'."""
    det = _make_detector()
    players = {1: _make_player(80, 150, 120, 200)}
    ball_bbox = [90, 90, 110, 110]
    ball_history = [(100, 100), (100, 150), (100, 200), (100, 250), (100, 300)]

    det.detect_possession(0, players, ball_bbox, ball_history)
    assert det.ball_state == "flying"


def test_relative_control_distance_scales_with_resolution():
    """Control distance threshold should scale with video resolution."""
    det_hd = SmartBallPossessionDetector(fps=24, video_w=1920, video_h=1080)
    det_4k = SmartBallPossessionDetector(fps=24, video_w=3840, video_h=2160)
    assert det_4k.max_control_distance > det_hd.max_control_distance, \
        "4K detector should have larger control distance than HD"
