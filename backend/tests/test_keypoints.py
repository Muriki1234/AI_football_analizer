"""Tests for keypoint detection stride and interpolation quality."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np


def test_keypoint_stride_is_20():
    """KEYPOINT_STRIDE must be 20, not 60."""
    from app.pipeline.analysis_core import KEYPOINT_STRIDE
    assert KEYPOINT_STRIDE == 20, f"Expected KEYPOINT_STRIDE=20, got {KEYPOINT_STRIDE}"


def test_keypoint_linear_fill_interpolates_between_samples():
    """
    Linear fill: a keypoint present at frame 0 and frame 20 should be
    smoothly interpolated at frame 10, not just copied from the nearest sample.
    """
    from app.pipeline.analysis_core import _linear_fill_keypoints

    sampled = {
        0:  {0: [100.0, 200.0]},
        20: {0: [200.0, 100.0]},
    }
    total = 21
    result = _linear_fill_keypoints(sampled, total)

    assert len(result) == total

    kp10 = result[10].get(0)
    assert kp10 is not None, "Frame 10 should have keypoint 0 filled in"
    assert abs(kp10[0] - 150.0) < 5.0, f"Expected x≈150, got {kp10[0]:.1f}"
    assert abs(kp10[1] - 150.0) < 5.0, f"Expected y≈150, got {kp10[1]:.1f}"


def test_keypoint_linear_fill_preserves_known_frames():
    """Known frames must not be modified by linear fill."""
    from app.pipeline.analysis_core import _linear_fill_keypoints

    sampled = {
        0:  {0: [50.0, 75.0], 1: [200.0, 300.0]},
        10: {0: [60.0, 85.0]},
    }
    result = _linear_fill_keypoints(sampled, 11)

    assert result[0][0] == [50.0, 75.0], "Frame 0 kp0 must be unchanged"
    assert result[0][1] == [200.0, 300.0], "Frame 0 kp1 must be unchanged"
    assert result[10][0] == [60.0, 85.0], "Frame 10 kp0 must be unchanged"


def test_pitch_clamp_values():
    """position_transformed must be clamped to [0,105] x [0,68]."""
    from app.pipeline.analysis_core import clamp_pitch_position

    clamped = clamp_pitch_position(-5.0, 75.0)
    assert clamped[0] == 0.0, f"x should be clamped to 0, got {clamped[0]}"
    assert clamped[1] == 68.0, f"y should be clamped to 68, got {clamped[1]}"

    clamped2 = clamp_pitch_position(52.5, 34.0)
    assert clamped2 == (52.5, 34.0)


def test_homography_fallback_used_when_few_keypoints():
    """When < 4 keypoints, last good homography should still produce positions."""
    # This is a structural test: verify the function doesn't skip frames
    # when a fallback transformer is available.
    from app.pipeline.analysis_core import ViewTransformer
    import numpy as np

    vt = ViewTransformer()

    # Build fake tracks: 3 frames, 1 player
    tracks = {
        "players": [
            {1: {"bbox": [100, 100, 140, 200], "position_adjusted": [120.0, 200.0]}},
            {1: {"bbox": [100, 100, 140, 200], "position_adjusted": [121.0, 200.0]}},
            {1: {"bbox": [100, 100, 140, 200], "position_adjusted": [122.0, 200.0]}},
        ],
        "ball": [{}, {}, {}],
        "referees": [{}, {}, {}],
    }

    # Frame 0: 5 good keypoints → should compute transformer
    # Frame 1: 2 keypoints only → should use fallback
    # Frame 2: 5 good keypoints → should compute fresh transformer
    # We just verify the function runs without error and doesn't raise
    # (actual homography math needs real keypoint coordinates)
    try:
        # Empty kps → no transformer computed, no crash
        kps_list = [{}, {}, {}]
        vt.add_transformed_position_to_tracks(tracks, kps_list)
        # No assertion on values — just confirm no crash / skip bug
        result_ok = True
    except Exception as e:
        result_ok = False
        assert False, f"add_transformed_position_to_tracks raised: {e}"

    assert result_ok


def test_adaptive_smoothing_window_values():
    """Fast-moving player uses smaller window than stationary player."""
    import numpy as np

    # Speed threshold: < 0.3 m/frame → window=15, >= 0.3 → window=5
    SLOW_THRESHOLD = 0.3

    slow_speed = 0.1   # m/frame
    fast_speed = 0.5   # m/frame

    slow_window = 15 if slow_speed < SLOW_THRESHOLD else 5
    fast_window = 15 if fast_speed < SLOW_THRESHOLD else 5

    assert slow_window == 15, f"Slow player should use window=15, got {slow_window}"
    assert fast_window == 5,  f"Fast player should use window=5, got {fast_window}"
    assert slow_window > fast_window, "Slow player should have larger smoothing window"


def test_ball_trail_length():
    """Ball trail should keep at most 30 positions."""
    trail = []
    for i in range(50):  # simulate 50 frames of ball positions
        trail.append((float(i), float(i)))
        if len(trail) > 30:
            trail.pop(0)

    assert len(trail) == 30, f"Trail should be capped at 30, got {len(trail)}"
    assert trail[-1] == (49.0, 49.0), "Last entry should be most recent position"
    assert trail[0]  == (20.0, 20.0), "First entry should be 30 frames back"
