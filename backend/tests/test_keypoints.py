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
