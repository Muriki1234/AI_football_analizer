"""
test_canonical_pitch_coordinate_normalizer.py
=============================================
Unit tests for CanonicalPitchCoordinateNormalizer.
"""

import unittest
from server.pipeline.canonical_pitch_coordinate_normalizer import (
    CanonicalPitchCoordinateNormalizer,
    NormalizedPitchPoint,
)


class TestCanonicalPitchCoordinateNormalizer(unittest.TestCase):
    def setUp(self):
        self.normalizer = CanonicalPitchCoordinateNormalizer(
            target_length_m=105.0,
            target_width_m=68.0,
            source_is_soccana_padded=True,
        )

    def test_soccana_padded_cm_to_fifa_meters(self):
        """
        Verify that Soccana keypoint coordinates (cm) map accurately to standard FIFA meters.
        """
        # Soccana left goal line top touchline: (750 cm, 100 cm) -> FIFA (0.0m, 0.0m)
        p_origin = self.normalizer.normalize([750.0, 100.0])
        self.assertTrue(p_origin.is_valid)
        self.assertAlmostEqual(p_origin.x_m, 0.0, delta=0.01)
        self.assertAlmostEqual(p_origin.y_m, 0.0, delta=0.01)

        # Soccana field center: (6000 cm, 3500 cm) -> FIFA (52.5m, 34.0m)
        p_center = self.normalizer.normalize([6000.0, 3500.0])
        self.assertTrue(p_center.is_valid)
        self.assertAlmostEqual(p_center.x_m, 52.5, delta=0.01)
        self.assertAlmostEqual(p_center.y_m, 34.0, delta=0.01)

        # Soccana right goal line bottom touchline: (11250 cm, 6900 cm) -> FIFA (105.0m, 68.0m)
        p_end = self.normalizer.normalize([11250.0, 6900.0])
        self.assertTrue(p_end.is_valid)
        self.assertAlmostEqual(p_end.x_m, 105.0, delta=0.01)
        self.assertAlmostEqual(p_end.y_m, 68.0, delta=0.01)

    def test_centimeter_to_meter_auto_detection(self):
        """
        Verify automatic scale detection when inputs are in cm vs m without padding.
        """
        norm_unpadded = CanonicalPitchCoordinateNormalizer(source_is_soccana_padded=False)

        # Point already in meters
        p_m = norm_unpadded.normalize([52.5, 34.0])
        self.assertTrue(p_m.is_valid)
        self.assertEqual(p_m.source_unit, "meters")
        self.assertAlmostEqual(p_m.x_m, 52.5)

        # Point in centimeters
        p_cm = norm_unpadded.normalize([5250.0, 3400.0])
        self.assertTrue(p_cm.is_valid)
        self.assertEqual(p_cm.source_unit, "centimeters")
        self.assertAlmostEqual(p_cm.x_m, 52.5)

    def test_screen_pixel_coordinate_rejection(self):
        """
        Verify that raw screen bbox center coordinates (e.g. 960, 540) are rejected
        to prevent pixel-space distance inflation.
        """
        p_pixel = self.normalizer.normalize([960.0, 540.0])
        self.assertFalse(p_pixel.is_valid, "Screen resolution 1080p center must be rejected as invalid pixel")
        self.assertEqual(p_pixel.source_unit, "invalid_pixel")

    def test_projective_singularity_rejection(self):
        """
        Verify that wild homography extrapolation errors are rejected.
        """
        # Extreme warp in centimeters (500m x -300m)
        p_singularity_cm = self.normalizer.normalize([50000.0, -30000.0])
        self.assertFalse(p_singularity_cm.is_valid)
        self.assertEqual(p_singularity_cm.source_unit, "projective_glitch")

        # Extreme warp in meters (-50m x 200m)
        p_singularity_m = self.normalizer.normalize([-50.0, 200.0])
        self.assertFalse(p_singularity_m.is_valid)
        self.assertEqual(p_singularity_m.source_unit, "projective_glitch")

    def test_run_off_handling(self):
        """
        Verify that corner kick / throw-in players outside the pitch line are permitted with run-off flag.
        """
        # Corner taker: 1 meter behind left goal line, 0.5m outside sideline
        # In Soccana coordinates: X = 750 - 100 = 650 cm, Y = 100 - 50 = 50 cm
        p_corner = self.normalizer.normalize([650.0, 50.0], allow_run_off=True)
        self.assertTrue(p_corner.is_valid)
        self.assertTrue(p_corner.is_run_off)
        self.assertLess(p_corner.x_m, 0.0)
        self.assertLess(p_corner.y_m, 0.0)

    def test_minimap_pixel_alignment(self):
        """
        Verify precise pixel alignment on MinimapOverlay canvas.
        Canvas: 260px wide, 156px high, PAD: 10px.
        Left goal line (0m) -> 10px
        Center line (52.5m) -> 130px
        Right goal line (105m) -> 250px
        """
        px_left, py_left = self.normalizer.to_minimap_pixel(0.0, 34.0, canvas_width=260.0, canvas_height=156.0, padding=10.0)
        self.assertEqual(px_left, 10.0)
        self.assertEqual(py_left, 78.0)

        px_center, py_center = self.normalizer.to_minimap_pixel(52.5, 34.0, canvas_width=260.0, canvas_height=156.0, padding=10.0)
        self.assertEqual(px_center, 130.0)
        self.assertEqual(py_center, 78.0)

        px_right, py_right = self.normalizer.to_minimap_pixel(105.0, 34.0, canvas_width=260.0, canvas_height=156.0, padding=10.0)
        self.assertEqual(px_right, 250.0)
        self.assertEqual(py_right, 78.0)

    def test_extract_from_player_info_ignores_bboxes(self):
        """
        Verify pipeline info dictionary extractor safely ignores raw bboxes.
        """
        # Dict with only screen bbox (homography failed)
        info_no_homography = {
            "bbox": [900, 500, 1020, 600],
            "team": 1,
        }
        extracted = self.normalizer.extract_from_player_info(info_no_homography)
        self.assertIsNone(extracted, "Must return None when homography is missing, never screen pixels")

        # Dict with position_transformed
        info_with_homography = {
            "bbox": [900, 500, 1020, 600],
            "position_transformed": [6000.0, 3500.0],
            "team": 1,
        }
        pos = self.normalizer.extract_from_player_info(info_with_homography)
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos[0], 52.5, delta=0.01)
        self.assertAlmostEqual(pos[1], 34.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
