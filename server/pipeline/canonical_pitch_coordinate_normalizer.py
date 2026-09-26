"""
canonical_pitch_coordinate_normalizer.py
========================================
FIFA 105x68m Canonical Pitch Coordinate Normalizer & Metric Guard.

Solves the systemic coordinate and scale rift across the football AI pipeline:
1. Reconciles the underlying Soccana 120x70m (12000x7000 cm) keypoint grid with
   the standard FIFA 105x68m playing pitch.
2. Auto-detects and resolves the 100x centimeter-to-meter ambiguity across all
   downstream tactical engines (`tasks.py` lines 2102, 2422, 2678, 2755, 2834, etc.).
3. Eliminates coordinate space bleedthrough in pass detection where missing
   homography fell back to screen pixel coordinates (e.g. (960, 540)), causing
   1000m jumps in Euclidean distance.
4. Provides verified boundary clamping with run-off margin support (-3m to +3m)
   and projective singularity rejection.
5. Emits canonical minimap canvas projection coordinates aligned with pitch markings.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


@dataclass
class NormalizedPitchPoint:
    x_m: float
    y_m: float
    is_valid: bool
    is_clamped: bool
    is_run_off: bool
    source_unit: str  # 'centimeters', 'meters', 'soccana_padded', 'invalid_pixel'


class CanonicalPitchCoordinateNormalizer:
    """
    Unified coordinate normalizer for football analytics.
    Canonical FIFA field dimensions: 105.0m x 68.0m.
    """

    FIFA_LENGTH_M: float = 105.0
    FIFA_WIDTH_M: float = 68.0

    # Soccana 120x70 coordinate model constants
    SOCCANA_FULL_LENGTH_M: float = 120.0
    SOCCANA_FULL_WIDTH_M: float = 70.0
    # Soccana playing field margins (cm -> m)
    SOCCANA_MARGIN_X_M: float = 7.5   # (120 - 105) / 2
    SOCCANA_MARGIN_Y_M: float = 1.0   # (70 - 68) / 2

    # Physical run-off allowance (meters beyond white lines for corners/throw-ins)
    MAX_RUN_OFF_M: float = 4.0

    def __init__(
        self,
        target_length_m: float = 105.0,
        target_width_m: float = 68.0,
        source_is_soccana_padded: bool = True,
    ) -> None:
        self.target_length_m = target_length_m
        self.target_width_m = target_width_m
        self.source_is_soccana_padded = source_is_soccana_padded

    def normalize(
        self,
        raw_pos: Union[Sequence[float], np.ndarray],
        allow_run_off: bool = True,
    ) -> Optional[NormalizedPitchPoint]:
        """
        Normalize raw position coordinates into canonical FIFA pitch space (meters).
        Handles:
          - Centimeter scale ([0, 12000] or [0, 10500])
          - Meter scale ([0, 120] or [0, 105])
          - Padded Soccana grid to unpadded FIFA pitch
          - Out-of-bounds rejection and run-off clamping
        """
        if raw_pos is None or len(raw_pos) < 2:
            return None

        try:
            rx = float(raw_pos[0])
            ry = float(raw_pos[1])
        except (TypeError, ValueError):
            return None

        if math.isnan(rx) or math.isnan(ry) or math.isinf(rx) or math.isinf(ry):
            return None

        # 1. Detect and reject screen pixel coordinates (e.g. 1920x1080 resolution)
        # Standard broadcast video height is 720 or 1080. If rx > 150 and ry > 120,
        # it could be either cm or pixels. But if ry > 120 and ry <= 1080 and rx <= 1920,
        # and not matching pitch aspect ratio (~1.54), inspect:
        is_screen_pixel = False
        if 150.0 < rx <= 1920.0 and 100.0 < ry <= 1080.0:
            # Check if this could be centimeters:
            # On a 12000x7000 pitch, cm scale has X up to 12000 and Y up to 7000.
            # A value like (960, 540) is exactly 1080p center!
            if rx < 2000.0 and ry < 1200.0:
                # If values are in pixel range (<= 1920, <= 1080) and ratio resembles 16:9 (1.77)
                # rather than football pitch (1.54), check if it was raw bbox center:
                aspect = rx / max(1.0, ry)
                if 1.6 < aspect < 1.9:
                    is_screen_pixel = True

        if is_screen_pixel:
            return NormalizedPitchPoint(
                x_m=0.0,
                y_m=0.0,
                is_valid=False,
                is_clamped=False,
                is_run_off=False,
                source_unit="invalid_pixel",
            )

        # 2. Determine scale unit (Centimeters vs Meters)
        source_unit = "meters"
        x_m = rx
        y_m = ry

        # Centimeter scale on a 12000x7000 pitch typically has values in thousands (>300)
        # and both coordinates positive. Outliers with negative values or small magnitudes are meters.
        if (rx > 250.0 and ry > 100.0 and rx > 0 and ry > 0) or (rx > 350.0 and ry >= 0):
            # In centimeters
            source_unit = "centimeters"
            x_m = rx / 100.0
            y_m = ry / 100.0

        # 3. Handle Soccana Grid Mapping (120x70m with 7.5m / 1.0m margins)
        if self.source_is_soccana_padded:
            if (source_unit == "centimeters" and rx <= 13000.0 and ry <= 8000.0) or (105.0 < x_m <= 125.0 and 0.0 <= y_m <= 75.0):
                # Raw point comes from Soccana 120x70 coordinate model
                source_unit = "soccana_padded"
                # Strip 7.5m left margin and 1.0m top margin, then scale to FIFA
                x_m = (x_m - self.SOCCANA_MARGIN_X_M) * (self.target_length_m / (self.SOCCANA_FULL_LENGTH_M - 2 * self.SOCCANA_MARGIN_X_M))
                y_m = (y_m - self.SOCCANA_MARGIN_Y_M) * (self.target_width_m / (self.SOCCANA_FULL_WIDTH_M - 2 * self.SOCCANA_MARGIN_Y_M))

        # 4. Check for Projective Singularities / Outliers
        margin = self.MAX_RUN_OFF_M if allow_run_off else 0.0
        min_x, max_x = -margin, self.target_length_m + margin
        min_y, max_y = -margin, self.target_width_m + margin

        if x_m < min_x - 10.0 or x_m > max_x + 10.0 or y_m < min_y - 10.0 or y_m > max_y + 10.0:
            # Extreme homography warp singularity
            return NormalizedPitchPoint(
                x_m=0.0,
                y_m=0.0,
                is_valid=False,
                is_clamped=False,
                is_run_off=False,
                source_unit="projective_glitch",
            )

        is_clamped = False
        is_run_off = False

        if x_m < 0.0 or x_m > self.target_length_m or y_m < 0.0 or y_m > self.target_width_m:
            is_run_off = True

        # Clamp to allowed boundary
        clamped_x = max(min_x, min(max_x, x_m))
        clamped_y = max(min_y, min(max_y, y_m))

        if clamped_x != x_m or clamped_y != y_m:
            is_clamped = True

        return NormalizedPitchPoint(
            x_m=round(clamped_x, 3),
            y_m=round(clamped_y, 3),
            is_valid=True,
            is_clamped=is_clamped,
            is_run_off=is_run_off,
            source_unit=source_unit,
        )

    def extract_from_player_info(
        self,
        info: Dict[str, Any],
        allow_run_off: bool = True,
    ) -> Optional[Tuple[float, float]]:
        """
        Safely extracts and normalizes player coordinates from a pipeline track dictionary.
        Prioritizes `position_transformed` or `position_minimap`, and strictly ignores screen bboxes!
        """
        if not info:
            return None

        # Check position_transformed first (in meters or Soccana)
        pos = info.get("position_transformed")
        if pos is not None and len(pos) == 2:
            res = self.normalize(pos, allow_run_off=allow_run_off)
            if res is not None and res.is_valid:
                return (res.x_m, res.y_m)

        # Fallback to position_minimap (cm)
        pos_mm = info.get("position_minimap")
        if pos_mm is not None and len(pos_mm) == 2:
            res = self.normalize(pos_mm, allow_run_off=allow_run_off)
            if res is not None and res.is_valid:
                return (res.x_m, res.y_m)

        return None

    def to_minimap_pixel(
        self,
        x_m: float,
        y_m: float,
        canvas_width: float = 260.0,
        canvas_height: float = 156.0,
        padding: float = 10.0,
    ) -> Tuple[float, float]:
        """
        Convert canonical pitch coordinates (meters) to canvas pixel coordinates for MinimapOverlay.
        Guarantees that x_m = 0 aligns with left goal line, and x_m = 105 aligns with right goal line.
        """
        draw_w = canvas_width - padding * 2.0
        draw_h = canvas_height - padding * 2.0

        # Normalization factor [0.0, 1.0]
        norm_x = max(0.0, min(1.0, x_m / self.target_length_m))
        norm_y = max(0.0, min(1.0, y_m / self.target_width_m))

        px = padding + norm_x * draw_w
        py = padding + norm_y * draw_h
        return (round(px, 1), round(py, 1))
