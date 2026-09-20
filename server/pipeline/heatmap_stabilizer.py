"""
heatmap_stabilizer.py — Spatial-Temporal Fidelity & Trajectory Interpolator for Football Analytics

Addresses 4 core production failure modes in player heatmaps:
1. Coordinate space divergence: Centimeters (12000x7000) vs Meters (105x68) vs Normalized (0..1).
2. Tracklet fragmentation: Frame skipping (gating/accelerator) and brief occlusions cause
   sprints to appear as sparse disconnected dots while stationary dwell over-saturates.
3. Outlier/homography glitch suppression: Projective matrix jitter projecting outside pitch boundaries.
4. Matplotlib fallback clipping: Matplotlib pitch with ax.set_xlim(0, 105) clipping centimeter coordinates.

External Grounding:
- mplsoccer: metric 2D binning + scipy.ndimage.gaussian_filter physical bandwidth
- floodlight-sports: bounded temporal trajectory interpolation (<= 30 frames)
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Physical Pitch Defaults (FIFA standard: 105m x 68m, Roboflow Sports: 12000cm x 7000cm)
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
PITCH_LENGTH_CM = 12000.0
PITCH_WIDTH_CM = 7000.0


def detect_coordinate_system(points: np.ndarray) -> str:
    """
    Detects whether points are in:
    - 'cm': Centimeters (0..12000, 0..7000)
    - 'm': Meters (0..105, 0..68)
    - 'norm': Normalized (0..1, 0..1)
    """
    if len(points) == 0:
        return "m"
    max_val = float(np.nanmax(points))
    if max_val > 500.0:
        return "cm"
    elif max_val > 1.05:
        return "m"
    else:
        return "norm"


def normalize_coordinates(
    points: Union[List, np.ndarray],
    target_unit: str = "m",
    clip_pitch: bool = True,
    margin_ratio: float = 0.05,
) -> np.ndarray:
    """
    Normalizes 2D coordinates to either 'm' (meters: 105x68), 'cm' (centimeters: 12000x7000),
    or 'norm' (0..1).

    Args:
        points: Array-like of shape (N, 2)
        target_unit: 'm', 'cm', or 'norm'
        clip_pitch: Whether to clamp points to pitch bounds (+ margin)
        margin_ratio: Margin beyond pitch boundaries to allow before clipping
    """
    if points is None or len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float32)
    pts = pts[:, :2]

    # Filter out NaNs / Infs
    valid_mask = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    if not np.any(valid_mask):
        return np.empty((0, 2), dtype=np.float32)
    pts = pts[valid_mask]

    src_unit = detect_coordinate_system(pts)

    # First convert src_unit -> meters (105 x 68)
    if src_unit == "cm":
        m_x = pts[:, 0] * (PITCH_LENGTH_M / PITCH_LENGTH_CM)
        m_y = pts[:, 1] * (PITCH_WIDTH_M / PITCH_WIDTH_CM)
    elif src_unit == "norm":
        m_x = pts[:, 0] * PITCH_LENGTH_M
        m_y = pts[:, 1] * PITCH_WIDTH_M
    else:  # already meters
        m_x = pts[:, 0].copy()
        m_y = pts[:, 1].copy()

    # Optional boundary clamping
    if clip_pitch:
        min_x = -PITCH_LENGTH_M * margin_ratio
        max_x = PITCH_LENGTH_M * (1.0 + margin_ratio)
        min_y = -PITCH_WIDTH_M * margin_ratio
        max_y = PITCH_WIDTH_M * (1.0 + margin_ratio)
        m_x = np.clip(m_x, min_x, max_x)
        m_y = np.clip(m_y, min_y, max_y)

    # Convert from meters to target_unit
    if target_unit == "cm":
        out_x = m_x * (PITCH_LENGTH_CM / PITCH_LENGTH_M)
        out_y = m_y * (PITCH_WIDTH_CM / PITCH_WIDTH_M)
    elif target_unit == "norm":
        out_x = m_x / PITCH_LENGTH_M
        out_y = m_y / PITCH_WIDTH_M
    else:  # meters
        out_x = m_x
        out_y = m_y

    return np.column_stack([out_x, out_y])


def interpolate_trajectory(
    frame_points: Union[Dict[int, Union[List, Tuple]], List[Tuple[int, float, float]]],
    max_gap_frames: int = 30,
) -> List[Tuple[int, float, float]]:
    """
    Interpolates gaps in tracking data up to `max_gap_frames` (e.g. 1 second at 30 FPS).
    Gaps larger than `max_gap_frames` (e.g. when player leaves camera frame) are NOT
    interpolated to prevent spurious straight-line motion across the pitch.

    Args:
        frame_points: Dict {frame_idx: [x, y]} or List of (frame_idx, x, y)
        max_gap_frames: Maximum gap to linearly bridge

    Returns:
        Sorted list of (frame_idx, x, y) tuples including interpolated frames.
    """
    if not frame_points:
        return []

    if isinstance(frame_points, dict):
        raw_items = sorted(frame_points.items(), key=lambda k: k[0])
        parsed: List[Tuple[int, float, float]] = []
        for f_idx, pt in raw_items:
            if pt is not None and len(pt) >= 2:
                x, y = float(pt[0]), float(pt[1])
                if math.isfinite(x) and math.isfinite(y):
                    parsed.append((int(f_idx), x, y))
    else:
        parsed = []
        for item in sorted(frame_points, key=lambda k: k[0]):
            f_idx = int(item[0])
            x, y = float(item[1]), float(item[2])
            if math.isfinite(x) and math.isfinite(y):
                parsed.append((f_idx, x, y))

    if len(parsed) <= 1:
        return parsed

    result: List[Tuple[int, float, float]] = [parsed[0]]
    for i in range(len(parsed) - 1):
        f0, x0, y0 = parsed[i]
        f1, x1, y1 = parsed[i + 1]
        gap = f1 - f0
        if 1 < gap <= max_gap_frames:
            # Linear interpolation for intermediate frames
            for step in range(1, gap):
                alpha = step / float(gap)
                interp_f = f0 + step
                interp_x = x0 + alpha * (x1 - x0)
                interp_y = y0 + alpha * (y1 - y0)
                result.append((interp_f, interp_x, interp_y))
        result.append((f1, x1, y1))

    return result


def compute_density_grid(
    points_meters: np.ndarray,
    grid_res: Tuple[int, int] = (105, 68),
    sigma_meters: float = 3.5,
) -> np.ndarray:
    """
    Computes a physically calibrated 2D spatial density grid in pitch meters.

    Args:
        points_meters: Array of (N, 2) in meters [0..105, 0..68]
        grid_res: (grid_nx, grid_ny) resolution
        sigma_meters: Gaussian bandwidth in physical meters (default: 3.5m)

    Returns:
        Normalized density grid of shape (grid_ny, grid_nx) in [0.0, 1.0].
    """
    nx, ny = grid_res
    if len(points_meters) == 0:
        return np.zeros((ny, nx), dtype=np.float32)

    # 2D Histogram binning
    x_edges = np.linspace(0.0, PITCH_LENGTH_M, nx + 1)
    y_edges = np.linspace(0.0, PITCH_WIDTH_M, ny + 1)

    hist, _, _ = np.histogram2d(
        points_meters[:, 0],
        points_meters[:, 1],
        bins=[x_edges, y_edges],
    )
    # hist shape is (nx, ny); transpose to (ny, nx) for image-like coordinates
    density = hist.T.astype(np.float32)

    # Compute Gaussian smoothing kernel size based on meter resolution
    dx = PITCH_LENGTH_M / nx
    dy = PITCH_WIDTH_M / ny
    sigma_pixels_x = sigma_meters / dx
    sigma_pixels_y = sigma_meters / dy

    try:
        from scipy.ndimage import gaussian_filter
        density = gaussian_filter(density, sigma=(sigma_pixels_y, sigma_pixels_x))
    except ImportError:
        # High-performance NumPy separable 1D Gaussian fallback
        def _gaussian_1d(arr: np.ndarray, sigma: float, axis: int) -> np.ndarray:
            radius = int(math.ceil(sigma * 3))
            kernel_x = np.arange(-radius, radius + 1)
            kernel = np.exp(-0.5 * (kernel_x / max(sigma, 1e-3)) ** 2)
            kernel /= kernel.sum()
            return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis, arr)

        density = _gaussian_1d(density, sigma_pixels_y, axis=0)
        density = _gaussian_1d(density, sigma_pixels_x, axis=1)

    max_val = float(density.max())
    if max_val > 1e-6:
        density /= max_val

    return density


class HeatmapSpatialStabilizer:
    """
    Production-grade Heatmap Stabilizer coordinating coordinate normalization,
    trajectory gap interpolation, metric KDE generation, and multi-backend rendering.
    """

    def __init__(
        self,
        max_gap_frames: int = 30,
        sigma_meters: float = 3.5,
        target_grid_res: Tuple[int, int] = (105, 68),
    ):
        self.max_gap_frames = max_gap_frames
        self.sigma_meters = sigma_meters
        self.target_grid_res = target_grid_res

    def prepare_points(
        self,
        raw_data: Union[Dict[int, List], List[Tuple[int, List]], List[List]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes raw point observations into:
        1. points_m: Cleaned, interpolated positions in meters [0..105, 0..68]
        2. points_cm: Cleaned, interpolated positions in centimeters [0..12000, 0..7000]
        """
        if isinstance(raw_data, dict):
            interpolated = interpolate_trajectory(raw_data, max_gap_frames=self.max_gap_frames)
            raw_pts = np.array([[x, y] for _, x, y in interpolated], dtype=np.float32)
        elif isinstance(raw_data, list) and len(raw_data) > 0:
            if isinstance(raw_data[0], tuple) and len(raw_data[0]) >= 3:
                interpolated = interpolate_trajectory(raw_data, max_gap_frames=self.max_gap_frames)
                raw_pts = np.array([[x, y] for _, x, y in interpolated], dtype=np.float32)
            else:
                raw_pts = np.asarray(raw_data, dtype=np.float32)
        else:
            raw_pts = np.empty((0, 2), dtype=np.float32)

        if len(raw_pts) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        points_m = normalize_coordinates(raw_pts, target_unit="m")
        points_cm = normalize_coordinates(raw_pts, target_unit="cm")
        return points_m, points_cm

    def render(
        self,
        raw_data: Union[Dict[int, List], List],
        output_path: Union[str, Path],
        prefer_sports: bool = True,
    ) -> bool:
        """
        Renders the stabilized heatmap image to output_path.
        Automatically handles Sports vs Matplotlib backends with proper metric coordinates.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        points_m, points_cm = self.prepare_points(raw_data)

        if len(points_m) < 3:
            return self._render_empty(output_path)

        if prefer_sports:
            try:
                import cv2
                import supervision as sv
                from sports.common.pitch import SoccerPitchConfiguration, draw_pitch, draw_points_on_pitch

                config = SoccerPitchConfiguration()
                pitch = draw_pitch(config=config)
                blank = np.zeros_like(pitch)
                white = sv.Color.from_hex("#FFFFFF")

                sample = points_cm[::2] if len(points_cm) > 400 else points_cm
                dot_layer = draw_points_on_pitch(
                    config=config,
                    xy=sample,
                    face_color=white,
                    edge_color=white,
                    radius=2,
                    pitch=blank,
                )
                gray = cv2.cvtColor(dot_layer, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (101, 101), 0)
                normed = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                hmap = cv2.applyColorMap(normed, cv2.COLORMAP_JET)

                mask = normed > 18
                result = pitch.copy()
                result[mask] = cv2.addWeighted(pitch[mask], 0.35, hmap[mask], 0.65, 0)
                cv2.imwrite(str(output_path), result)
                return True
            except (ImportError, Exception):
                pass

        # Fallback to Matplotlib (now fully verified with metric bounds [0..105, 0..68])
        return self._render_matplotlib(points_m, output_path)

    def _render_matplotlib(self, points_m: np.ndarray, output_path: Path) -> bool:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 8), facecolor="#22312b")
        ax.set_facecolor("#22312b")

        # Draw pitch outline and penalty boxes in meters
        for rect in [
            plt.Rectangle((0, 0), PITCH_LENGTH_M, PITCH_WIDTH_M, fill=False, ec="white", lw=2),
            plt.Rectangle((0, 13.84), 16.5, 40.32, fill=False, ec="white", lw=1.5),
            plt.Rectangle((PITCH_LENGTH_M - 16.5, 13.84), 16.5, 40.32, fill=False, ec="white", lw=1.5),
            plt.Rectangle((0, 24.84), 5.5, 18.32, fill=False, ec="white", lw=1.0),
            plt.Rectangle((PITCH_LENGTH_M - 5.5, 24.84), 5.5, 18.32, fill=False, ec="white", lw=1.0),
        ]:
            ax.add_patch(rect)

        # Halfway line & center circle
        ax.axvline(x=PITCH_LENGTH_M / 2.0, color="white", lw=1.5)
        circle = plt.Circle((PITCH_LENGTH_M / 2.0, PITCH_WIDTH_M / 2.0), 9.15, fill=False, ec="white", lw=1.5)
        ax.add_patch(circle)

        # Compute metric density grid
        density = compute_density_grid(
            points_m,
            grid_res=self.target_grid_res,
            sigma_meters=self.sigma_meters,
        )

        extent = [0, PITCH_LENGTH_M, 0, PITCH_WIDTH_M]
        im = ax.imshow(
            density,
            extent=extent,
            origin="lower",
            cmap="hot",
            alpha=0.65,
            interpolation="bicubic",
        )

        ax.scatter(points_m[::5, 0], points_m[::5, 1], c="cyan", s=3, alpha=0.3, label="Trajectory")

        ax.set_xlim(0, PITCH_LENGTH_M)
        ax.set_ylim(0, PITCH_WIDTH_M)
        ax.set_title("Player Heatmap (Metric Spatial Fidelity)", color="white", fontsize=15, fontweight="bold", pad=12)
        ax.set_xlabel("Pitch Length (m)", color="white", fontsize=11)
        ax.set_ylabel("Pitch Width (m)", color="white", fontsize=11)
        ax.tick_params(colors="white")

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        return True

    def _render_empty(self, output_path: Path) -> bool:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 8), facecolor="#22312b")
        ax.set_facecolor("#22312b")
        rect = plt.Rectangle((0, 0), PITCH_LENGTH_M, PITCH_WIDTH_M, fill=False, ec="white", lw=2)
        ax.add_patch(rect)
        ax.axvline(x=PITCH_LENGTH_M / 2.0, color="white", lw=1.5)
        ax.text(
            PITCH_LENGTH_M / 2.0,
            PITCH_WIDTH_M / 2.0,
            "Insufficient Tracking Data for Heatmap",
            color="white",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.set_xlim(0, PITCH_LENGTH_M)
        ax.set_ylim(0, PITCH_WIDTH_M)
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=120, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        return True
