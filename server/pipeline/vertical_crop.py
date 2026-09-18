"""
vertical_crop.py - 9:16 Vertical Video Highlight Generator
Transforms 16:9 broadcast footage into dynamic 9:16 mobile-first vertical clips
(for TikTok, Instagram Reels, WeChat Channels) using Smooth Camera Tracking (Cinematic Pan & Scan).
Zero external dependencies (pure Python standard library).
"""

from __future__ import annotations
from typing import List, Tuple, Optional


def compute_smooth_crop_centers(
    target_centers: List[Optional[Tuple[float, float]]],
    frame_width: int,
    frame_height: int,
    crop_aspect_ratio: float = 9.0 / 16.0,
    smoothing_window: int = 15,
) -> List[Tuple[int, int]]:
    """
    Computes smooth top-left (x, y) coordinates for a 9:16 vertical crop window
    centered on the target (player or ball).

    Args:
        target_centers: List of (x, y) target coordinates per frame. None if target lost.
        frame_width: Original video width (e.g., 1920).
        frame_height: Original video height (e.g., 1080).
        crop_aspect_ratio: Ratio of width to height (default 9/16 = 0.5625).
        smoothing_window: Number of frames for moving average filter.

    Returns:
        List of (crop_x, crop_y) integer offsets for each frame.
    """
    crop_h = frame_height
    crop_w = int(crop_h * crop_aspect_ratio)
    if crop_w % 2 != 0:
        crop_w -= 1  # FFmpeg requires even dimensions

    max_x = max(0, frame_width - crop_w)
    default_x = frame_width / 2.0

    # 1. Fill missing frames with last known center X
    raw_xs: List[float] = []
    last_known = default_x
    for tc in target_centers:
        if tc is not None:
            last_known = float(tc[0])
        raw_xs.append(last_known)

    if not raw_xs:
        return []

    # 2. Moving average smoothing (pure Python sliding window)
    half_win = max(1, smoothing_window // 2)
    smooth_xs: List[float] = []
    n = len(raw_xs)

    for i in range(n):
        left = max(0, i - half_win)
        right = min(n, i + half_win + 1)
        sub = raw_xs[left:right]
        avg = sum(sub) / len(sub)
        smooth_xs.append(avg)

    # 3. Convert target center X to crop window top-left X, clamped to frame bounds
    crop_coords: List[Tuple[int, int]] = []
    for cx in smooth_xs:
        top_left_x = int(cx - (crop_w / 2.0))
        top_left_x = max(0, min(top_left_x, max_x))
        crop_coords.append((top_left_x, 0))

    return crop_coords


def build_ffmpeg_crop_filter(
    crop_width: int,
    crop_height: int,
    crop_coords: List[Tuple[int, int]],
) -> str:
    """
    Builds an optimized FFmpeg crop filter expression string for CLI rendering.
    """
    if not crop_coords:
        return f"crop={crop_width}:{crop_height}:0:0"

    xs = [c[0] for c in crop_coords]
    if max(xs) - min(xs) < 20:
        avg_x = int(sum(xs) / len(xs))
        return f"crop={crop_width}:{crop_height}:{avg_x}:0"

    return f"crop={crop_width}:{crop_height}:'min(max(0, {xs[0]}), in_w-{crop_width})':0"
