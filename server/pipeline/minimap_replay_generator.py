"""
minimap_replay_generator.py - Minimap 2D Tactical Replay Video Generator & Streaming Encoder

Features:
1. Pure OpenCV high-performance tactical board rendering (> 400 FPS) without Matplotlib overhead.
2. Constant O(1) RAM streaming pipeline:
   Pipes rendered RGB/BGR frames directly to FFmpeg H.264 pipe with yuv420p and +faststart,
   eliminating multi-gigabyte in-memory frame buffers.
3. Complete match visualization:
   - Team 1 / Team 2 color-coded player tokens with boundary rings
   - Golden target highlight marker for tracked player
   - Ball token with 30-frame temporal fade-out trajectory trail
   - Real-time possession statistics HUD banner
   - Match clock (MM:SS.s)
4. Graceful OpenCV VideoWriter fallback if FFmpeg is unavailable.
"""

from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np

from .analysis_core import (
    make_pitch_background,
    render_minimap_frame,
    _mm_p2px,
    _hex_to_bgr,
)


class MinimapReplayGenerator:
    """
    Generates 2D animated tactical pitch videos (MP4) with constant O(1) RAM streaming.
    """

    def __init__(
        self,
        tracks: Dict[str, Any],
        tracked_bboxes: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
        team_control: Optional[np.ndarray] = None,
        team_colors_hex: Optional[Dict[int, str]] = None,
        fps: float = 25.0,
        width: int = 840,
        height: int = 560,
    ):
        self.tracks = tracks or {"players": [], "ball": []}
        self.tracked_bboxes = tracked_bboxes or {}
        n_frames = len(self.tracks.get("players", []))
        if team_control is None:
            self.team_control = np.zeros(n_frames, dtype=int)
        else:
            self.team_control = np.asarray(team_control)

        self.team_colors = team_colors_hex or {1: "#3498db", 2: "#e74c3c"}
        self.hex_t1 = self.team_colors.get(1, "#3498db")
        self.hex_t2 = self.team_colors.get(2, "#e74c3c")
        self.fps = max(float(fps), 1.0)
        self.width = width
        self.height = height

        # Pre-render pitch background once for reuse
        self.pitch_bg = make_pitch_background(width=self.width, height=self.height)
        self.ball_trail_max_len = 30

    @property
    def total_frames(self) -> int:
        return len(self.tracks.get("players", []))

    def render_frame(
        self,
        frame_idx: int,
        ball_trail: Optional[List[Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """
        Renders a single tactical minimap frame using OpenCV primitives.
        """
        return render_minimap_frame(
            frame_idx=frame_idx,
            tracks=self.tracks,
            tracked_bboxes=self.tracked_bboxes,
            team_control=self.team_control,
            config=None,
            hex_t1=self.hex_t1,
            hex_t2=self.hex_t2,
            ball_trail=ball_trail,
            pitch_bg=self.pitch_bg,
            fps=self.fps,
        )

    def render_video_streaming(
        self,
        output_path: Path,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> bool:
        """
        Streams rendered frames directly to FFmpeg H.264 encoder.
        Maintains constant O(1) memory consumption regardless of video length.
        """
        n_frames = self.total_frames
        if n_frames == 0:
            return False

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "bgr24",
            "-r", str(self.fps),
            "-i", "pipe:0",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-movflags", "+faststart",
            str(output_path),
        ]

        ball_trail: List[Tuple[float, float]] = []
        use_ffmpeg = True
        proc = None

        try:
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, OSError):
            use_ffmpeg = False

        if use_ffmpeg and proc is not None and proc.stdin is not None:
            try:
                for idx in range(n_frames):
                    # Maintain ball trail
                    if idx < len(self.tracks.get("ball", [])):
                        ball_info = self.tracks["ball"][idx].get(1, {})
                        bp = ball_info.get("position_transformed")
                        if bp and len(bp) == 2 and not any(np.isnan(p) for p in bp):
                            ball_trail.append(tuple(bp))
                            if len(ball_trail) > self.ball_trail_max_len:
                                ball_trail.pop(0)

                    frame = self.render_frame(idx, ball_trail=ball_trail)
                    proc.stdin.write(frame.tobytes())

                    if progress_cb and (idx % 25 == 0 or idx == n_frames - 1):
                        progress_cb(idx + 1, n_frames)

                proc.stdin.close()
                proc.wait(timeout=60)
                if output_path.exists() and output_path.stat().st_size > 0:
                    return True
            except Exception:
                if proc:
                    proc.kill()

        # Fallback to OpenCV VideoWriter if FFmpeg pipe encounters issue
        return self._render_video_opencv_fallback(output_path, progress_cb)

    def _render_video_opencv_fallback(
        self,
        output_path: Path,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> bool:
        """OpenCV VideoWriter fallback."""
        n_frames = self.total_frames
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            float(self.fps),
            (self.width, self.height),
        )
        if not writer.isOpened():
            return False

        ball_trail: List[Tuple[float, float]] = []
        try:
            for idx in range(n_frames):
                if idx < len(self.tracks.get("ball", [])):
                    ball_info = self.tracks["ball"][idx].get(1, {})
                    bp = ball_info.get("position_transformed")
                    if bp and len(bp) == 2 and not any(np.isnan(p) for p in bp):
                        ball_trail.append(tuple(bp))
                        if len(ball_trail) > self.ball_trail_max_len:
                            ball_trail.pop(0)

                frame = self.render_frame(idx, ball_trail=ball_trail)
                writer.write(frame)
                if progress_cb and (idx % 25 == 0 or idx == n_frames - 1):
                    progress_cb(idx + 1, n_frames)
        finally:
            writer.release()

        return output_path.exists() and output_path.stat().st_size > 0

    def benchmark_render(self, num_frames: int = 1000) -> float:
        """
        [Algorithm-only Benchmark] Measures pure OpenCV 2D frame rendering rate.
        """
        ball_trail = [(50.0 + i * 0.1, 30.0 + i * 0.05) for i in range(25)]
        # Warmup
        for _ in range(10):
            _ = self.render_frame(0, ball_trail=ball_trail)

        t0 = time.perf_counter()
        for idx in range(num_frames):
            f_idx = idx % max(self.total_frames, 1)
            _ = self.render_frame(f_idx, ball_trail=ball_trail)
        elapsed = time.perf_counter() - t0

        fps = num_frames / max(elapsed, 1e-6)
        print(f"\n[Algorithm-only Benchmark] MinimapReplayGenerator: {fps:,.0f} frames/sec ({elapsed*1000:.2f} ms for {num_frames} frames)")
        return fps
