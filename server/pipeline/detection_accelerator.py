"""
detection_accelerator.py - Long-Video Detection Acceleration Engine

Architectural Pillars:
1. TacticalViewGater: Sub-millisecond broadcast frame classification (HSV grass field ratio)
   skipping non-gameplay closeups, replays, and crowd shots.
2. AdaptiveTemporalStrideController: Motion & dynamics-aware stride modulation (2-6 frames)
   extrapolating steady-motion tracks while tightening detection on rapid transitions.
3. AsyncFramePrefetcher: Threaded double-buffered frame pipeline eliminating GPU starvation
   by overlapping video decoding with model tensor inference.
4. LongVideoDetectionAccelerator: Unified pipeline coordinator providing transparent
   throughput metrics and inference reduction ratios.
"""

import logging
import os
import queue
import threading
import time
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TacticalViewGater:
    """
    Sub-millisecond broadcast camera classifier.
    Distinguishes wide tactical broadcast footage (tactical camera) from
    closeups, audience shots, bench cameras, and broadcast replay overlays.
    """

    def __init__(
        self,
        min_grass_ratio: float = 0.20,
        hsv_lower: Tuple[int, int, int] = (28, 25, 25),
        hsv_upper: Tuple[int, int, int] = (88, 255, 255),
        downsample_size: Tuple[int, int] = (160, 90),
    ):
        self.min_grass_ratio = min_grass_ratio
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.downsample_size = downsample_size

    def compute_grass_ratio_from_thumb(self, small: np.ndarray) -> float:
        """Computes grass ratio directly from an already downsampled thumbnail."""
        if small is None or small.size == 0:
            return 0.0
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        return float(np.count_nonzero(mask) / mask.size)

    def compute_grass_ratio(self, frame: np.ndarray) -> float:
        """Computes the fraction of pixels matching the pitch grass color mask."""
        if frame is None or frame.size == 0:
            return 0.0

        # Downsample to thumbnail for sub-millisecond evaluation
        small = cv2.resize(frame, self.downsample_size, interpolation=cv2.INTER_NEAREST)
        return self.compute_grass_ratio_from_thumb(small)

    def is_tactical_view(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Returns (is_tactical, grass_ratio).
        If True, frame is an open-field tactical broadcast shot worthy of YOLO inference.
        If False, frame is a closeup / crowd / off-pitch cutaway that can skip heavy detection.
        """
        ratio = self.compute_grass_ratio(frame)
        return (ratio >= self.min_grass_ratio), round(ratio, 4)

    def batch_classify(self, frames: List[np.ndarray]) -> List[Tuple[bool, float]]:
        """Vectorized / iterated classification across a frame chunk."""
        return [self.is_tactical_view(f) for f in frames]


class AdaptiveTemporalStrideController:
    """
    Dynamically modulates detection stride based on frame dynamics and tactical state.
    - Base stride (e.g. 3) in normal play.
    - Extended stride (up to 6) during steady, low-motion ball possession.
    - Tightened stride (down to 1 or 2) during high camera pan or rapid ball transitions.
    - Full skip (stride = infinity) during non-tactical broadcast cutaways.
    """

    def __init__(
        self,
        base_stride: int = 3,
        min_stride: int = 2,
        max_stride: int = 6,
        high_motion_threshold: float = 8.0,
        low_motion_threshold: float = 2.0,
    ):
        self.base_stride = base_stride
        self.min_stride = min_stride
        self.max_stride = max_stride
        self.high_motion_threshold = high_motion_threshold
        self.low_motion_threshold = low_motion_threshold

        self.last_detection_frame = -1
        self.current_stride = base_stride

    def evaluate_frame(
        self,
        frame_idx: int,
        is_tactical: bool,
        motion_magnitude: float = 0.0,
    ) -> bool:
        """
        Determines whether the current frame must trigger a heavy YOLO detection.
        """
        # Always detect frame 0 if tactical
        if self.last_detection_frame < 0:
            if is_tactical:
                self.last_detection_frame = frame_idx
                return True
            return False

        if not is_tactical:
            # Non-tactical cutaways do not trigger tactical detections
            return False

        # Compute dynamic stride based on motion dynamics
        if motion_magnitude >= self.high_motion_threshold:
            self.current_stride = self.min_stride
        elif motion_magnitude <= self.low_motion_threshold:
            self.current_stride = self.max_stride
        else:
            self.current_stride = self.base_stride

        delta = frame_idx - self.last_detection_frame
        if delta >= self.current_stride:
            self.last_detection_frame = frame_idx
            return True

        return False

    def reset(self):
        self.last_detection_frame = -1
        self.current_stride = self.base_stride


class AsyncFramePrefetcher:
    """
    Double-buffered background frame reader.
    Eliminates GPU wait stalls by reading ahead in a dedicated producer thread.
    """

    def __init__(self, frame_provider: Callable[[], Optional[np.ndarray]], max_buffered_frames: int = 120):
        self.frame_provider = frame_provider
        self.max_buffered_frames = max_buffered_frames
        self.queue: queue.Queue = queue.Queue(maxsize=max_buffered_frames)
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.total_frames_read = 0

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_provider()
                if frame is None:
                    # End of stream sentinel
                    self.queue.put((None, self.total_frames_read), timeout=2.0)
                    break
                self.queue.put((frame, self.total_frames_read), timeout=2.0)
                self.total_frames_read += 1
            except queue.Full:
                continue
            except Exception as e:
                self.queue.put((e, self.total_frames_read))
                break

    def start(self):
        self.stop_event.clear()
        self.total_frames_read = 0
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def stream_batches(self, batch_size: int) -> Generator[List[Tuple[np.ndarray, int]], None, None]:
        """Yields batches of frames from the prefetch queue."""
        batch: List[Tuple[np.ndarray, int]] = []
        while True:
            item, idx = self.queue.get()
            if isinstance(item, Exception):
                raise item
            if item is None:
                if batch:
                    yield batch
                break

            batch.append((item, idx))
            if len(batch) >= batch_size:
                yield batch
                batch = []

    def stop(self):
        self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)


class LongVideoDetectionAccelerator:
    """
    Unified Detection Acceleration Pipeline.
    Combines TacticalViewGater + AdaptiveTemporalStrideController to drastically
    reduce unnecessary YOLO model forwards on long broadcast match videos.
    """

    def __init__(
        self,
        min_grass_ratio: float = 0.20,
        base_stride: int = 3,
        min_stride: int = 2,
        max_stride: int = 5,
    ):
        self.gater = TacticalViewGater(min_grass_ratio=min_grass_ratio)
        self.stride_controller = AdaptiveTemporalStrideController(
            base_stride=base_stride,
            min_stride=min_stride,
            max_stride=max_stride,
        )

    def plan_chunk(
        self,
        chunk: List[np.ndarray],
        start_idx: int = 0,
        prev_thumb: Optional[np.ndarray] = None,
        max_consecutive_skip: int = 25,
    ) -> Tuple[List[int], Optional[np.ndarray], Dict[str, Any]]:
        """
        Fast microsecond planner for a chunk of video frames.
        Returns:
            det_local_indices: list of frame indices [0..len(chunk)-1] to run YOLO on.
            last_thumb: thumbnail of the last frame in chunk (to pass as prev_thumb to next chunk).
            stats: dict of metrics (savings, tactical count, etc.).
        """
        if not chunk:
            return [], prev_thumb, {"total_frames": 0, "detected_count": 0, "reduction_pct": 0.0}

        det_local: List[int] = []
        last_valid_thumb = prev_thumb
        skipped_non_tactical = 0

        for j, frame in enumerate(chunk):
            global_fi = start_idx + j
            if frame is None or frame.size == 0:
                continue

            # 160x90 thumbnail for sub-millisecond evaluation
            small = cv2.resize(frame, self.gater.downsample_size, interpolation=cv2.INTER_NEAREST)
            ratio = self.gater.compute_grass_ratio_from_thumb(small)
            is_tactical = (ratio >= self.gater.min_grass_ratio)

            if last_valid_thumb is not None:
                diff = cv2.absdiff(last_valid_thumb, small)
                motion = float(np.mean(diff))
            else:
                motion = 3.0
            last_valid_thumb = small

            last_det_global = self.stride_controller.last_detection_frame
            frames_since_det = (global_fi - last_det_global) if last_det_global >= 0 else 9999

            if not is_tactical:
                skipped_non_tactical += 1
                # Safety net: never skip more than max_consecutive_skip frames continuously
                if frames_since_det >= max_consecutive_skip:
                    det_local.append(j)
                    self.stride_controller.last_detection_frame = global_fi
                continue

            should_detect = self.stride_controller.evaluate_frame(
                frame_idx=global_fi,
                is_tactical=is_tactical,
                motion_magnitude=motion,
            )

            if should_detect or frames_since_det >= max_consecutive_skip or (not det_local and global_fi == 0):
                if not det_local or det_local[-1] != j:
                    det_local.append(j)
                self.stride_controller.last_detection_frame = global_fi

        if not det_local and chunk:
            det_local.append(0)
            self.stride_controller.last_detection_frame = start_idx

        stats = {
            "total_frames": len(chunk),
            "detected_count": len(det_local),
            "skipped_non_tactical": skipped_non_tactical,
            "reduction_pct": round((1.0 - len(det_local) / max(len(chunk), 1)) * 100, 1),
        }
        return det_local, last_valid_thumb, stats

    def plan_detection_schedule(
        self,
        frames: List[np.ndarray],
        motion_signals: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates an entire chunk or stream of frames and returns detection execution plan.
        """
        total_frames = len(frames)
        motions = motion_signals or [3.0] * total_frames

        detected_indices: List[int] = []
        skipped_non_tactical: List[int] = []
        skipped_adaptive_stride: List[int] = []
        grass_ratios: List[float] = []

        self.stride_controller.reset()

        for idx, (frame, motion) in enumerate(zip(frames, motions)):
            is_tactical, ratio = self.gater.is_tactical_view(frame)
            grass_ratios.append(ratio)

            if not is_tactical:
                skipped_non_tactical.append(idx)
                continue

            should_detect = self.stride_controller.evaluate_frame(
                frame_idx=idx,
                is_tactical=is_tactical,
                motion_magnitude=motion,
            )

            if should_detect:
                detected_indices.append(idx)
            else:
                skipped_adaptive_stride.append(idx)

        total_detections = len(detected_indices)
        savings_ratio = round((1.0 - total_detections / max(total_frames, 1)) * 100, 1)

        return {
            "total_frames": total_frames,
            "detected_frames_count": total_detections,
            "detected_indices": detected_indices,
            "skipped_non_tactical_count": len(skipped_non_tactical),
            "skipped_adaptive_stride_count": len(skipped_adaptive_stride),
            "compute_reduction_pct": f"{savings_ratio}%",
            "compute_reduction_ratio": round(savings_ratio / 100.0, 3),
            "average_grass_ratio": round(float(np.mean(grass_ratios)), 3) if grass_ratios else 0.0,
        }


def stream_video_chunks_prefetch(
    video_path: str,
    chunk_size: int = 500,
    prefetch_queue_size: int = 1,
) -> Generator[Tuple[int, List[np.ndarray]], None, None]:
    """
    Asynchronous chunked video stream generator.
    Decodes the next chunk in a background thread while the caller processes the current chunk.
    Eliminates synchronous cv2.VideoCapture decode stalls in long-video processing.
    """
    chunk_queue: queue.Queue = queue.Queue(maxsize=prefetch_queue_size)
    stop_event = threading.Event()

    def _producer():
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        start_idx = 0
        try:
            if not cap.isOpened():
                raise RuntimeError(f"VideoCapture could not open video: {video_path}")

            while not stop_event.is_set():
                chunk = []
                for _ in range(chunk_size):
                    if stop_event.is_set():
                        break
                    ret, frame = cap.read()
                    if not ret:
                        break
                    chunk.append(frame)
                if not chunk:
                    break

                while not stop_event.is_set():
                    try:
                        chunk_queue.put((start_idx, chunk), timeout=0.2)
                        break
                    except queue.Full:
                        continue

                start_idx += len(chunk)
                if len(chunk) < chunk_size:
                    break
        except Exception as exc:
            while not stop_event.is_set():
                try:
                    chunk_queue.put((exc, None), timeout=0.2)
                    break
                except queue.Full:
                    continue
        finally:
            cap.release()
            while not stop_event.is_set():
                try:
                    chunk_queue.put((None, None), timeout=0.2)
                    break
                except queue.Full:
                    continue

    producer_thread = threading.Thread(target=_producer, daemon=True)
    producer_thread.start()

    try:
        while True:
            item = chunk_queue.get()
            start_idx, chunk = item
            if isinstance(start_idx, Exception):
                raise start_idx
            if start_idx is None:
                break
            yield start_idx, chunk
    finally:
        stop_event.set()
        producer_thread.join(timeout=2.0)


def stream_video_chunks_safe(
    video_path: str,
    chunk_size: int = 500,
    prefetch_queue_size: int = 1,
) -> Generator[Tuple[int, List[np.ndarray]], None, None]:
    """
    Robust chunk generator: tries stream_video_chunks_prefetch first.
    If prefetching is disabled or raises any exception at any point, cleanly logs a warning
    and seamlessly falls back to synchronous cv2.VideoCapture without failing the pipeline.
    """
    use_prefetch = (
        os.environ.get("ENABLE_ASYNC_PREFETCH", "1").strip().lower()
        not in ("0", "false", "no")
    )

    if not use_prefetch:
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        start_idx = 0
        try:
            while True:
                chunk = []
                for _ in range(chunk_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    chunk.append(frame)
                if not chunk:
                    break
                yield start_idx, chunk
                start_idx += len(chunk)
                if len(chunk) < chunk_size:
                    break
        finally:
            cap.release()
        return

    next_frame_needed = 0
    try:
        for start_idx, chunk in stream_video_chunks_prefetch(
            video_path, chunk_size=chunk_size, prefetch_queue_size=prefetch_queue_size
        ):
            yield start_idx, chunk
            next_frame_needed = start_idx + len(chunk)
    except Exception as exc:
        logger.warning(
            f"[stream_video_chunks_safe] Async prefetch encountered error ({exc!r}), "
            f"seamlessly falling back to synchronous reader from frame {next_frame_needed}."
        )
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        try:
            if next_frame_needed > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(next_frame_needed))
                actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if 0 <= actual_pos < next_frame_needed:
                    for _ in range(next_frame_needed - actual_pos):
                        cap.grab()
            start_idx = next_frame_needed
            while True:
                chunk = []
                for _ in range(chunk_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    chunk.append(frame)
                if not chunk:
                    break
                yield start_idx, chunk
                start_idx += len(chunk)
                if len(chunk) < chunk_size:
                    break
        finally:
            cap.release()
