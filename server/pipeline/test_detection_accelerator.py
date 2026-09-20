"""
test_detection_accelerator.py - Automated Unit Test & Benchmark Suite
for LongVideoDetectionAccelerator, TacticalViewGater, and AdaptiveTemporalStrideController.
"""

import time
import unittest
import cv2
import numpy as np

from server.pipeline.detection_accelerator import (
    TacticalViewGater,
    AdaptiveTemporalStrideController,
    AsyncFramePrefetcher,
    LongVideoDetectionAccelerator,
)


class TestTacticalViewGater(unittest.TestCase):
    def setUp(self):
        self.gater = TacticalViewGater(min_grass_ratio=0.28)

    def _create_green_pitch_frame(self, h=720, w=1280):
        # BGR: Green field is roughly B=30, G=140, R=50 (HSV Hue ~ 45-60)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (35, 145, 55)
        return frame

    def _create_non_pitch_frame(self, h=720, w=1280):
        # BGR: Red/Blue crowd/bench shot
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (180, 40, 30)
        return frame

    def test_pitch_frame_classified_as_tactical(self):
        pitch = self._create_green_pitch_frame()
        is_tactical, ratio = self.gater.is_tactical_view(pitch)
        self.assertTrue(is_tactical)
        self.assertGreater(ratio, 0.8)

    def test_closeup_frame_classified_as_non_tactical(self):
        crowd = self._create_non_pitch_frame()
        is_tactical, ratio = self.gater.is_tactical_view(crowd)
        self.assertFalse(is_tactical)
        self.assertLess(ratio, 0.1)

    def test_batch_classify(self):
        pitch = self._create_green_pitch_frame(100, 100)
        crowd = self._create_non_pitch_frame(100, 100)
        results = self.gater.batch_classify([pitch, crowd, pitch])
        self.assertEqual(len(results), 3)
        self.assertTrue(results[0][0])
        self.assertFalse(results[1][0])
        self.assertTrue(results[2][0])

    def test_empty_frame_handling(self):
        empty = np.array([], dtype=np.uint8)
        is_tactical, ratio = self.gater.is_tactical_view(empty)
        self.assertFalse(is_tactical)
        self.assertEqual(ratio, 0.0)


class TestAdaptiveTemporalStrideController(unittest.TestCase):
    def setUp(self):
        self.controller = AdaptiveTemporalStrideController(
            base_stride=3,
            min_stride=2,
            max_stride=6,
            high_motion_threshold=8.0,
            low_motion_threshold=2.0,
        )

    def test_frame_0_detected_when_tactical(self):
        self.assertTrue(self.controller.evaluate_frame(0, is_tactical=True))

    def test_non_tactical_always_skipped(self):
        self.assertFalse(self.controller.evaluate_frame(0, is_tactical=False))
        self.assertFalse(self.controller.evaluate_frame(10, is_tactical=False))

    def test_high_motion_tightens_to_min_stride(self):
        self.controller.evaluate_frame(0, is_tactical=True, motion_magnitude=10.0)
        # Frame 1: delta=1 < 2 -> Skip
        self.assertFalse(self.controller.evaluate_frame(1, is_tactical=True, motion_magnitude=10.0))
        # Frame 2: delta=2 >= 2 -> Detect
        self.assertTrue(self.controller.evaluate_frame(2, is_tactical=True, motion_magnitude=10.0))

    def test_low_motion_extends_to_max_stride(self):
        self.controller.evaluate_frame(0, is_tactical=True, motion_magnitude=1.0)
        # Frames 1-5 should be skipped (max_stride = 6)
        for f in range(1, 6):
            self.assertFalse(self.controller.evaluate_frame(f, is_tactical=True, motion_magnitude=1.0))
        # Frame 6: delta=6 >= 6 -> Detect
        self.assertTrue(self.controller.evaluate_frame(6, is_tactical=True, motion_magnitude=1.0))

    def test_reset(self):
        self.controller.evaluate_frame(0, is_tactical=True)
        self.controller.reset()
        self.assertEqual(self.controller.last_detection_frame, -1)


class TestAsyncFramePrefetcher(unittest.TestCase):
    def test_prefetch_queue_streaming(self):
        total_test_frames = 25
        frame_counter = 0

        def mock_provider():
            nonlocal frame_counter
            if frame_counter >= total_test_frames:
                return None
            frame_counter += 1
            return np.ones((50, 50, 3), dtype=np.uint8) * frame_counter

        prefetcher = AsyncFramePrefetcher(mock_provider, max_buffered_frames=10)
        prefetcher.start()

        collected_indices = []
        for batch in prefetcher.stream_batches(batch_size=8):
            for frame, idx in batch:
                collected_indices.append(idx)

        prefetcher.stop()
        self.assertEqual(len(collected_indices), total_test_frames)
        self.assertEqual(collected_indices, list(range(total_test_frames)))

    def test_stream_video_chunks_prefetch(self):
        import tempfile
        import os
        from server.pipeline.detection_accelerator import stream_video_chunks_prefetch

        # Generate small temporary video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(tmp_path, fourcc, 25.0, (64, 64))
            for i in range(45):
                frame = np.full((64, 64, 3), fill_value=(i % 255), dtype=np.uint8)
                out.write(frame)
            out.release()

            # Stream chunks of 15 frames
            chunks_received = []
            total_frames = 0
            for start_idx, chunk in stream_video_chunks_prefetch(tmp_path, chunk_size=15, prefetch_queue_size=2):
                chunks_received.append((start_idx, len(chunk)))
                total_frames += len(chunk)

            self.assertEqual(total_frames, 45)
            self.assertEqual(chunks_received, [(0, 15), (15, 15), (30, 15)])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestLongVideoDetectionAccelerator(unittest.TestCase):
    def setUp(self):
        self.accelerator = LongVideoDetectionAccelerator(
            min_grass_ratio=0.28,
            base_stride=3,
            min_stride=2,
            max_stride=6,
        )

    def test_mixed_broadcast_match_schedule(self):
        # 100-frame synthetic match segment:
        # Frames 0-49: Open tactical play (pitch green) with steady motion
        # Frames 50-79: Replay / closeup cutaway (red crowd/bench)
        # Frames 80-99: Fast break attack (pitch green) with high motion
        frames = []
        motions = []

        pitch_frame = np.zeros((180, 320, 3), dtype=np.uint8)
        pitch_frame[:, :] = (35, 145, 55)

        crowd_frame = np.zeros((180, 320, 3), dtype=np.uint8)
        crowd_frame[:, :] = (180, 40, 30)

        for i in range(50):
            frames.append(pitch_frame)
            motions.append(1.5)  # low motion -> stride 6

        for i in range(30):
            frames.append(crowd_frame)
            motions.append(4.0)

        for i in range(20):
            frames.append(pitch_frame)
            motions.append(9.5)  # high motion -> stride 2

        plan = self.accelerator.plan_detection_schedule(frames, motions)

        self.assertEqual(plan["total_frames"], 100)
        self.assertGreater(plan["skipped_non_tactical_count"], 25)
        # With tactical filtering and adaptive stride, detected frames should be <= 25 (>= 75% savings)
        self.assertLessEqual(plan["detected_frames_count"], 25)
        self.assertGreaterEqual(plan["compute_reduction_ratio"], 0.75)


class TestDetectionAcceleratorBenchmarks(unittest.TestCase):
    """
    Performance Benchmark Suite:
    Explicitly scoped as Algorithm-only Benchmark (in-memory thumbnail classification
    and schedule planning, excluding video disk I/O and GPU model inference).
    """

    def test_algorithm_only_benchmark_tactical_view_gater(self):
        gater = TacticalViewGater()
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        test_frame[:, :] = (35, 145, 55)

        # Warmup
        for _ in range(50):
            gater.is_tactical_view(test_frame)

        t0 = time.perf_counter()
        n_iters = 1000
        for _ in range(n_iters):
            gater.is_tactical_view(test_frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_frame_ms = elapsed_ms / n_iters

        print(f"\n[Algorithm-only Benchmark] TacticalViewGater: {n_iters} frames in {elapsed_ms:.1f}ms "
              f"({per_frame_ms:.3f}ms/frame, throughput: {n_iters / (elapsed_ms / 1000):.0f} FPS)")
        self.assertLess(per_frame_ms, 0.5, "TacticalViewGater must execute in < 0.5ms per frame")

    def test_algorithm_only_benchmark_schedule_planning(self):
        accelerator = LongVideoDetectionAccelerator()
        pitch_frame = np.zeros((180, 320, 3), dtype=np.uint8)
        pitch_frame[:, :] = (35, 145, 55)

        n_frames = 5000
        frames = [pitch_frame] * n_frames
        motions = [3.0] * n_frames

        t0 = time.perf_counter()
        plan = accelerator.plan_detection_schedule(frames, motions)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"\n[Algorithm-only Benchmark] Schedule Planning: {n_frames} frames in {elapsed_ms:.1f}ms "
              f"({elapsed_ms / n_frames:.4f}ms/frame, savings: {plan['compute_reduction_pct']})")
        self.assertLess(elapsed_ms, 1000.0, "Schedule planning for 5000 frames must finish in < 1.0s")


if __name__ == "__main__":
    unittest.main()
