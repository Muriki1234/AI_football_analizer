import unittest
import time
from server.pipeline.jersey_voting import (
    extract_torso_roi,
    sanitize_jersey_number,
    TrackletJerseyVotingEngine,
)

class TestJerseyVoting(unittest.TestCase):
    def test_extract_torso_roi(self):
        # 100x200 box from (50, 100) to (150, 300)
        bbox = (50.0, 100.0, 150.0, 300.0)
        torso = extract_torso_roi(bbox, torso_ratio=0.45)
        # Height is 200, 45% is 90 -> y2 should be 190.0
        self.assertEqual(torso, (50.0, 100.0, 150.0, 190.0))

    def test_sanitize_jersey_number(self):
        self.assertEqual(sanitize_jersey_number("10"), 10)
        self.assertEqual(sanitize_jersey_number("  #7 "), 7)
        self.assertEqual(sanitize_jersey_number("99"), 99)
        self.assertEqual(sanitize_jersey_number("1"), 1)
        self.assertIsNone(sanitize_jersey_number("0"))
        self.assertIsNone(sanitize_jersey_number("100"))
        self.assertIsNone(sanitize_jersey_number("abc"))
        self.assertIsNone(sanitize_jersey_number(""))
        self.assertIsNone(sanitize_jersey_number("-5"))

    def test_empty_tracklet(self):
        engine = TrackletJerseyVotingEngine()
        res = engine.resolve_tracklet(track_id=99)
        self.assertEqual(res["status"], "UNRESOLVED")
        self.assertIsNone(res["number"])
        self.assertEqual(res["confidence"], 0.0)

    def test_noisy_ocr_consensus(self):
        engine = TrackletJerseyVotingEngine()
        # True jersey is #10
        # 15 frames of #10 (high confidence 0.85-0.95)
        # 3 frames of noisy OCR misread: #1 (0.45), #0 (0.40), #16 (0.50)
        for f in range(15):
            engine.add_observation(track_id=1, frame_idx=f, raw_number="10", confidence=0.90)
        engine.add_observation(track_id=1, frame_idx=15, raw_number="1", confidence=0.45)
        engine.add_observation(track_id=1, frame_idx=16, raw_number="0", confidence=0.40) # sanitized away as 0
        engine.add_observation(track_id=1, frame_idx=17, raw_number="16", confidence=0.50)

        res = engine.resolve_tracklet(track_id=1, min_support=3, confidence_threshold=0.70)
        self.assertEqual(res["number"], 10)
        self.assertEqual(res["status"], "CONFIRMED")
        self.assertGreater(res["confidence"], 0.80)
        self.assertEqual(res["support"], 15)

    def test_insufficient_support_tentative(self):
        engine = TrackletJerseyVotingEngine()
        # Only 1 observation of #7
        engine.add_observation(track_id=2, frame_idx=1, raw_number="7", confidence=0.88)
        res = engine.resolve_tracklet(track_id=2, min_support=3)
        self.assertEqual(res["number"], 7)
        self.assertEqual(res["status"], "TENTATIVE")
        self.assertEqual(res["support"], 1)

    def test_temporal_recency_weighting(self):
        engine = TrackletJerseyVotingEngine(decay_rate=0.05)
        # Far past: 3 frames of misread #8 at frame 0..2
        for f in range(3):
            engine.add_observation(track_id=3, frame_idx=f, raw_number="8", confidence=0.7)
        # Recent: 5 frames of clear #9 at frame 200..204
        for f in range(200, 205):
            engine.add_observation(track_id=3, frame_idx=f, raw_number="9", confidence=0.9)

        res = engine.resolve_tracklet(track_id=3)
        self.assertEqual(res["number"], 9)
        self.assertEqual(res["status"], "CONFIRMED")

    def test_performance_scale(self):
        # 22 players on the pitch x 500 frames = 11,000 observations
        engine = TrackletJerseyVotingEngine()
        t0 = time.perf_counter()
        for p in range(1, 23):
            jersey = p
            for f in range(0, 500, 10):
                engine.add_observation(track_id=p, frame_idx=f, raw_number=jersey, confidence=0.85)

        for p in range(1, 23):
            res = engine.resolve_tracklet(track_id=p)
            self.assertEqual(res["number"], p)
            self.assertEqual(res["status"], "CONFIRMED")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertLess(elapsed_ms, 50.0, f"Scale test took {elapsed_ms:.2f}ms, expected < 50ms")

if __name__ == "__main__":
    unittest.main()
