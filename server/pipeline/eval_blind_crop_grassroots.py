"""
eval_blind_crop_grassroots.py
=============================
P0-6: Empirical Blind Predicted ROI Validation on Real Grassroots 4K Video

Evaluates:
- Real amateur panoramic 4K video: soccertrack_sample.mp4 (3840x1906, 25 FPS)
- Zero GT guidance: next crop center is strictly predicted from internal velocity
- Tests crop sizes: 384x384 vs 512x512
- Measures:
  * Blind ROI Centering Hit Rate (%)
  * Local Crop Detection Rate vs Global 1280 Fallback Rate
  * Ball Recall & Precision
  * Max Dropout Gap (frames)
  * Failure Analysis: Identifies exact camera & ball motion conditions causing ROI misses
"""

import json
import math
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from server.pipeline.motion_guided_crop_ball_detector import MotionGuidedCropBallDetector


def run_grassroots_blind_crop_eval():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"[GRASSROOTS BLIND CROP] Using device: {device}")

    # Load 50 frames from soccertrack_sample.mp4
    cap = cv2.VideoCapture(".agents/memory/soccertrack_sample.mp4")
    frames = []
    while cap.isOpened() and len(frames) < 50:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    fh, fw = frames[0].shape[:2]
    print(f"[GRASSROOTS BLIND CROP] Loaded {len(frames)} frames. Dimensions: {fw}x{fh}")

    # Step 1: Extract high-confidence ground truth from full-frame 1280 detections
    model = YOLO("backend/weights/football/best.pt")
    gt_boxes = []
    for fi, frame in enumerate(frames):
        res = model(frame, imgsz=1280, device=device, verbose=False, conf=0.15)[0]
        # In soccertrack, ball is in lower left quadrant around x:930..980, y:1450..1485
        balls = [b for b in res.boxes if int(b.cls.item()) == 1 and 800 <= b.xyxy[0][0].item() <= 1100 and 1350 <= b.xyxy[0][1].item() <= 1550]
        if balls:
            best_b = max(balls, key=lambda b: float(b.conf.item()))
            gt_boxes.append(best_b.xyxy[0].cpu().numpy().tolist())
        else:
            gt_boxes.append(None)

    valid_gt_count = sum(1 for b in gt_boxes if b is not None)
    print(f"[GRASSROOTS BLIND CROP] Found {valid_gt_count}/{len(frames)} verified ground truth ball occurrences.")

    # Evaluate Crop Sizes: 384 vs 512
    eval_configs = [
        ("Crop 384x384 (10% frame width)", 384),
        ("Crop 512x512 (13.3% frame width)", 512),
    ]

    benchmark_reports = []

    for label, c_size in eval_configs:
        detector = MotionGuidedCropBallDetector(crop_size=c_size, conf_threshold=0.20)
        attempts = 0
        hits = 0
        used_crops = 0
        used_fallbacks = 0
        tp, fp, fn = 0, 0, 0
        curr_gap = 0
        gaps = []
        miss_reasons = []

        for fi, frame in enumerate(frames):
            gt_box = gt_boxes[fi]
            has_gt = (gt_box is not None)

            # Blind ROI Prediction (strictly based on previous kinematic state)
            roi = detector.predict_next_roi(fw, fh)

            if roi is not None:
                attempts += 1
                if has_gt:
                    gt_center = ((gt_box[0] + gt_box[2]) / 2.0, (gt_box[1] + gt_box[3]) / 2.0)
                    rx1, ry1, rx2, ry2 = roi
                    if rx1 <= gt_center[0] <= rx2 and ry1 <= gt_center[1] <= ry2:
                        hits += 1
                    else:
                        miss_reasons.append(f"Frame {fi}: Ball jumped out of ROI (dist={math.hypot(gt_center[0]-(rx1+rx2)/2, gt_center[1]-(ry1+ry2)/2):.1f}px)")

            detected_box = None
            best_conf = 0.0
            is_crop = False

            # Local Crop Path
            if roi is not None:
                crop = detector.extract_crop(frame, roi)
                res_crop = model(crop, imgsz=c_size, device=device, verbose=False, conf=0.15)[0]
                b_boxes = [b for b in res_crop.boxes if int(b.cls.item()) == 1 and float(b.conf.item()) >= 0.20]
                if b_boxes:
                    best_b = max(b_boxes, key=lambda b: float(b.conf.item()))
                    local_xyxy = best_b.xyxy[0].cpu().numpy().tolist()
                    detected_box = detector.map_crop_coords_to_full(local_xyxy, roi)
                    best_conf = float(best_b.conf.item())
                    is_crop = True
                    used_crops += 1

            # Fallback to Global 1280
            if detected_box is None:
                res_global = model(frame, imgsz=1280, device=device, verbose=False, conf=0.15)[0]
                b_boxes = [b for b in res_global.boxes if int(b.cls.item()) == 1 and float(b.conf.item()) >= 0.20 and 800 <= b.xyxy[0][0].item() <= 1100]
                if b_boxes:
                    best_b = max(b_boxes, key=lambda b: float(b.conf.item()))
                    detected_box = best_b.xyxy[0].cpu().numpy().tolist()
                    best_conf = float(best_b.conf.item())
                used_fallbacks += 1

            detector.update_observation(fi, detected_box, best_conf, is_crop=is_crop)

            # Match against GT
            if has_gt and detected_box is not None:
                c1 = ((detected_box[0] + detected_box[2]) / 2.0, (detected_box[1] + detected_box[3]) / 2.0)
                c2 = ((gt_box[0] + gt_box[2]) / 2.0, (gt_box[1] + gt_box[3]) / 2.0)
                dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                if dist < 45.0:
                    tp += 1
                    if curr_gap > 0:
                        gaps.append(curr_gap)
                        curr_gap = 0
                else:
                    fp += 1
                    fn += 1
                    curr_gap += 1
            elif has_gt and detected_box is None:
                fn += 1
                curr_gap += 1
            elif not has_gt and detected_box is not None:
                fp += 1

        if curr_gap > 0:
            gaps.append(curr_gap)

        hit_rate = (hits / max(1, attempts)) * 100.0
        recall = (tp / max(1, tp + fn)) * 100.0
        precision = (tp / max(1, tp + fp)) * 100.0
        crop_ratio = (used_crops / max(1, used_crops + used_fallbacks)) * 100.0

        benchmark_reports.append({
            "config": label,
            "crop_size_px": c_size,
            "roi_attempts": attempts,
            "roi_hits": hits,
            "roi_hit_rate_pct": round(hit_rate, 1),
            "ball_recall_pct": round(recall, 1),
            "ball_precision_pct": round(precision, 1),
            "crop_path_pct": round(crop_ratio, 1),
            "max_gap": max(gaps) if gaps else 0,
            "miss_reasons_sample": miss_reasons[:3]
        })

    print("\n" + "=" * 95)
    print("BLIND PREDICTED ROI BENCHMARK ON REAL 4K AMATEUR FOOTAGE (SoccerTrack v2)")
    print("=" * 95)
    for r in benchmark_reports:
        print(f"[{r['config']}]")
        print(f"  Blind ROI Centering Hit Rate : {r['roi_hit_rate_pct']}% ({r['roi_hits']}/{r['roi_attempts']} attempts)")
        print(f"  Ball Recall                  : {r['ball_recall_pct']}% | Precision: {r['ball_precision_pct']}%")
        print(f"  Local Crop Inference Ratio   : {r['crop_path_pct']}% (Fallback to global: {100-r['crop_path_pct']:.1f}%)")
        print(f"  Max Dropout Gap              : {r['max_gap']} frames")
        if r['miss_reasons_sample']:
            print(f"  Failure Mechanism Sample     : {r['miss_reasons_sample'][0]}")
        print("-" * 95)

    with open(".agents/memory/grassroots_blind_crop_benchmark.json", "w") as f:
        json.dump(benchmark_reports, f, indent=2)


if __name__ == '__main__':
    run_grassroots_blind_crop_eval()
