"""
eval_blind_predicted_crop.py
============================
Strict Blind Empirical Evaluation of Predicted ROI Ball Detection
(No Ground Truth Guidance in Production Inference Loop)

Evaluates:
- Predicted ROI Hit Rate: Does the blind predicted 384x384 window actually contain the true ball?
- Detection Recall & Precision
- Track Continuity & Max Gap Length
- Fallback Trigger Frequency (when crop misses)
- Latency (ms/frame) & FPS
"""

import cv2
import json
import time
import torch
import numpy as np
from ultralytics import YOLO
from server.pipeline.motion_guided_crop_ball_detector import MotionGuidedCropBallDetector

def is_point_inside_box(pt, box):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]

def center_dist(b1, b2):
    c1 = ((b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0)
    c2 = ((b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0)
    return np.hypot(c1[0] - c2[0], c1[1] - c2[1])

def run_blind_crop_evaluation(video_path: str, gt_path: str, model_path: str, start_f=600, end_f=750):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"[BLIND EVAL] Running strictly blind ROI crop test on {start_f}..{end_f-1} ({end_f-start_f} frames)...")

    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    gt_balls = gt_data['ball_tracks'][start_f:end_f]

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frames = [cap.read()[1] for _ in range(end_f - start_f)]
    cap.release()

    model = YOLO(model_path)
    detector = MotionGuidedCropBallDetector(crop_size=384, conf_threshold=0.35)

    # Telemetry
    roi_attempts = 0
    roi_hits = 0  # True ball center was inside predicted ROI
    crop_inferences = 0
    global_fallbacks = 0
    
    tp, fp, fn = 0, 0, 0
    gaps = []
    curr_gap = 0
    predictions = []

    t0 = time.perf_counter()

    for fi, frame in enumerate(frames):
        h, w = frame.shape[:2]
        
        # 1. Blind prediction from detector's own internal kinematics ONLY
        roi = detector.predict_next_roi(w, h)
        
        # Check ROI Hit Rate against GT (for auditing purposes only, NOT used by detector)
        has_gt = bool(gt_balls[fi] and (1 in gt_balls[fi] or '1' in gt_balls[fi]))
        gt_box = (gt_balls[fi].get(1) or gt_balls[fi].get('1', {})).get('bbox') if has_gt else None
        
        if roi is not None:
            roi_attempts += 1
            if has_gt:
                gt_center = ((gt_box[0] + gt_box[2]) / 2.0, (gt_box[1] + gt_box[3]) / 2.0)
                if is_point_inside_box(gt_center, roi):
                    roi_hits += 1

        # 2. Execution phase
        detected_box = None
        best_conf = 0.0
        used_crop = False

        if roi is not None:
            # Execute local high-res crop
            crop = detector.extract_crop(frame, roi)
            crop_inferences += 1
            res_crop = model(crop, imgsz=384, device=device, verbose=False, conf=0.25)[0]
            b_boxes = [b for b in res_crop.boxes if int(b.cls.item()) == 1 and float(b.conf.item()) >= 0.35]
            if b_boxes:
                b_best = max(b_boxes, key=lambda b: float(b.conf.item()))
                best_conf = float(b_best.conf.item())
                local_xyxy = b_best.xyxy[0].cpu().numpy().tolist()
                detected_box = detector.map_crop_coords_to_full(local_xyxy, roi)
                used_crop = True

        if detected_box is None:
            # Fallback: global search at moderate resolution (960)
            global_fallbacks += 1
            res_global = model(frame, imgsz=960, device=device, verbose=False, conf=0.25)[0]
            b_boxes = [b for b in res_global.boxes if int(b.cls.item()) == 1 and float(b.conf.item()) >= 0.35]
            if b_boxes:
                b_best = max(b_boxes, key=lambda b: float(b.conf.item()))
                best_conf = float(b_best.conf.item())
                detected_box = b_best.xyxy[0].cpu().numpy().tolist()

        # Update detector state strictly from detection output
        obs = detector.update_observation(fi, detected_box, best_conf, is_crop=used_crop)
        predictions.append(detected_box)

        # Accuracy evaluation against GT
        if has_gt and detected_box is not None:
            if center_dist(detected_box, gt_box) < 35.0:
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

    elapsed = time.perf_counter() - t0
    fps = len(frames) / elapsed
    ms_per_frame = (elapsed / len(frames)) * 1000.0

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    roi_hit_rate = roi_hits / max(1, roi_attempts)

    report = {
        "total_frames": len(frames),
        "gt_ball_frames": sum(1 for b in gt_balls if b and (1 in b or '1' in b)),
        "roi_attempts": roi_attempts,
        "roi_hits": roi_hits,
        "roi_hit_rate_pct": round(roi_hit_rate * 100, 1),
        "crop_inferences": crop_inferences,
        "global_fallbacks": global_fallbacks,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "max_gap_frames": max(gaps) if gaps else 0,
        "avg_gap_frames": round(float(np.mean(gaps)), 1) if gaps else 0.0,
        "ms_per_frame": round(ms_per_frame, 1),
        "fps": round(fps, 1),
    }

    print("\n" + "="*70)
    print("BLIND PREDICTED ROI PRODUCTION BENCHMARK REPORT")
    print("="*70)
    print(f"Total Frames Analyzed     : {report['total_frames']} (GT Ball Frames: {report['gt_ball_frames']})")
    print(f"Blind ROI Prediction Attempts: {report['roi_attempts']}")
    print(f"Blind ROI Hit Rate        : {report['roi_hits']} / {report['roi_attempts']} ({report['roi_hit_rate_pct']}%)")
    print(f"Crop vs Global Ratio      : {report['crop_inferences']} Crops ({report['crop_inferences']/report['total_frames']*100:.1f}%) | {report['global_fallbacks']} Global Fallbacks")
    print(f"Detection Performance     : TP={tp}, FP={fp}, FN={fn}")
    print(f"  Recall                  : {report['recall']*100:.1f}%")
    print(f"  Precision               : {report['precision']*100:.1f}%")
    print(f"  F1 Score                : {report['f1']:.3f}")
    print(f"  Max Dropout Gap         : {report['max_gap_frames']} frames ({report['max_gap_frames']/25:.2f}s)")
    print(f"Throughput & Latency      : {report['ms_per_frame']} ms/frame ({report['fps']} FPS)")
    print("="*70)

    with open(".agents/memory/blind_crop_benchmark_results.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == '__main__':
    run_blind_crop_evaluation(
        video_path="backend/uploads/fe7f8619b7ea_test_17.mp4",
        gt_path=".agents/memory/golden_ground_truth_750.json",
        model_path="backend/weights/football/best.pt",
        start_f=600,
        end_f=750
    )
