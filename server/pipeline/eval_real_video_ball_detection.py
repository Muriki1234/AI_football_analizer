"""
eval_real_video_ball_detection.py — Real Video Ball Detection & Precision/Recall Evaluation

Ground-truth evaluation against AI_Football_Broadcast_Golden_Evaluation_Set:
Video: backend/uploads/fe7f8619b7ea_test_17.mp4 (1080p, 25fps)
Annotations: .agents/memory/golden_ground_truth_750.json

Evaluates:
- Baseline (Monolithic conf = 0.59 from analysis_core.py)
- Candidate A (Decoupled conf = 0.22 from decoupled_detection_tracker.py)
- Candidate B (Decoupled conf = 0.35)

Metrics computed per configuration:
- TP, FP, FN
- Precision, Recall, F1
- Max consecutive dropout gap (frames)
- Downstream player-ball association continuity
"""

import json
import math
import os
import time
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO


def calc_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = boxAArea + boxBArea - interArea
    return interArea / denom if denom > 0 else 0.0


def center_dist(boxA, boxB):
    cA = ((boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0)
    cB = ((boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0)
    return math.hypot(cA[0] - cB[0], cA[1] - cB[1])


def run_ball_eval(
    video_path: str,
    gt_path: str,
    weights_path: str,
    start_frame: int = 600,
    num_frames: int = 150,
    device: str = "cpu"
) -> Dict[str, Any]:
    print(f"Loading Ground Truth from {gt_path}...")
    with open(gt_path, "r") as f:
        gt_data = json.load(f)
    gt_ball_tracks = gt_data["ball_tracks"]

    end_frame = min(len(gt_ball_tracks), start_frame + num_frames)
    target_frames = list(range(start_frame, end_frame))
    annotated_frames = [f for f in target_frames if gt_ball_tracks[f]]
    print(f"Evaluation window: frames [{start_frame}..{end_frame-1}] ({len(target_frames)} frames, {len(annotated_frames)} with GT ball)")

    print(f"Loading video {video_path}...")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in target_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames into memory.")

    print(f"Loading YOLO model from {weights_path}...")
    model = YOLO(weights_path)
    ball_class_id = 1
    for cid, name in model.names.items():
        if "ball" in name.lower():
            ball_class_id = cid
            break

    # Warmup
    _ = model.predict(frames[:2], conf=0.20, verbose=False, device=device)

    # We evaluate across multiple confidence thresholds
    test_thresholds = [0.59, 0.35, 0.22, 0.15]
    results = {}

    for conf_thresh in test_thresholds:
        t0 = time.perf_counter()
        preds = []
        batch_size = 16
        for b_start in range(0, len(frames), batch_size):
            b_chunk = frames[b_start:b_start + batch_size]
            b_preds = model.predict(b_chunk, conf=conf_thresh, verbose=False, device=device)
            preds.extend(b_preds)
        inference_time = time.perf_counter() - t0

        tp = 0
        fp = 0
        fn = 0
        detected_per_frame = []

        for offset, fidx in enumerate(target_frames):
            pred_boxes = []
            if offset < len(preds):
                r = preds[offset]
                for box in r.boxes:
                    if int(box.cls[0]) == ball_class_id:
                        pred_boxes.append((box.xyxy[0].tolist(), float(box.conf[0])))

            # Best ball candidate in frame
            best_pred = max(pred_boxes, key=lambda x: x[1]) if pred_boxes else None
            detected_per_frame.append(best_pred is not None)

            gt_entry = gt_ball_tracks[fidx]
            has_gt = bool(gt_entry)

            if has_gt:
                gt_bbox = list(gt_entry.values())[0]["bbox"]
                if best_pred is not None:
                    # Match criterion: center dist < 35px or IoU > 0.20 (ball is ~15px)
                    dist = center_dist(best_pred[0], gt_bbox)
                    iou = calc_iou(best_pred[0], gt_bbox)
                    if dist <= 35.0 or iou >= 0.20:
                        tp += 1
                    else:
                        fp += 1  # False alarm on wrong object
                        fn += 1  # Missed the true ball
                else:
                    fn += 1  # Missed ball
            else:
                if best_pred is not None:
                    fp += 1  # False positive when ball was out of view/occluded

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Max consecutive dropout gap
        max_gap = 0
        cur_gap = 0
        for d in detected_per_frame:
            if not d:
                cur_gap += 1
                if cur_gap > max_gap:
                    max_gap = cur_gap
            else:
                cur_gap = 0

        results[str(conf_thresh)] = {
            "conf_threshold": conf_thresh,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "max_consecutive_dropout_frames": max_gap,
            "inference_fps": round(len(frames) / max(1e-6, inference_time), 1),
        }

    return results


if __name__ == "__main__":
    res = run_ball_eval(
        video_path="backend/uploads/fe7f8619b7ea_test_17.mp4",
        gt_path=".agents/memory/golden_ground_truth_750.json",
        weights_path="backend/weights/football/best.pt",
        start_frame=600,
        num_frames=150,
        device="cpu",
    )
    print("\n" + "=" * 65)
    print(" REAL VIDEO BALL DETECTION BENCHMARK RESULTS (Frames 600-750)")
    print("=" * 65)
    print(f"{'Conf Thresh':<12} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Precision':<10} | {'Recall':<8} | {'F1':<8} | {'MaxGap':<6} | {'FPS':<6}")
    print("-" * 65)
    for k, v in res.items():
        print(f"{v['conf_threshold']:<12.2f} | {v['tp']:<4} | {v['fp']:<4} | {v['fn']:<4} | {v['precision']:<10.3f} | {v['recall']:<8.3f} | {v['f1']:<8.3f} | {v['max_consecutive_dropout_frames']:<6} | {v['inference_fps']:<6.1f}")
    print("=" * 65)
