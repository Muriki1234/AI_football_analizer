"""
benchmark_multiresolution_ball_player.py
========================================
Comprehensive Multi-Resolution Benchmark for Football & Player Detection:
Evaluates YOLO11n on real 1080p broadcast football video (backend/uploads/fe7f8619b7ea_test_17.mp4)
across input resolutions [640, 960, 1280, 1536, 1920] against dense consensus ground truth.

Measures:
- Ball Detection: TP, FP, FN, Precision, Recall, F1, Max Gap
- Player Detection: TP, FP, FN, Precision, Recall, F1
- Speed & Latency: Wall-clock time, ms/frame, FPS
- Memory & Scaling Behavior
"""

import cv2
import json
import time
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def center_dist(box1, box2):
    c1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    c2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    return np.hypot(c1[0] - c2[0], c1[1] - c2[1])

def run_benchmark(video_path: str, gt_path: str, model_path: str, start_f=600, end_f=750):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"[BENCHMARK] Device: {device}, Frames: {start_f}..{end_f-1} ({end_f-start_f} frames)")
    
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)

    gt_balls = gt_data['ball_tracks'][start_f:end_f]
    gt_players = gt_data['player_tracks'][start_f:end_f]

    # Pre-extract frames to ensure fair timing
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frames = []
    for _ in range(end_f - start_f):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"[BENCHMARK] Pre-loaded {len(frames)} frames into memory.")

    resolutions = [640, 960, 1280, 1536, 1920]
    results = {}

    model = YOLO(model_path)
    
    # Warmup
    _ = model(frames[0], imgsz=640, device=device, verbose=False)

    for res in resolutions:
        print(f"\n---> Evaluating imgsz = {res}...")
        t0 = time.perf_counter()
        
        # Batch or sequential inference
        inferences = []
        for frame in frames:
            inf = model(frame, imgsz=res, device=device, verbose=False, conf=0.25)
            inferences.append(inf[0])
            
        elapsed = time.perf_counter() - t0
        fps = len(frames) / elapsed
        ms_per_frame = (elapsed / len(frames)) * 1000.0

        # Evaluate Ball Detection (conf threshold 0.35)
        ball_tp = 0
        ball_fp = 0
        ball_fn = 0
        ball_gaps = []
        curr_gap = 0

        # Evaluate Player Detection (conf threshold 0.40)
        player_tp = 0
        player_fp = 0
        player_fn = 0

        for idx, (res_inf, gt_b_dict, gt_p_dict) in enumerate(zip(inferences, gt_balls, gt_players)):
            boxes = res_inf.boxes
            # Parse ball predictions
            ball_preds = []
            player_preds = []
            if boxes is not None and len(boxes) > 0:
                for b_box in boxes:
                    cls_id = int(b_box.cls.item())
                    score = float(b_box.conf.item())
                    xyxy = b_box.xyxy[0].cpu().numpy().tolist()
                    if cls_id == 1 and score >= 0.35:
                        ball_preds.append((xyxy, score))
                    elif cls_id == 0 and score >= 0.40:
                        player_preds.append((xyxy, score))

            # Ball Ground Truth
            has_gt_ball = False
            gt_ball_box = None
            if gt_b_dict and (1 in gt_b_dict or '1' in gt_b_dict):
                b_info = gt_b_dict.get(1) or gt_b_dict.get('1', {})
                b_coords = b_info.get('bbox')
                if b_coords and len(b_coords) == 4:
                    has_gt_ball = True
                    gt_ball_box = b_coords

            matched_ball = False
            if has_gt_ball:
                for p_box, _ in ball_preds:
                    if center_dist(p_box, gt_ball_box) < 35.0 or compute_iou(p_box, gt_ball_box) > 0.15:
                        matched_ball = True
                        break
                if matched_ball:
                    ball_tp += 1
                    if curr_gap > 0:
                        ball_gaps.append(curr_gap)
                        curr_gap = 0
                else:
                    ball_fn += 1
                    curr_gap += 1
            else:
                curr_gap += 1

            if len(ball_preds) > 0 and not matched_ball and not has_gt_ball:
                ball_fp += len(ball_preds)
            elif len(ball_preds) > 1 and matched_ball:
                ball_fp += (len(ball_preds) - 1)

            # Player Evaluation
            gt_player_boxes = [pinfo['bbox'] for pinfo in gt_p_dict.values() if 'bbox' in pinfo and len(pinfo['bbox']) == 4]
            matched_gt = set()
            for p_pred_box, _ in player_preds:
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt_box in enumerate(gt_player_boxes):
                    if gt_idx in matched_gt:
                        continue
                    iou = compute_iou(p_pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                if best_iou >= 0.45:
                    player_tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    player_fp += 1
            player_fn += (len(gt_player_boxes) - len(matched_gt))

        if curr_gap > 0:
            ball_gaps.append(curr_gap)

        b_prec = ball_tp / (ball_tp + ball_fp) if (ball_tp + ball_fp) > 0 else 0.0
        b_rec = ball_tp / (ball_tp + ball_fn) if (ball_tp + ball_fn) > 0 else 0.0
        b_f1 = (2 * b_prec * b_rec) / (b_prec + b_rec) if (b_prec + b_rec) > 0 else 0.0
        max_ball_gap = max(ball_gaps) if ball_gaps else 0

        p_prec = player_tp / (player_tp + player_fp) if (player_tp + player_fp) > 0 else 0.0
        p_rec = player_tp / (player_tp + player_fn) if (player_tp + player_fn) > 0 else 0.0
        p_f1 = (2 * p_prec * p_rec) / (p_prec + p_rec) if (p_prec + p_rec) > 0 else 0.0

        results[res] = {
            "imgsz": res,
            "fps": round(fps, 1),
            "ms_per_frame": round(ms_per_frame, 1),
            "ball": {
                "tp": ball_tp, "fp": ball_fp, "fn": ball_fn,
                "precision": round(b_prec, 4),
                "recall": round(b_rec, 4),
                "f1": round(b_f1, 4),
                "max_gap": max_ball_gap
            },
            "player": {
                "tp": player_tp, "fp": player_fp, "fn": player_fn,
                "precision": round(p_prec, 4),
                "recall": round(p_rec, 4),
                "f1": round(p_f1, 4),
            }
        }
        print(f"  Ball   -> Prec: {b_prec:.3f}, Rec: {b_rec:.3f}, F1: {b_f1:.3f}, MaxGap: {max_ball_gap}")
        print(f"  Player -> Prec: {p_prec:.3f}, Rec: {p_rec:.3f}, F1: {p_f1:.3f}")
        print(f"  Speed  -> {ms_per_frame:.1f} ms/frame ({fps:.1f} FPS)")

    out_file = Path(".agents/memory/multiresolution_benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[BENCHMARK] Saved detailed multi-resolution benchmark to {out_file}")

if __name__ == '__main__':
    run_benchmark(
        video_path="backend/uploads/fe7f8619b7ea_test_17.mp4",
        gt_path=".agents/memory/golden_ground_truth_750.json",
        model_path="backend/weights/football/best.pt",
        start_f=600,
        end_f=750
    )
