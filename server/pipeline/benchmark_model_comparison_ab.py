"""
benchmark_model_comparison_ab.py
================================
P0-2: Empirical Model Comparison A/B Benchmark

Evaluates:
1. Current Model: backend/weights/football/best.pt (YOLO11n-based, 2.59M params, 3-class)
2. Candidate A: .agents/models/v9b_best.pt (acatorcini/yolov9-soccer-ball, 11.14M params, ball-specialist)
3. Candidate B: yolo11s.pt (Ultralytics YOLO11s, 9.46M params, COCO sports ball class 32)

Evaluated under strict empirical conditions:
- Identical 150 frames (frames 600..749 of fe7f8619b7ea_test_17.mp4)
- Identical Golden Ground Truth (.agents/memory/golden_ground_truth_750.json)
- Identical Spatial Distance Threshold (<35px center distance for TP)
- Warmup before timing
- Tested across full-frame 640 and native 1280
"""

import json
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def center_dist(box1, box2):
    c1 = ((box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0)
    c2 = ((box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0)
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def evaluate_detector(model, model_name, ball_class_id, conf_thresh, frames, gt_balls, imgsz=1280, device='mps'):
    # Warmup
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model(dummy, imgsz=imgsz, device=device, verbose=False)

    tp, fp, fn = 0, 0, 0
    gaps = []
    curr_gap = 0
    latencies = []

    for fi, frame in enumerate(frames):
        has_gt = bool(gt_balls[fi] and (1 in gt_balls[fi] or '1' in gt_balls[fi]))
        gt_box = (gt_balls[fi].get(1) or gt_balls[fi].get('1', {})).get('bbox') if has_gt else None

        t0 = time.perf_counter()
        res = model(frame, imgsz=imgsz, device=device, verbose=False, conf=conf_thresh)[0]
        dt = time.perf_counter() - t0
        latencies.append(dt)

        detected_boxes = []
        for b in res.boxes:
            cls_id = int(b.cls.item())
            conf_val = float(b.conf.item())
            if cls_id == ball_class_id:
                box_xyxy = b.xyxy[0].cpu().numpy().tolist()
                detected_boxes.append((box_xyxy, conf_val))

        best_det = None
        if detected_boxes:
            best_det = max(detected_boxes, key=lambda x: x[1])[0]

        if has_gt and best_det is not None:
            if center_dist(best_det, gt_box) < 35.0:
                tp += 1
                if curr_gap > 0:
                    gaps.append(curr_gap)
                    curr_gap = 0
            else:
                fp += 1
                fn += 1
                curr_gap += 1
        elif has_gt and best_det is None:
            fn += 1
            curr_gap += 1
        elif not has_gt and best_det is not None:
            fp += 1

    if curr_gap > 0:
        gaps.append(curr_gap)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_latency_ms = float(np.mean(latencies)) * 1000.0
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    return {
        "model_name": model_name,
        "imgsz": imgsz,
        "conf_threshold": conf_thresh,
        "tp": tp, "fp": fp, "fn": fn,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "max_gap": max(gaps) if gaps else 0,
        "avg_latency_ms": round(avg_latency_ms, 2),
        "fps": round(fps, 1),
    }


def run_benchmark():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"[BENCHMARK] Using device: {device}")

    # Load video frames
    cap = cv2.VideoCapture("backend/uploads/fe7f8619b7ea_test_17.mp4")
    frames = []
    f_idx = 0
    while cap.isOpened() and f_idx < 750:
        ret, frame = cap.read()
        if not ret:
            break
        if 600 <= f_idx < 750:
            frames.append(frame)
        f_idx += 1
    cap.release()
    print(f"[BENCHMARK] Loaded {len(frames)} evaluation frames (600..749).")

    # Load GT
    with open(".agents/memory/golden_ground_truth_750.json") as f:
        gt_data = json.load(f)
    gt_balls = gt_data['ball_tracks'][600:750]

    # Model 1: Current YOLO11n football detector (class 1 is Ball)
    print("\n---> Evaluating Model 1: Current YOLO11n (football/best.pt)...")
    m_curr = YOLO("backend/weights/football/best.pt")
    res_curr_640 = evaluate_detector(m_curr, "Current YOLO11n @ 640px", 1, 0.25, frames, gt_balls, imgsz=640, device=device)
    res_curr_1280 = evaluate_detector(m_curr, "Current YOLO11n @ 1280px", 1, 0.25, frames, gt_balls, imgsz=1280, device=device)

    # Model 2: Candidate A: YOLOv9-soccer-ball (class 0 is Ball, recommended conf 0.20)
    print("\n---> Evaluating Model 2: Candidate A (acatorcini/yolov9-soccer-ball)...")
    m_v9b = YOLO(".agents/models/v9b_best.pt")
    res_v9b_640 = evaluate_detector(m_v9b, "Candidate A (YOLOv9-ball) @ 640px", 0, 0.20, frames, gt_balls, imgsz=640, device=device)
    res_v9b_1280 = evaluate_detector(m_v9b, "Candidate A (YOLOv9-ball) @ 1280px", 0, 0.20, frames, gt_balls, imgsz=1280, device=device)

    # Model 3: Candidate B: YOLO11s official (class 32 is 'sports ball')
    print("\n---> Evaluating Model 3: Candidate B (Ultralytics YOLO11s COCO)...")
    m_11s = YOLO("yolo11s.pt")
    res_11s_640 = evaluate_detector(m_11s, "Candidate B (YOLO11s-COCO) @ 640px", 32, 0.25, frames, gt_balls, imgsz=640, device=device)
    res_11s_1280 = evaluate_detector(m_11s, "Candidate B (YOLO11s-COCO) @ 1280px", 32, 0.25, frames, gt_balls, imgsz=1280, device=device)

    all_results = [res_curr_640, res_curr_1280, res_v9b_640, res_v9b_1280, res_11s_640, res_11s_1280]

    print("\n" + "=" * 95)
    print("EMPIRICAL MODEL A/B COMPARISON BENCHMARK (FRAMES 600..749)")
    print("=" * 95)
    header = f"{'Model & Config':<36} | {'Recall':<8} | {'Precision':<9} | {'F1':<6} | {'MaxGap':<6} | {'Latency':<9} | {'FPS':<6}"
    print(header)
    print("-" * 95)
    for r in all_results:
        row = f"{r['model_name']:<36} | {r['recall']*100:>6.1f}% | {r['precision']*100:>8.1f}% | {r['f1']:>6.3f} | {r['max_gap']:>6} | {r['avg_latency_ms']:>6.1f}ms | {r['fps']:>6.1f}"
        print(row)
    print("=" * 95)

    with open(".agents/memory/model_comparison_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    run_benchmark()
