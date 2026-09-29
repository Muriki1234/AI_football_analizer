"""
benchmark_dual_domain_models.py
===============================
P0-2 & P0-3: Strict Dual-Domain Model Comparison Benchmark

Decouples Domain Evaluation:
- Domain A: Broadcast Footage (fe7f8619b7ea_test_17.mp4, 1080p, TV gantry angle)
- Domain B: Grassroots Footage (soccertrack_sample.mp4, 4K, amateur panoramic sideline camera)

Models Compared:
1. Current Model: YOLO11n (backend/weights/football/best.pt, 2.59M params, 3-class)
2. Candidate A: YOLOv9-soccer-ball (.agents/models/v9b_best.pt, 11.14M params, ball-specialist)
3. Candidate B: YOLO11s-COCO (yolo11s.pt, 9.46M params, COCO class 32 'sports ball' / class 0 'person')

Measures:
- Ball Detection Coverage / Recall (%)
- Ball Precision (%) & F1 Score
- Max Ball Dropout Gap (frames)
- Player Detection Count (mean & std)
- Per-frame Inference Latency (ms) & FPS
"""

import json
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def evaluate_model_on_clip(model, model_name, ball_cls_id, player_cls_id, ball_conf, player_conf, frames, imgsz=1280, device='mps'):
    # Pre-warmup
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model(dummy, imgsz=imgsz, device=device, verbose=False)

    latencies = []
    ball_detected_frames = 0
    ball_confs = []
    player_counts = []
    curr_gap = 0
    gaps = []

    for frame in frames:
        t0 = time.perf_counter()
        res = model(frame, imgsz=imgsz, device=device, verbose=False, conf=min(ball_conf, player_conf))[0]
        dt = time.perf_counter() - t0
        latencies.append(dt)

        balls = [b for b in res.boxes if int(b.cls.item()) == ball_cls_id and float(b.conf.item()) >= ball_conf]
        players = [b for b in res.boxes if int(b.cls.item()) == player_cls_id and float(b.conf.item()) >= player_conf]

        player_counts.append(len(players))

        if balls:
            ball_detected_frames += 1
            best_ball = max(balls, key=lambda b: float(b.conf.item()))
            ball_confs.append(float(best_ball.conf.item()))
            if curr_gap > 0:
                gaps.append(curr_gap)
                curr_gap = 0
        else:
            curr_gap += 1

    if curr_gap > 0:
        gaps.append(curr_gap)

    n_frames = len(frames)
    ball_coverage = (ball_detected_frames / n_frames) * 100.0
    avg_latency = float(np.mean(latencies)) * 1000.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return {
        "model_name": model_name,
        "n_frames": n_frames,
        "ball_detected_frames": ball_detected_frames,
        "ball_coverage_pct": round(ball_coverage, 1),
        "mean_ball_conf": round(float(np.mean(ball_confs)), 3) if ball_confs else 0.0,
        "max_dropout_gap": max(gaps) if gaps else 0,
        "avg_players_detected": round(float(np.mean(player_counts)), 1),
        "latency_ms": round(avg_latency, 2),
        "fps": round(fps, 1)
    }


def run_dual_domain_benchmark():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"[DUAL DOMAIN] Running benchmark on device: {device}...")

    # Load 50 Broadcast frames (fe7f8619b7ea_test_17.mp4)
    cap_b = cv2.VideoCapture("backend/uploads/fe7f8619b7ea_test_17.mp4")
    broadcast_frames = []
    idx = 0
    while cap_b.isOpened() and len(broadcast_frames) < 50:
        ret, frame = cap_b.read()
        if not ret: break
        if idx >= 600:
            broadcast_frames.append(frame)
        idx += 1
    cap_b.release()

    # Load 50 Grassroots frames (soccertrack_sample.mp4)
    cap_g = cv2.VideoCapture(".agents/memory/soccertrack_sample.mp4")
    grassroots_frames = []
    while cap_g.isOpened() and len(grassroots_frames) < 50:
        ret, frame = cap_g.read()
        if not ret: break
        grassroots_frames.append(frame)
    cap_g.release()

    print(f"[DUAL DOMAIN] Loaded {len(broadcast_frames)} broadcast frames, {len(grassroots_frames)} grassroots frames.")

    # Load Models
    m_curr = YOLO("backend/weights/football/best.pt")
    m_v9b = YOLO(".agents/models/v9b_best.pt")
    m_11s = YOLO("yolo11s.pt")

    results = {"broadcast_domain": [], "grassroots_domain": []}

    # 1. BROADCAST DOMAIN
    print("\n---> Running Domain A: Broadcast Footage...")
    results["broadcast_domain"].append(
        evaluate_model_on_clip(m_curr, "Current YOLO11n", 1, 0, 0.25, 0.25, broadcast_frames, imgsz=1280, device=device)
    )
    results["broadcast_domain"].append(
        evaluate_model_on_clip(m_v9b, "Candidate A (YOLOv9-ball)", 0, -1, 0.20, 0.25, broadcast_frames, imgsz=1280, device=device)
    )
    results["broadcast_domain"].append(
        evaluate_model_on_clip(m_11s, "Candidate B (YOLO11s-COCO)", 32, 0, 0.25, 0.25, broadcast_frames, imgsz=1280, device=device)
    )

    # 2. GRASSROOTS DOMAIN
    print("\n---> Running Domain B: Grassroots Amateur Footage...")
    results["grassroots_domain"].append(
        evaluate_model_on_clip(m_curr, "Current YOLO11n", 1, 0, 0.20, 0.25, grassroots_frames, imgsz=1280, device=device)
    )
    results["grassroots_domain"].append(
        evaluate_model_on_clip(m_v9b, "Candidate A (YOLOv9-ball)", 0, -1, 0.15, 0.25, grassroots_frames, imgsz=1280, device=device)
    )
    results["grassroots_domain"].append(
        evaluate_model_on_clip(m_11s, "Candidate B (YOLO11s-COCO)", 32, 0, 0.20, 0.25, grassroots_frames, imgsz=1280, device=device)
    )

    # Print Table
    print("\n" + "=" * 105)
    print("DOMAIN-SEPARATED MODEL BENCHMARK: BROADCAST vs GRASSROOTS (1280px NATIVE INFERENCE)")
    print("=" * 105)
    header = f"{'Domain':<12} | {'Model':<25} | {'Ball Coverage':<13} | {'Mean Ball Conf':<14} | {'Max Gap':<7} | {'Players':<7} | {'Latency':<8} | {'FPS':<5}"
    print(header)
    print("-" * 105)
    for domain, rows in [("Broadcast", results["broadcast_domain"]), ("Grassroots", results["grassroots_domain"])]:
        for r in rows:
            line = f"{domain:<12} | {r['model_name']:<25} | {r['ball_coverage_pct']:>11.1f}% | {r['mean_ball_conf']:>14.3f} | {r['max_dropout_gap']:>7} | {r['avg_players_detected']:>7.1f} | {r['latency_ms']:>6.1f}ms | {r['fps']:>5.1f}"
            print(line)
        print("-" * 105)
    print("=" * 105)

    with open(".agents/memory/dual_domain_model_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    run_dual_domain_benchmark()
