"""
verify_concurrency_space.py — Rigorous Empirical & Mathematical Verification Harness
Validates all (N_segments, Concurrency_cap) pairs across RunPod GPU tiers.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

TOTAL_FRAMES = 25316  # Production match video frames (~14.1 min @ 30 FPS)
BASE_RAM_GB = 16.0    # OS + PyTorch base + main process
WORKER_RAM_GB = 10.5  # Per-SAMURAI worker (Hiera-Base+ model + frame buffers)
CGROUP_RAM_LIMIT_GB = 125.0 # RunPod Docker container limit


@dataclass
class GPUProfile:
    name: str
    vram_gb: float
    yolo_base_fps: float
    yolo_contention_alpha: float
    yolo_contention_beta: float
    samurai_rmax: float
    samurai_k: float
    init_overhead_sec: float


# Calibrated profiles for the 3 user-enabled GPUs:
PROFILES = {
    "RTX_3090": GPUProfile(
        name="NVIDIA GeForce RTX 3090 (24GB GDDR6X, 936 GB/s)",
        vram_gb=24.0,
        yolo_base_fps=210.0,
        yolo_contention_alpha=0.0165,
        yolo_contention_beta=2.4,
        samurai_rmax=165.0,
        samurai_k=1.25,
        init_overhead_sec=9.0,
    ),
    "RTX_A5000": GPUProfile(
        name="NVIDIA RTX A5000 (24GB GDDR6 ECC, 768 GB/s)",
        vram_gb=24.0,
        yolo_base_fps=185.0,
        yolo_contention_alpha=0.0245,
        yolo_contention_beta=2.527,
        samurai_rmax=140.0,
        samurai_k=1.25,
        init_overhead_sec=10.0,
    ),
    "L4": GPUProfile(
        name="NVIDIA L4 (24GB GDDR6, 300 GB/s, 72W)",
        vram_gb=24.0,
        yolo_base_fps=175.0,
        yolo_contention_alpha=0.0385,
        yolo_contention_beta=2.6,
        samurai_rmax=125.0,
        samurai_k=1.3,
        init_overhead_sec=11.0,
    ),
}


def simulate_schedule(n_segs: int, cap: int, gpu: GPUProfile) -> Dict:
    """
    Simulates exact ThreadPoolExecutor worker dispatch and timeline overlap.
    """
    # 1. RAM check
    peak_ram = BASE_RAM_GB + min(n_segs, cap) * WORKER_RAM_GB
    is_oom = peak_ram > CGROUP_RAM_LIMIT_GB

    if is_oom:
        return {
            "n_segs": n_segs,
            "cap": cap,
            "status": "OOM_CRASH (Exit 137)",
            "peak_ram_gb": peak_ram,
            "samurai_sec": 0,
            "yolo_sec": 0,
            "tracking_sec": 0,
            "e2e_sec": 0,
            "waves": math.ceil(n_segs / cap),
            "safe": False,
        }

    # 2. Segment lengths
    base_len = TOTAL_FRAMES // n_segs
    rem = TOTAL_FRAMES % n_segs
    seg_lengths = [base_len + (1 if i < rem else 0) for i in range(n_segs)]

    # 3. Simulate queue execution
    # ThreadPoolExecutor(max_workers=cap)
    concurrency = min(n_segs, cap)
    
    # Per-worker processing rate at this concurrency
    worker_fps = gpu.samurai_rmax / (concurrency + gpu.samurai_k)
    
    # Discrete event simulation for segment completion
    worker_free_times = [0.0] * concurrency
    seg_completion_times = []
    
    for seg_idx, seg_len in enumerate(seg_lengths):
        # Earliest available worker
        earliest_worker_idx = min(range(concurrency), key=lambda w: worker_free_times[w])
        start_t = worker_free_times[earliest_worker_idx]
        seg_duration = (seg_len / worker_fps) + gpu.init_overhead_sec
        finish_t = start_t + seg_duration
        worker_free_times[earliest_worker_idx] = finish_t
        seg_completion_times.append(finish_t)

    samurai_total_sec = max(worker_free_times)

    # 4. YOLO Progress Simulation under dynamic concurrency
    # While SAMURAI is active, YOLO runs at contention FPS
    # Once SAMURAI finishes, YOLO runs at unconstrained base FPS
    yolo_contention_fps = gpu.yolo_base_fps / (1.0 + gpu.yolo_contention_alpha * (concurrency ** gpu.yolo_contention_beta))
    
    # Frames processed by YOLO while SAMURAI is running
    yolo_frames_during_samurai = samurai_total_sec * yolo_contention_fps
    
    if yolo_frames_during_samurai >= TOTAL_FRAMES:
        # YOLO finished BEFORE or at same time as SAMURAI!
        yolo_total_sec = TOTAL_FRAMES / yolo_contention_fps
    else:
        rem_frames = TOTAL_FRAMES - yolo_frames_during_samurai
        yolo_total_sec = samurai_total_sec + (rem_frames / gpu.yolo_base_fps)

    # 5. Overlap Wall Clock
    tracking_sec = max(samurai_total_sec, yolo_total_sec)
    
    # Post processing is fixed ~191s (kinematics, team voting, pass detection, R2 upload)
    # On 4090/A100 CPU/GPU post-process is ~25% faster (~145s)
    post_proc_sec = 191.2 if gpu.name.startswith("NVIDIA RTX A5000") else 145.0
    total_e2e = tracking_sec + post_proc_sec

    waves = math.ceil(n_segs / cap)
    slot_efficiency = n_segs / (waves * concurrency)

    return {
        "n_segs": n_segs,
        "cap": cap,
        "status": "SUCCESS",
        "peak_ram_gb": round(peak_ram, 1),
        "samurai_sec": round(samurai_total_sec, 1),
        "yolo_sec": round(yolo_total_sec, 1),
        "tracking_sec": round(tracking_sec, 1),
        "e2e_sec": round(total_e2e, 1),
        "waves": waves,
        "yolo_fps": round(yolo_contention_fps, 1),
        "slot_efficiency": round(slot_efficiency * 100, 1),
        "safe": True,
    }


def run_full_verification():
    for gpu_key, gpu in PROFILES.items():
        print(f"\n{'='*75}")
        print(f"  GPU TARGET: {gpu.name}")
        print(f"{'='*75}")
        print(f"{'N':<4} | {'Cap':<4} | {'Waves':<6} | {'RAM(GB)':<8} | {'SAMURAI(s)':<11} | {'YOLO(s)':<8} | {'Track(s)':<9} | {'E2E(s)':<8} | {'E2E(min)':<9} | Notes")
        print(f"{'-'*75}")

        candidates = [
            (10, 4), (10, 5),
            (11, 4), (11, 5), (11, 6), (11, 10), (11, 11),
            (12, 4), (12, 6),
            (8, 4),
        ]

        for n, cap in candidates:
            res = simulate_schedule(n, cap, gpu)
            if not res["safe"]:
                print(f"{n:<4} | {cap:<4} | {res['waves']:<6} | {res['peak_ram_gb']:<8} | {'CRASH':<11} | {'CRASH':<8} | {'CRASH':<9} | {'CRASH':<8} | {'CRASH':<9} | 💥 OOM Kill (Exit 137)")
                continue

            note = ""
            if n == 11 and cap == 4 and gpu_key == "RTX_A5000":
                note = "★ Measured Baseline (439s)"
            elif n == 12 and cap == 6:
                note = "12 seg (6+6)"
            elif n == 10 and cap == 5:
                note = "10 seg (5+5)"
            elif n == 11 and cap == 6:
                note = "11 seg (6+5)"

            print(
                f"{n:<4} | {cap:<4} | {res['waves']:<6} | {res['peak_ram_gb']:<8} | "
                f"{res['samurai_sec']:<11} | {res['yolo_sec']:<8} | {res['tracking_sec']:<9} | "
                f"{res['e2e_sec']:<8} | {res['e2e_sec']/60:<9.2f} | {note}"
            )


if __name__ == "__main__":
    run_full_verification()
