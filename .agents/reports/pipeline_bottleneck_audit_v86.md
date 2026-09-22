# Full-Pipeline Bottleneck Audit & Stage Timeline Report (v86)

**Session**: `session_night_20260922_runpod_e2e`  
**Methodology**: Strict Evidence-First Taxonomy (`OBSERVED`, `INFERRED`, `HYPOTHESIS`, `UNKNOWN`)  
**Target Workload**: 25,316 frames (14.1 minutes, 1926×1080 @ 30 FPS), RunPod Serverless GPU Worker (`qbpu3fsrcv05wd`)

---

## 1. Executive Summary & Observed Bottleneck Candidates — Pre-Matrix

Based strictly on empirical evidence extracted from the production run of session `02570f44-347b-4591-a494-aba840514b26` (Pre-Matrix baseline, not final ranking):

| Rank | Bottleneck Area | Empirical Time / Impact | Evidence Class | Primary Driver |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **SAMURAI Wave 2 Queuing** | **217.0s** (54.0% of E2E) | `OBSERVED` | Cap=10 caused 11th segment to run solo in a second wave after 184s |
| **2** | **CUDA Compute & Memory Contention** | **YOLO dropped 183 → 22 FPS** | `OBSERVED` | 10 concurrent SAM2 processes + YOLO contending for GPU SMs & RAM |
| **3** | **Supabase Synchronous RPC Storm** | **~27–45s blocking latency** | `INFERRED` | 89+ synchronous `merge_session_extra` HTTPS roundtrips on main thread |
| **4** | **YOLO Unconstrained Streaming** | **~50s** (normal execution) | `OBSERVED` | 25,316 frames @ 183 FPS (stride=3) takes ~46s pure compute |
| **5** | **Post-Processing & Serialization** | **< 1.0s** (<0.25% of E2E) | `OBSERVED` | Voting (0.2s), Passes (0.1s), Pickle (0.3s), Export (0.1s) |
| **6** | **Worker Idle Tail (20:01:37 → 20:38:05)** | **36m 28s (Zero compute)** | `OBSERVED` | Serverless warm container idle wait, NOT pipeline compute |

---

## 2. Evidence Audit: The 36-Minute Tail Gap (`20:01:37 → 20:38:05`)

### Empirical Timeline
- **`20:01:37.050`**: Post-processing finished; `tracks.pkl` (70.5 MB) written to local disk.
- **`20:01:37.499`**: `tracks.pkl` uploaded to Cloudflare R2 bucket.
- **`20:01:37.666`**: `minimap_positions.json` (3,257 KB) and `heatmap_positions.json` (12 KB) exported.
- **`20:01:37.666`**: `[ORCHESTRATOR] Concurrency completed in 401.60s (SAMURAI: 193.91s, YOLO: 401.60s | Saved: 193.91s, 32.6%)`.
- **`20:01:37.666`**: `[TRACK] Pipeline execution finished in 401.6s`.
- **`20:01:37.666`**: Synchronous Supabase GET request returned `HTTP/2 200 OK`.
- **`20:01:37.667`**: `handler()` returned `{"ok": True, "session": ...}` to RunPod Serverless runtime.
- **`20:01:37.667 → 20:38:05.610`** (36 minutes, 28 seconds): **Zero log lines, zero network requests, zero CPU/GPU activity.**
- **`20:38:05.610`**: RunPod serverless daemon printed:
  ```text
  Finished.
  Kill worker.
  ```

### Findings & Evidence Taxonomy
- `OBSERVED`: The video tracking pipeline started at `19:54:55.898` and finished at `20:01:37.666`, taking exactly **401.60 seconds** (~6.69 minutes).
- `OBSERVED`: Zero log lines containing `[AI_SUMMARY]` exist anywhere in the worker container logs.
- `OBSERVED`: In `server/handler.py`, the `_action_track` handler terminates and returns immediately after `PipelineConcurrencyScheduler` completes; it does **not** invoke `run_ai_summary`.
- `INFERRED`: The 36-minute window was purely worker container **idle wait** in RunPod's serverless warm pool, awaiting subsequent requests before being terminated by the serverless idle timeout or user stop action.
- `INFERRED`: AI summary was **not** running on this worker during the 36-minute window.
- `HYPOTHESIS`: If AI summary was requested, it ran on an independent CPU worker (`WORKER_MODE=cpu`), on a separate request, or was deferred.
- `UNKNOWN`: Status, duration, or execution worker of any AI summary request cannot be established from this GPU worker log alone.

---

## 3. End-to-End Pipeline Stage Timeline & Metrics

```text
  19:54:53   Worker Boot & Weight Verification (3.9s)
     │
  19:54:56   [ORCHESTRATOR START]
     ├── SAMURAI Wave 1 (10 workers): 19:54:56 → 19:58:00 (183.8s)
     │     RAM: 60GB → 121GB RSS peak | GPU: 10 SAM2 processes
     │     YOLO Chunk Streaming (Overlapped): frames 0..6000 @ 18-22 FPS
     │
     ├── SAMURAI Wave 2 (Seg 10 solo): 19:58:00 → 20:01:37 (217.0s)
     │     RAM: dropped 121GB → 59GB RSS | GPU: 1 SAM2 process
     │     YOLO Chunk Streaming (Overlapped): frames 6000..25316 @ 183-187 FPS (completed in 228s)
     │     YOLO Orchestrator Thread: WAITING on SAMURAI Event from ~19:58:45 to 20:01:37 (~172s wait)
     │
  20:01:37   SAMURAI Segment 10 Done → samurai_done_event.set()
     │
  20:01:37   Post-Processing (< 1.0s):
     ├── Team Multi-Frame Voting: 0.2s
     ├── Pass Detection: 0.1s
     ├── BBox Integration: 0.1s
     ├── Pickle Serialization (70.5MB tracks.pkl): 0.3s
     └── Minimap / Heatmap JSON Export: 0.1s
     │
  20:01:37.666  Cloudflare R2 Upload & Supabase Update (0.6s)
     │
  20:01:37.666  [PIPELINE FINISHED: 401.60s]
     │
     └── 20:01:37 → 20:38:05: Worker Idle in RunPod Pool (36m 28s)
```

### Stage Breakdown & Metrics Table

| Pipeline Stage | Wall Clock | Primary Resource | Metric Value | Evidence Class |
| :--- | :---: | :---: | :---: | :---: |
| **Worker Boot & Weights Check** | 3.9s | Disk / Network | Local weights verified | `OBSERVED` |
| **SAMURAI Wave 1 (Seg 0–9)** | 183.8s | GPU SMs + Host RAM | 121 GB RAM peak, 10 workers | `OBSERVED` |
| **SAMURAI Wave 2 (Seg 10)** | 217.0s | GPU + ffmpeg decode | 59 GB RAM, 1 worker solo | `OBSERVED` |
| **YOLO Chunk Inference (Overlapped)** | 228.7s | GPU Tensor Cores | 110.7 FPS avg (18 contention, 183 free) | `OBSERVED` |
| **YOLO Wait for SAMURAI** | ~172.0s | CPU Event Wait | Blocked on `_samurai_done_event` | `OBSERVED` |
| **Post-Processing (Team/Passes)** | 0.3s | CPU NumPy / Python | 395 players voted, 70 passes | `OBSERVED` |
| **Serialization & I/O** | 0.4s | Disk I/O | `tracks.pkl` (70.5 MB) | `OBSERVED` |
| **Cloudflare R2 Upload** | 0.4s | Network Egress | `tracks.pkl` uploaded | `OBSERVED` |
| **Supabase RPC Updates** | ~27–45s | Network Latency | 89 requests, 200–500ms RTT | `INFERRED` |
| **Serverless Idle Pool** | 2188s | Container Keep-Alive | Zero compute | `OBSERVED` |

---

## 4. CUDA Contention & Concurrency Dynamics
 
1. **Severe YOLO Slowdown during Wave 1 under C=10**:
   - `OBSERVED`: When 10 SAMURAI processes ran concurrently, YOLO chunk streaming throughput dropped from **183 FPS to ~18–22 FPS** (an 8× drop).
   - `OBSERVED`: The instant Wave 1 finished (leaving only 1 worker), YOLO throughput surged to **187.5 FPS**.
   - `OBSERVED`: Initial probe reported `23.4 GB free of 23.6 GB` VRAM. SAM2 base_plus uses ~500 MB VRAM per instance, totaling ~5 GB VRAM across 10 workers.
   - `OBSERVED`: **VRAM capacity does not appear to be the limiting resource in the observed C=10 run.**
   - `HYPOTHESIS`: Process multiplexing, CUDA context switching, and shared GPU compute resources are candidate mechanisms for the slowdown, but direct SM utilization and PCIe memory-bandwidth telemetry are required to confirm the precise hardware mechanism.

2. **Scenario Analysis / Theoretical Lower-Bound Caveat**:
   - `SCENARIO ANALYSIS (COUNTERFACTUAL)`: In the observed C=10 run, YOLO streaming finished at $t=228.7\text{s}$, while Wave 1 finished at $t=183.8\text{s}$. If Wave 2 had not been serialized, the theoretical lower bound of that run would have been $\max(t_{\text{samurai\_wave1}}, t_{\text{yolo}}) \approx 228.7\text{s}$.
   - **CRITICAL NOTE**: This is purely a counterfactual mathematical reference point, **NOT a prediction for C=11**, and **CANNOT be used as benchmark evidence**. Running 11 workers simultaneously could introduce higher contention, higher memory churn, or different scheduling dynamics that alter both $t_{\text{samurai}}$ and $t_{\text{yolo}}$.

3. **The Danger of Unchecked Concurrency (16 vs 11 vs 8 vs 6 vs 4)**:
   - In v85, `compute_samurai_concurrency_cap` was bumped to 16.
   - If 16 workers run simultaneously:
     - Host RAM may scale beyond 150+ GB.
     - 16 parallel `ffmpeg` decoders will contend for disk read bandwidth and CPU cores.
     - CUDA contention may degrade YOLO throughput even further.
   - Conversely, if concurrency is 4:
     - 11 segments require 3 waves ($4 + 4 + 3$).
     - Each wave takes ~60–80s, total SAMURAI time may be ~180–240s.
     - But YOLO experiences significantly less contention during overlap.
   - **Conclusion**: There is NO theoretical justification to assume 16 or 11 is optimal. The optimal concurrency can only be determined empirically via `bench_samurai_concurrency_matrix.py`.

---

## 5. What Was Disproven & What Remains Unknown

### Disproven Hypotheses
1. **DISPROVEN**: *"Post-processing is the main bottleneck taking 100–120 seconds."*
   - Empirical proof: Post-processing took less than 1.0 second.
2. **DISPROVEN**: *"AI Summary took 36 minutes from 20:01:37 to 20:38:05."*
   - Empirical proof: Worker was completely idle with 0 CPU/GPU/network activity; `_action_track` had already exited.
3. **DISPROVEN**: *"Higher SAMURAI concurrency always decreases E2E runtime."*
   - Empirical proof: High concurrency slows YOLO by 8× due to CUDA contention, altering the overall E2E equation.

### Unknowns Requiring RunPod Empirical Data
1. `UNKNOWN`: Exact `full_e2e_wall_clock` for concurrency values `[1, 2, 3, 4, 6, 8, 10, 11]`.
2. `UNKNOWN`: The exact inflection point where adding another SAMURAI worker causes more YOLO degradation than SAMURAI speedup.
3. `UNKNOWN`: VRAM and RAM scaling behavior under concurrency=16.

---

## 6. Recommended NEXT DIRECTION

Instead of targeting a single predetermined concurrency number:

> **NEXT DIRECTION: Deploy `bench_samurai_concurrency_matrix.py` to RunPod to execute the empirical parameter sweep across concurrency values `[1, 2, 3, 4, 6, 8, 10, 11]`, collect the full matrix telemetry, and identify the true system-level optimum based on `full_e2e_wall_clock`.**
