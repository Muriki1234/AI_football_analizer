# AI Football Assistant — RunPod Deployment Guide

A step-by-step runbook to go from zero to a working, independent production deployment on RunPod. Targeted at operators who don't want to think — copy, paste, click, done.

---

## 0. What you're building

| Layer | Tech | Where it lives |
|---|---|---|
| Backend API | FastAPI + uvicorn, single-worker, GPU | RunPod **Pod** (1× RTX 4090 or A4000) |
| Storage | `/workspace` Network Volume (SQLite + outputs) | RunPod **Network Volume**, 50 GB |
| Model weights | `soccana_best.pt`, `soccana_kpts_best.pt` | **Hugging Face Hub** private repo |
| Container image | `ghcr.io/<you>/ai-football-assistant:v1` | **GitHub Container Registry** |
| Frontend | Vite + React | Vercel / Netlify / any static host |

Two RunPod deployment shapes are supported:

1. **Pod** (recommended for now) — always-on GPU worker serving the FastAPI HTTP API directly. Stable, simple, has SSE streaming.
2. **Serverless** (future) — `server/handler.py` is a stub that can be registered as a RunPod Serverless worker. Not wired into the frontend yet.

This guide walks through the **Pod** path end to end.

---

## 1. Prerequisites

You need accounts and CLIs on your **local Mac**:

```bash
# GitHub CLI (for GHCR)
brew install gh
gh auth login               # pick HTTPS, login via browser

# Hugging Face CLI (for weights)
pip install -U "huggingface_hub[cli]"
huggingface-cli login       # paste a "Write" access token

# Docker Desktop (for building the image)
# https://www.docker.com/products/docker-desktop — launch it, wait for the whale 🐳 to turn green

# Optional: RunPod CLI (not required, web UI is fine)
pip install runpod
```

You'll also need:

- A **RunPod** account with some credit (~$5 covers hours of 4090 time).
- A **DashScope** (Qwen-VL) API key for the AI summary feature. Sign up at <https://dashscope.console.aliyun.com/>.

---

## 2. Upload model weights to Hugging Face

We don't ship model weights in the Docker image (100+ MB bloats pulls). The server downloads them from HF Hub on first boot and caches them to `/workspace/weights/` on the Network Volume.

### 2.1 Create a private HF repo

```bash
# Replace `hansdu` with your HF username
huggingface-cli repo create ai-football-assistant-weights --type model --private
```

### 2.2 Push the two `.pt` files

The two files — `soccana_best.pt` (YOLO detector) and `soccana_kpts_best.pt` (pitch keypoint model) — live in your Colab Drive. Download them to your Mac, then:

```bash
cd ~/Downloads                  # or wherever the .pt files are
huggingface-cli upload hansdu/ai-football-assistant-weights \
  soccana_best.pt soccana_best.pt
huggingface-cli upload hansdu/ai-football-assistant-weights \
  soccana_kpts_best.pt soccana_kpts_best.pt
```

Verify on `https://huggingface.co/hansdu/ai-football-assistant-weights/tree/main` — you should see both files.

### 2.3 Mint a read-only token for the server

Go to <https://huggingface.co/settings/tokens> → **Create new token** → role **Read**. Name it `runpod-pull`. Copy the `hf_...` string — we'll paste it into RunPod env vars in step 5.

---

## 3. Build and push the Docker image

### 3.1 From the repo root on your Mac

```bash
cd ~/Desktop/AI_Football_Assistant

# 3.1.1 Build (first run ~5-10 min; CUDA base image is ~3 GB)
docker build -t ghcr.io/<your-gh-username>/ai-football-assistant:v1 .

# 3.1.2 Sanity-check locally (CPU-only, expect /ready to 503 until weights pull)
docker run --rm -p 8000:8000 \
  -e API_KEY=local-test \
  -e HF_REPO_ID=hansdu/ai-football-assistant-weights \
  -e HF_TOKEN=hf_xxx... \
  ghcr.io/<your-gh-username>/ai-football-assistant:v1
# in another terminal: curl http://127.0.0.1:8000/health  → {"status":"ok"}
```

> Replace `<your-gh-username>` with your actual GitHub username (lowercase). Edit `Makefile` `IMAGE` default too if you want `make docker-build` to just work.

### 3.2 Push to GHCR

```bash
# Login to GHCR using the `gh` CLI's OAuth token
echo $(gh auth token) | docker login ghcr.io -u <your-gh-username> --password-stdin

# Push
docker push ghcr.io/<your-gh-username>/ai-football-assistant:v1
```

### 3.3 Make the package pullable

By default GHCR images inherit repo visibility. If your repo is private and you want RunPod to pull without auth, go to:

`https://github.com/users/<you>/packages/container/ai-football-assistant/settings`

Scroll to **Danger Zone → Change package visibility → Public**.

(If you prefer private, add `registry_auth` secrets to the RunPod Pod template — but public is simpler.)

---

## 4. Create the RunPod Network Volume

Network Volumes persist across Pod restarts. Everything in `/workspace` survives: SQLite session DB, uploaded videos, cached weights, rendered outputs.

1. Go to **<https://www.runpod.io/console/user/storage>**.
2. Click **New Network Volume**.
3. Pick a datacenter with 4090 availability (e.g. **CA-MTL-1** or **EU-RO-1**).
4. Name: `football-assistant-data`. Size: **50 GB** (adjust later).
5. Click **Create**. Note the **Volume ID** — you'll attach it in the next step.

---

## 5. Create the RunPod Pod

1. Go to **<https://www.runpod.io/console/deploy>**.
2. Filter by **GPU Type**: `RTX 4090` (or `RTX A4000` for cheaper dev). **Important**: pick a datacenter that **matches your Network Volume's region** (Network Volumes are region-locked).
3. Click **Deploy** on a candidate.
4. In the deploy dialog:

   **Template → click "Custom" → "Edit Template"**

   | Field | Value |
   |---|---|
   | Container Image | `ghcr.io/<you>/ai-football-assistant:v1` |
   | Container Disk | `20 GB` |
   | Volume Disk | *(leave blank — we're using a Network Volume instead)* |
   | Volume Mount Path | `/workspace` |
   | Expose HTTP Ports | `8000` |
   | Expose TCP Ports | *(leave blank)* |

   **Network Volume** → select `football-assistant-data` (mounts at `/workspace` automatically).

   **Environment Variables** (click **+ Add**):

   ```
   API_KEY=<generate-a-long-random-string>
   HF_REPO_ID=hansdu/ai-football-assistant-weights
   HF_TOKEN=hf_xxx...                    # the read-only token from step 2.3
   DASHSCOPE_API_KEY=sk-xxx...           # your Qwen-VL key
   CORS_ORIGINS=https://your-frontend.vercel.app   # or * for dev
   LOG_LEVEL=INFO
   ```

   To generate an API key: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`

5. Click **Deploy On-Demand**. Wait ~60 s for the image pull + container start.

### 5.1 Grab the Pod's public URL

Once the Pod status is **Running**:

1. Click the Pod → **Connect** → **HTTP Services** → **Port 8000**.
2. Copy the URL — it looks like `https://abc123-8000.proxy.runpod.net`. This is your backend.

### 5.2 Smoke test

```bash
export POD_URL="https://abc123-8000.proxy.runpod.net"
export API_KEY="<the-key-you-set>"

# Liveness
curl $POD_URL/health
# → {"status":"ok"}

# Readiness (waits for weights download on first boot — can take 30-60 s)
curl $POD_URL/ready
# → {"ready":true,"weights":true,"db":true}

# Authenticated session create
curl -X POST $POD_URL/api/sessions \
  -H "X-API-Key: $API_KEY" \
  -F "file=@/path/to/short-clip.mp4"
# → {"session_id":"...","status":"uploaded",...}
```

If `/ready` returns `{"weights":false}`, check the Pod logs (**Logs** tab) — usually it's a bad `HF_TOKEN` or the HF repo name is wrong.

---

## 6. Deploy the frontend

The frontend is a plain Vite build; drop it anywhere that serves static files. Vercel is the path of least resistance.

### 6.1 Set env vars

Create `frontend/.env.production` (gitignored — or set these in Vercel's UI):

```
VITE_API_BASE_URL=https://abc123-8000.proxy.runpod.net
VITE_API_KEY=<the-same-API_KEY-from-RunPod>
```

### 6.2 Build and deploy

```bash
cd frontend
npm install
npm run build
# dist/ is the output — drag into Vercel / Netlify
```

Or with Vercel CLI:

```bash
npm i -g vercel
cd frontend
vercel --prod
# When asked for env vars, paste VITE_API_BASE_URL and VITE_API_KEY
```

### 6.3 Update CORS

Back in the RunPod Pod → **Edit Pod** → update `CORS_ORIGINS` to exactly your frontend origin (e.g. `https://football.vercel.app`). Restart the Pod. Same-origin * is fine for testing but don't ship it.

---

## 7. First end-to-end test

1. Open `https://football.vercel.app` (or wherever you deployed).
2. **Upload** a 10-30 second clip (MP4, 720p+). Watch the progress toast.
3. **Trim** — click **Use whole video** to skip, or drag the handles.
4. **Dashboard** — you should see the stage label tick through `extracting → detecting → tracking → rendering`, with features unlocking one by one.
5. Click any feature tile — the artifact (video or chart) should render inline.

If it hangs on the first stage for >60 s, open the browser **Network** tab. You want to see:

- `POST /api/sessions/<id>/analyze` → 202
- `GET /api/sessions/<id>/events` → `text/event-stream` staying open, receiving `event: session` and `event: task` frames

If SSE never connects, the most common cause is Vercel's `middleware.ts` eating the upgrade headers, or Cloudflare in front of RunPod buffering the stream. Both are fixable but rarely hit in this stack.

---

## 8. Operations

### Pod logs

**Pod → Logs** tab, or via CLI: `runpodctl logs <pod-id>`. The app logs structured JSON at INFO level by default.

### Updating the image

```bash
# Local: bump TAG, rebuild, push
docker build -t ghcr.io/<you>/ai-football-assistant:v2 .
docker push ghcr.io/<you>/ai-football-assistant:v2

# RunPod: Pod → Edit Pod → Container Image → set :v2 → Save + Restart
```

The Network Volume is untouched, so sessions survive the rolling restart (though in-flight analyses are killed — clients just have to re-upload).

### Clearing stuck sessions

Sessions older than 48 h auto-clean via the APScheduler job. To nuke manually:

```bash
# From your Mac:
ssh root@<pod-ssh-host> -p <pod-ssh-port>
rm -rf /workspace/outputs/* /workspace/uploads/* /workspace/sessions.db*
# Restart Pod
```

### Cost checkpoints

| GPU | $/hr (on-demand) | Notes |
|---|---|---|
| RTX A4000 | ~$0.17 | Enough for 1-2 concurrent analyses |
| RTX 4090 | ~$0.39 | Snappy, what we'd actually ship |
| A100 40GB | ~$1.60 | Overkill unless you start running LLMs locally |

Storage: Network Volume is ~$0.07/GB/month; 50 GB = ~$3.50/month.

### Auto-stop to save money

Pods bill per second while **Running**. In the dashboard you can set an **Idle Timeout**: Pod pauses after N minutes of no HTTP traffic, resumes on next request (~20 s cold start). Works great for low-traffic demo.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `/ready` → `{"weights":false}` | HF token wrong or repo private + no token | Re-check `HF_TOKEN`, confirm repo name exactly |
| Upload works but analyze never starts | API_KEY mismatch between frontend and backend | Check `VITE_API_KEY` === Pod `API_KEY` |
| SSE connects but no events | EventBus not bound to loop | Pod restarted mid-request — retry |
| `CUDA out of memory` in logs | Too-long clip + 4090 | Trim to <60 s, or upgrade to A100 |
| Qwen-VL summary task fails with 401 | DashScope key wrong / region mismatch | Use the `intl` DashScope endpoint key |
| Rendered overlay videos look broken in browser | Container missing ffmpeg `libx264` | Already installed in our Dockerfile — check logs for actual ffmpeg errors |

### Emergency rollback

If a new image breaks everything:

```bash
# Pod → Edit Pod → Container Image → ghcr.io/<you>/ai-football-assistant:v1 (the last good one) → Save + Restart
```

The Network Volume data is fine. Frontend keeps working as-is.

---

## 10. What's next (optional)

- **Serverless migration**. Wire `server/handler.py` to a RunPod Serverless endpoint for true cold-start scale-to-zero. Handler is already written; just needs `runpod-python` config + frontend flag.
- **Real domain + HTTPS**. Put Cloudflare in front of the RunPod proxy URL for a clean domain and better caching of `/artifacts/*`.
- **S3 for artifacts**. Move `/workspace/outputs` behind a presigned-URL layer so you can scale the Pod down to zero without losing rendered videos.

Ship the basic Pod first. Everything else is an optimization.
