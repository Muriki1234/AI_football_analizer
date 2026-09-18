# AI Football Assistant ⚽️🤖

A state-of-the-art Football (Soccer) Tactical Analysis system leveraging **YOLO** (for object detection), **SAMURAI / SAM2** (for zero-shot promptable tracking), and **React** (for interactive telestration and data visualization).

## 🌟 Key Features
- **AI-Powered Tracking**: Utilizes YOLOv8/v11 for player and ball detection, combined with ByteTrack and SAM2 for robust multi-object tracking.
- **Interactive Telestration**: React-based canvas overlays allowing coaches to draw arrows, highlight players, and measure distances directly on the video.
- **Tactical Minimap**: Homography-based perspective transformation mapping 2D broadcast camera angles to a top-down 2D tactical pitch.
- **Cloud Infrastructure**: Vercel Serverless routing, Cloudflare R2 for zero-egress large video storage, and RunPod Serverless GPU/CPU pools for cost-optimized inference.
- **Advanced Metrics**: Calculates player velocities, total distance, possession, and team centroid lines.

## 🏗 Architecture
- **`/frontend`**: React + Vite + Framer Motion. Uses optimized `requestAnimationFrame` and `OffscreenCanvas` for buttery smooth 60fps video overlays without DOM thrashing.
- **`/server`**: FastAPI + PyTorch. Dispatches heavy GPU tasks (YOLO+SAM2) to a RunPod GPU endpoint, while lightweight data tasks (heatmaps, summaries) are routed to a cheaper CPU endpoint to save up to 90% in costs.
- **`/samurai`**: SAMURAI submodule bridging Segment Anything 2 (SAM2) for temporally consistent mask tracking.

## 🚀 Getting Started (Local Development)

We provide a complete `docker-compose.yml` for local development. Note that the backend requires an NVIDIA GPU and the `nvidia-container-toolkit` for CUDA acceleration.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/AI_Football_Assistant.git
cd AI_Football_Assistant

# 2. Set up environment variables
cp server/.env.example server/.env

# 3. Start the stack (Backend on port 8000, Frontend on port 5173)
docker-compose up --build
```

## 🔒 Security & Deployment
- Videos are uploaded directly to **Cloudflare R2** via presigned URLs to bypass Supabase's 50MB limits.
- R2 objects are automatically pruned via bucket lifecycle rules aligned with the `SESSION_TTL_HOURS`.
- CI/CD is automated via GitHub Actions for both Node.js (ESLint) and Python (Flake8).

## 📄 License
MIT License
