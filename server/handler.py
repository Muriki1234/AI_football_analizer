"""
RunPod Serverless handler. Receives job payloads from the Vercel proxy and
drives the analysis pipeline.

Serverless workers spin up on demand, receive one job payload, and exit. The
payload tells us what to do; we drive the same pipeline entry points that
the REST routes call.

⚠️  ORCHESTRATION DRIFT WARNING
    This module and server/routes/ are two separate orchestration layers on
    top of the same pipeline (pipeline/tasks.py) and store (storage/db.py).
    They differ in shape:
      - handler.py (Serverless, production frontend uses this):
          one action runs SAMURAI + analysis CONCURRENTLY via a daemon thread.
      - routes/analysis.py (Pod, dev/legacy):
          separate POST /track and POST /analyze endpoints, run SEQUENTIALLY.
    When you change high-level flow (status transitions, error handling,
    auto-trigger of full_replay, etc.) — touch BOTH. There is no shared
    "orchestrator" module yet; this comment is the only signpost.

Full Serverless spec: https://docs.runpod.io/serverless/handlers/overview

Supported actions:
    "detect_frame" — Quick YOLO on a single frame (player selection UI)
    "track"        — SAMURAI tracking + full analysis
    "analyze"      — Full analysis (requires existing SAMURAI cache)
    "feature"      — Generate a single feature (heatmap, replay, etc.)

Example input:
    {
      "action": "analyze",
      "session_id": "abc123",
      "video_url": "https://.../clip.mp4",
      "feature": "heatmap"                     # required if action == "feature"
    }
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from supabase import create_client

from .config import settings
from .models.weights import ensure_weights
from .pipeline import tasks as pipeline_tasks
from .storage.db import SessionManager

log = logging.getLogger(__name__)

_sm: SessionManager | None = None
_supabase_client = None   # module-level cache — see _get_supabase()


def _get_supabase():
    """
    Cached service-role Supabase client. Building one isn't expensive, but
    we were calling create_client() once per video download AND once per
    annotated-frame upload, which adds up to 2 fresh clients per analysis.
    Module-level singleton kills that.
    """
    global _supabase_client
    if _supabase_client is None:
        if not (settings.SUPABASE_URL and settings.SUPABASE_SERVICE_KEY):
            return None
        _supabase_client = create_client(
            settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY
        )
    return _supabase_client


def _get_sm() -> SessionManager:
    global _sm
    if _sm is None:
        # CPU pool 不跑任何模型推理（只做 ai_summary + matplotlib 图表），
        # 不需要 YOLO / keypoint 权重。跳过下载省 cold start + 不需要 HF_TOKEN。
        # GPU pool (默认) 保持原样 —— 启动时拉权重，下游 Tracker 才能加载。
        if os.environ.get("WORKER_MODE", "gpu").strip().lower() != "cpu":
            ensure_weights()
        _sm = SessionManager(output_root=settings.output_root)
    return _sm


def _supabase_storage_key_from_url(url: str) -> str | None:
    """
    Extract the storage object key from a Supabase Storage URL.

    Public URL shape:  https://<proj>.supabase.co/storage/v1/object/public/<bucket>/<key>
    Signed URL shape:  https://<proj>.supabase.co/storage/v1/object/sign/<bucket>/<key>?token=...
    Returns the <key> portion (everything after <bucket>/), or None if the URL
    doesn't look like Supabase Storage.

    安全：必须先校验 host 是不是我们配置的 SUPABASE_URL —— 否则攻击者构造
    https://attacker.example.com/storage/v1/object/public/videos/victim/x.mp4
    时仅看 path 就会被当成 Supabase URL，让我们用 service_role 去下别人 bucket 里的对象。
    深度防御层（_download_video_once 里也校验过一次 host）。
    """
    try:
        parsed = urlparse(url)
        path = parsed.path
        marker = "/storage/v1/object/"
        idx = path.find(marker)
        if idx == -1:
            return None
        # Host 必须匹配 SUPABASE_URL 的 host。任一侧异常都拒绝。
        try:
            expected_host = urlparse(os.environ["SUPABASE_URL"]).hostname
        except Exception:
            return None
        if not expected_host or parsed.hostname != expected_host:
            return None
        # path tail: public/<bucket>/<key...>  or  sign/<bucket>/<key...>
        tail = path[idx + len(marker):]
        parts = tail.split("/", 2)
        if len(parts) < 3:
            return None
        return parts[2]
    except Exception:
        return None


_DOWNLOAD_RETRIES = 4
_DOWNLOAD_BACKOFF_BASE = 2.0   # 2s, 4s, 8s, 16s


def _download_video(url: str, dest: Path) -> None:
    """
    Download a video from Supabase Storage to local disk using the service-role
    SDK. This works whether the bucket is public or private — the service key
    bypasses RLS. Falls back to raw HTTP only if the URL isn't a recognizable
    Supabase Storage URL.

    Retries on any error with exponential backoff (4 attempts: 2/4/8/16 s).
    For large videos over flaky RunPod networking this is essential — a
    single TCP reset previously failed the whole pipeline.
    """
    import time as _time

    last_exc: Exception | None = None
    for attempt in range(_DOWNLOAD_RETRIES):
        try:
            _download_video_once(url, dest)
            # Sanity check: empty / truncated file should also retry
            size = dest.stat().st_size if dest.exists() else 0
            if size < 1024:
                raise RuntimeError(f"downloaded file too small ({size} B)")
            if attempt > 0:
                log.info("Video download succeeded on attempt %d", attempt + 1)
            return
        except Exception as exc:
            last_exc = exc
            if attempt < _DOWNLOAD_RETRIES - 1:
                wait = _DOWNLOAD_BACKOFF_BASE * (2 ** attempt)
                log.warning(
                    "Video download attempt %d/%d failed: %s — retrying in %.0fs",
                    attempt + 1, _DOWNLOAD_RETRIES, exc, wait,
                )
                _time.sleep(wait)
                # Wipe partial file before retry to avoid corrupted state
                try:
                    if dest.exists():
                        dest.unlink()
                except Exception:
                    pass
            else:
                log.error(
                    "Video download exhausted all %d attempts: %s",
                    _DOWNLOAD_RETRIES, exc,
                )
    raise RuntimeError(
        f"Failed to download {url[:80]}… after {_DOWNLOAD_RETRIES} attempts: {last_exc}"
    )


_MAX_DOWNLOAD_BYTES = int(os.environ.get("MAX_VIDEO_DOWNLOAD_BYTES", str(10 * 1024**3)))   # 10 GB


def _is_allowed_video_url(url: str) -> bool:
    """
    SSRF防线：拒绝 file://、本地 / 内网地址、云元数据 IP 等。只放行 https
    且 host 必须在白名单内（Supabase Storage、R2、CDN）。

    没这个校验时，攻击者发 {"video_url": "file:///etc/passwd"} 或
    "http://169.254.169.254/latest/meta-data/..." 后端会乐呵呵下载。
    """
    from urllib.parse import urlparse
    try:
        u = urlparse(url)
    except Exception:
        return False
    if u.scheme not in ("http", "https"):
        return False
    if not u.hostname:
        return False
    host = u.hostname.lower()
    # 拒绝任何 IP 直连（包括 169.254.169.254 元数据 / 127.0.0.1 / 内网）
    if host.replace(".", "").isdigit() or ":" in host:  # IPv4 / IPv6
        return False
    if host in ("localhost",) or host.endswith(".local") or host.endswith(".internal"):
        return False
    # 显式白名单 — 默认放 Supabase / Cloudflare R2，其他靠 env 加
    ALLOW = (
        ".supabase.co",
        ".supabase.in",
        ".r2.cloudflarestorage.com",
        ".r2.dev",
    )
    extra = os.environ.get("VIDEO_URL_ALLOWED_HOST_SUFFIXES", "")
    if extra:
        ALLOW = ALLOW + tuple(s.strip() for s in extra.split(",") if s.strip())
    return any(host == s.lstrip(".") or host.endswith(s) for s in ALLOW)


def _head_probe_size(url: str, headers: dict | None = None) -> int | None:
    """
    HEAD 预探 Content-Length。命中就能在握手阶段就拒掉超大文件，比开 GET 再
    看 response header 更早断开 —— 部分对象存储（含 Supabase）会真返回 size。
    返回 None 表示拿不到 / HEAD 不支持，调用方应回退到 GET 流式 + post-check。
    """
    import urllib.request
    try:
        req = urllib.request.Request(url, headers=headers or {}, method="HEAD")
        req.add_header("User-Agent", "pitchlogic/1.0")
        with urllib.request.urlopen(req, timeout=30) as resp:
            cl_str = resp.headers.get("Content-Length") or ""
            if not cl_str:
                return None
            try:
                return int(cl_str)
            except ValueError:
                return None
    except Exception:
        return None


def _stream_download(url: str, dest: Path, headers: dict | None = None) -> None:
    """流式 GET → 写盘。HEAD 预探 + GET Content-Length + 边读边校验，三层防护。

    P1.2 修复：之前若服务器不返回 Content-Length，唯一防线是边读边校验大小；
    Supabase SDK `.download()` 是把整个对象一次性读进内存，10GB 恶意文件可以
    在我们 cap 还没生效之前就把 worker OOM。这里改成自己用 urllib 流式下载，
    并先 HEAD 预探（Supabase Storage 真的会返回 Content-Length），命中就在
    握手阶段就拒掉。
    """
    import urllib.request
    # 第 1 层：HEAD 预探。失败 / 不支持就 fall through，不阻塞下载。
    head_size = _head_probe_size(url, headers=headers)
    if head_size is not None and head_size > _MAX_DOWNLOAD_BYTES:
        raise RuntimeError(
            f"refused: HEAD Content-Length {head_size // (1024**2)} MB > "
            f"{_MAX_DOWNLOAD_BYTES // (1024**2)} MB cap"
        )

    req = urllib.request.Request(url, headers=headers or {})
    req.add_header("User-Agent", "pitchlogic/1.0")
    with urllib.request.urlopen(req, timeout=300) as resp:
        # 第 2 层：GET response header 的 Content-Length（HEAD 不支持时的兜底）
        cl_str = resp.headers.get("Content-Length") or "0"
        try:
            content_length = int(cl_str)
        except ValueError:
            content_length = 0
        if content_length and content_length > _MAX_DOWNLOAD_BYTES:
            raise RuntimeError(
                f"refused: Content-Length {content_length // (1024**2)} MB > "
                f"{_MAX_DOWNLOAD_BYTES // (1024**2)} MB cap"
            )
        # 第 3 层：边读边累加 —— 服务器撒谎 / chunked 无 Content-Length 时唯一防线
        bytes_read = 0
        with dest.open("wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                bytes_read += len(chunk)
                if bytes_read > _MAX_DOWNLOAD_BYTES:
                    out.close()
                    try:
                        dest.unlink()
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"refused: stream exceeded {_MAX_DOWNLOAD_BYTES // (1024**2)} MB cap"
                    )
                out.write(chunk)
        if head_size is None and content_length == 0:
            # 服务器既无 HEAD 也无 Content-Length —— 记一笔便于后续排查
            log.warning(
                "Video download had no Content-Length (HEAD probe also failed); "
                "relied solely on streaming cap. URL=%s…", url[:80],
            )


def _supabase_host() -> str | None:
    """从 settings.SUPABASE_URL 抠出 host，用于校验 video_url 是不是真的指向自己 Supabase。"""
    try:
        return urlparse(settings.SUPABASE_URL).hostname
    except Exception:
        return None


def _download_video_once(url: str, dest: Path) -> None:
    """Single download attempt — the retry loop is in _download_video()."""
    # 关键安全检查：先校验 scheme + host，再决定走 Supabase 路径还是公网 fallback。
    # 之前的写法是先抠 storage key 再判断，攻击者只要在 path 里塞 /storage/v1/object/
    # 字串就能跳过白名单，让我们用 service_role 去下任意 storage key（48-bit session id 可枚举）。
    try:
        u = urlparse(url)
    except Exception:
        raise RuntimeError(f"invalid video_url: {url[:80]}…")
    if u.scheme not in ("http", "https") or not u.hostname:
        raise RuntimeError(f"refused: bad scheme/host in video_url")

    storage_key = _supabase_storage_key_from_url(url)
    supa_host = _supabase_host()

    # 路径 1：真正的 Supabase Storage URL（host 必须匹配我们配置的 SUPABASE_URL）
    if storage_key and supa_host and u.hostname == supa_host:
        # 用 service_role bearer + 直接 HTTP GET 流式下载，避开 SDK 的 .download()
        # 把整个文件预先读到内存（10GB 文件入内存 = 立刻 OOM）。
        base = settings.SUPABASE_URL.rstrip("/")
        storage_url = f"{base}/storage/v1/object/videos/{storage_key}"
        _stream_download(
            storage_url,
            dest,
            headers={"Authorization": f"Bearer {settings.SUPABASE_SERVICE_KEY}"},
        )
        return

    # 路径 2：外部 URL（R2 / CDN / 其他）— 必须过 host 白名单
    if not _is_allowed_video_url(url):
        raise RuntimeError(
            f"refused: video_url host not in allowlist (got {url[:80]}…)"
        )
    _stream_download(url, dest)


_MAX_ANALYSIS_HEIGHT = 720  # downscale anything above this to 720p to accelerate YOLO/SAMURAI


def _maybe_normalize_video(video_path: Path) -> bool:
    """If the video is taller than 1080p or >30fps, re-encode in-place.

    YOLO internally resizes to 640px, SAMURAI uses RESIZE_FACTOR=0.5.
    >1080p or >30fps is wasted data that just eats RAM and slows down parallel processing.
    """
    import subprocess as _sp

    try:
        # Probe resolution and fps with ffprobe
        probe = _sp.run(
            ["ffprobe", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate",
             "-of", "csv=p=0",
             str(video_path)],
            capture_output=True, text=True, timeout=30,
        )
        parts = probe.stdout.strip().split(",")
        if len(parts) < 3:
            print(f"[NORMALIZE] ⚠️  ffprobe returned unexpected output: {probe.stdout!r}", flush=True)
            return True  # can't determine, assume ok
        w, h = int(parts[0]), int(parts[1])
        fps_str = parts[2]
        fps_num, fps_den = fps_str.split("/") if "/" in fps_str else (fps_str, "1")
        fps = float(fps_num) / float(fps_den) if float(fps_den) > 0 else 30.0
    except Exception as exc:
        print(f"[NORMALIZE] ⚠️  ffprobe failed: {exc} — skipping normalization check", flush=True)
        return True  # can't determine, assume ok

    needs_downscale = h > _MAX_ANALYSIS_HEIGHT
    needs_decimate = fps > 31.0  # Allow some margin for 30.03, 30.0, etc.

    if not needs_downscale and not needs_decimate:
        print(f"[NORMALIZE] ✅ Video is {w}x{h} @ {fps:.1f}fps, no normalization needed", flush=True)
        return True

    print(f"[NORMALIZE] 🔄 Video is {w}x{h} @ {fps:.1f}fps, normalizing to "
          f"{min(h, _MAX_ANALYSIS_HEIGHT)}p @ {min(fps, 30.0):.1f}fps…", flush=True)

    tmp_path = video_path.with_suffix(".normalized.mp4")
    
    # Scale filter only if needed
    scale_cuda = f"-vf scale_cuda=-2:{_MAX_ANALYSIS_HEIGHT}" if needs_downscale else ""
    scale_cpu = f"-vf scale=-2:{_MAX_ANALYSIS_HEIGHT}" if needs_downscale else ""
    
    # Framerate filter
    fps_arg = ["-r", "30"] if needs_decimate else []

    # Try GPU encoder first, fall back to CPU.
    for encoder in ("h264_nvenc", "libx264"):
        if encoder == "h264_nvenc":
            cmd = [
                "ffmpeg", "-y", 
                "-hwaccel", "cuda", 
                "-hwaccel_output_format", "cuda",
                "-i", str(video_path),
            ]
            if scale_cuda:
                cmd.extend(["-vf", f"scale_cuda=-2:{_MAX_ANALYSIS_HEIGHT}"])
            cmd.extend(fps_arg)
            cmd.extend(["-c:v", "h264_nvenc", "-preset", "p1", "-cq", "28", "-an", str(tmp_path)])
        else:
            cmd = ["ffmpeg", "-y", "-i", str(video_path)]
            if scale_cpu:
                cmd.extend(["-vf", f"scale=-2:{_MAX_ANALYSIS_HEIGHT}"])
            cmd.extend(fps_arg)
            cmd.extend(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-an", str(tmp_path)])
            
        try:
            result = _sp.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 1024:
                orig_size = video_path.stat().st_size
                video_path.unlink()
                tmp_path.rename(video_path)
                new_size = video_path.stat().st_size
                print(f"[NORMALIZE] ✅ SUCCESS: {w}x{h} @ {fps:.1f}fps → {min(h, _MAX_ANALYSIS_HEIGHT)}p @ {min(fps, 30.0):.1f}fps, "
                      f"{orig_size//(1024*1024)}MB → {new_size//(1024*1024)}MB (encoder={encoder})", flush=True)
                return True
            else:
                stderr_tail = (result.stderr or "")[-200:]
                print(f"[NORMALIZE] ⚠️  {encoder} failed (rc={result.returncode}): {stderr_tail}", flush=True)
                if tmp_path.exists():
                    tmp_path.unlink()
        except Exception as exc:
            print(f"[DOWNSCALE] ⚠️  {encoder} error: {exc}", flush=True)
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    print(f"[DOWNSCALE] ❌ FAILED: all encoders failed for {w}x{h} video. "
          f"Analysis will proceed at original resolution with REDUCED parallelism.",
          flush=True)
    return False


def _ensure_local_video(session_id: str, video_url: str, sm: SessionManager) -> str:
    """
    Make sure the video exists on local disk. If not, download it from
    Supabase Storage. Returns the local path.

    Important: this function may be called LATE in the lifecycle —
    e.g. when an ai_summary feature task runs on a CPU worker for a
    session that already finished analysis. In that case we MUST NOT
    overwrite the session.status back to "uploaded" (which would route
    the user back to the pick-player screen). Preserve whatever status
    the session already has and only patch in video_path.
    """
    local_dir = settings.upload_root / session_id
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / "video.mp4"

    if local_path.exists() and local_path.stat().st_size > 0:
        return str(local_path)

    if not video_url:
        raise ValueError(f"No video_url for session {session_id}")

    log.info("Downloading video from %s …", video_url[:80])
    _download_video(video_url, local_path)
    log.info("Downloaded %d MB", local_path.stat().st_size // (1024 * 1024))

    # Auto-downscale to 1080p and decimate to 30fps to keep RAM usage predictable
    # and parallelism at maximum (12 procs). No-op if already ≤1080p and ≤30fps.
    _maybe_normalize_video(local_path)

    # 不要无脑把 status 设成 "uploaded" — 那会把已经 analysis_done 的
    # session 拖回 pick-player 页面。读当前 status 再回写同一个值（实际上
    # 只是为了顺带把 video_path 字段更新到 DB）。新 session 第一次进来时
    # status 还没设，就当作 "uploaded"。
    try:
        current = sm.get_session(session_id) or {}
        preserved_status = current.get("status") or "uploaded"
    except Exception:
        preserved_status = "uploaded"
    sm.update_status(session_id, preserved_status, video_path=str(local_path))
    return str(local_path)


def _run_auto_full_replay(session_id: str, sm: SessionManager) -> None:
    """Generate the showcase replay once analysis finishes, unless it exists.

    P2.3 修复：之前是在当前 Serverless job 内 inline 跑 run_full_replay()，5-15min
    的 GPU 渲染会把当前 job wall-clock 顶爆，导致 Vercel proxy 拿到 502。改成
    递归 enqueue 一个新的 RunPod Serverless job：当前 job 立刻返回 analysis 结果，
    渲染任务在新 worker 上独立跑。前端用 SSE / 轮询 task 状态即可。

    安全：递归调用用的是 RunPod 自己的 API，不会触发外部 SSRF；endpoint URL 和
    API key 都是 worker 自己环境变量里的，攻击者控制不到。

    本地 dev 环境（没配 RUNPOD_ENDPOINT_URL / RUNPOD_API_KEY）会 fall back
    到 inline 执行，保留之前的行为。
    """
    existing = [
        t for t in sm.list_tasks(session_id)
        if t.get("task_type") == "full_replay"
        and t.get("status") in ("queued", "running", "done")
    ]
    if existing:
        return

    session = sm.get_session(session_id)
    if not session or session.get("status") != "analysis_done":
        return

    task_id = sm.create_task(session_id, "full_replay")

    # 优先：异步 enqueue 新 RunPod job。失败 / 本地 dev → fall back 到 inline。
    runpod_endpoint = os.environ.get("RUNPOD_ENDPOINT_URL", "").strip()
    runpod_api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if runpod_endpoint and runpod_api_key:
        try:
            import json as _json
            import urllib.request as _urlreq
            # RunPod Serverless 的 /run endpoint 接受 {"input": {...}}，异步排队。
            run_url = runpod_endpoint.rstrip("/")
            if not run_url.endswith("/run"):
                run_url = run_url + "/run"
            body = _json.dumps({
                "input": {
                    "action": "feature",
                    "feature": "full_replay",
                    "session_id": session_id,
                },
            }).encode("utf-8")
            req = _urlreq.Request(
                run_url,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {runpod_api_key}",
                    "Content-Type": "application/json",
                },
            )
            with _urlreq.urlopen(req, timeout=30) as resp:
                status_code = resp.status
                if 200 <= status_code < 300:
                    log.info(
                        "[auto-full-replay] enqueued NEW RunPod job for session %s "
                        "(task %s) — current job returning early",
                        session_id, task_id,
                    )
                    return
                body_text = resp.read(512).decode("utf-8", errors="replace")
                log.warning(
                    "[auto-full-replay] RunPod enqueue returned %s: %s — falling back to inline",
                    status_code, body_text,
                )
        except Exception as exc:
            log.warning(
                "[auto-full-replay] RunPod enqueue failed (%s) — falling back to inline",
                exc,
            )
    else:
        log.warning(
            "[auto-full-replay] RUNPOD_ENDPOINT_URL / RUNPOD_API_KEY not set; "
            "running full_replay inline (may exceed Serverless wall-clock cap)"
        )

    # Fallback: inline 执行（原行为）。仅在本地 dev / env 缺失时进。
    log.info("[auto-full-replay] inline run %s for session %s", task_id, session_id)
    pipeline_tasks.run_hls_replay(session_id, session, task_id, sm)


def _action_detect_frame(session_id: str, s: dict, payload: dict, sm: SessionManager) -> dict:
    """Quick YOLO on a single frame for the player-selection UI."""
    frame_idx = int(payload.get("frame", 0))
    result = pipeline_tasks.detect_frame_players(session_id, s, frame_idx, sm)

    output_dir = sm.session_output_dir(session_id)
    frame_path = output_dir / result.get("annotated_frame_path", "first_frame.jpg")
    frame_url = None

    if frame_path.exists():
        # Prefer R2 upload (no size limits, same infra as video upload)
        try:
            from .storage.r2 import upload_to_r2
            r2_url = upload_to_r2(frame_path, f"{session_id}/first_frame.jpg")
            if r2_url:
                frame_url = r2_url
        except Exception as r2_exc:
            log.warning("R2 frame upload failed, falling back to Supabase: %s", r2_exc)

        # Fallback: Supabase Storage (for backward compat / if R2 not configured)
        if not frame_url:
            supa = _get_supabase()
            if supa is not None:
                try:
                    storage_key = f"{session_id}/first_frame.jpg"
                    with open(frame_path, "rb") as f:
                        supa.storage.from_("videos").upload(
                            storage_key, f,
                            file_options={"content-type": "image/jpeg", "upsert": "true"}
                        )
                    frame_url = supa.storage.from_("videos").get_public_url(storage_key)
                except Exception as supa_exc:
                    log.warning("Supabase frame upload also failed: %s", supa_exc)

    return {
        "players": result.get("players", []),
        "players_data": result.get("players", []),
        "annotated_frame_url": frame_url,
        "image_dimensions": result.get("image_dimensions"),
    }


def _parse_match_periods(payload: dict, total_frames_hint: int, client_fps: float = None, actual_fps: float = 25.0) -> list[tuple[int, int]]:
    """
    Read match_periods from payload. Format from frontend:
      [[startFrame, endFrame], ...]  (sorted, non-overlapping)
    Defaults to a single full-video period if missing/empty.
    """
    scale_ratio = actual_fps / client_fps if client_fps and client_fps > 0 else 1.0

    raw = payload.get("match_periods")
    if not raw:
        return [(0, total_frames_hint)]
    out: list[tuple[int, int]] = []
    for p in raw:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            s = max(0, int(round(int(p[0]) * scale_ratio)))
            e = min(total_frames_hint, int(round(int(p[1]) * scale_ratio)))
            if e > s + 1:
                out.append((s, e))
    if not out:
        return [(0, total_frames_hint)]
    out.sort()
    return out


def _parse_segments(payload: dict, total_frames_hint: int = 1500,
                    periods: list[tuple[int, int]] | None = None,
                    client_fps: float = None, actual_fps: float = 25.0) -> list[dict]:
    """
    Normalize either:
      - single bbox (legacy): payload["bbox"] = {x1,y1,x2,y2} or [...]
      - segments array (new): payload["segments"] = [{frame, bbox, period_idx?}, ...]
    into the segments list shape that run_samurai_tracking_multi expects:
      [{"start_frame": int, "end_frame": int, "bbox": {x,y,w,h}, "period_idx": int}, ...]

    `periods` constrains each segment's end_frame to the bounds of its
    period — without this, the last seg in period 1 would span across
    a skipped halftime gap into period 2.
    """
    scale_ratio = actual_fps / client_fps if client_fps and client_fps > 0 else 1.0

    segments_in = payload.get("segments")
    if segments_in:
        out = []
        for seg in segments_in:
            bb = seg.get("bbox", {})
            if isinstance(bb, dict):
                x1 = float(bb.get("x1", bb.get("x", 0)))
                y1 = float(bb.get("y1", bb.get("y", 0)))
                x2 = float(bb.get("x2", bb.get("x", 0) + bb.get("w", 0)))
                y2 = float(bb.get("y2", bb.get("y", 0) + bb.get("h", 0)))
            elif isinstance(bb, (list, tuple)) and len(bb) == 4:
                x1, y1, x2, y2 = [float(v) for v in bb]
            else:
                continue
            
            scaled_frame = max(0, min(total_frames_hint - 1, int(round(int(seg.get("frame", 0)) * scale_ratio))))
            out.append({
                "start_frame": scaled_frame,
                "period_idx":  int(seg.get("period_idx", 0)),
                "bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
                "img_dims": seg.get("img_dims"),
            })
        out.sort(key=lambda s_: s_["start_frame"])

        # Fill end_frame:
        #   - if periods supplied: end is min(next seg in same period start,
        #                                     this period's end_frame)
        #   - if no periods:       end is next seg start, or video end
        for i, s_ in enumerate(out):
            same_period_next = next(
                (n["start_frame"] for n in out[i + 1:]
                 if n["period_idx"] == s_["period_idx"]),
                None
            )
            if periods:
                period_end = periods[s_["period_idx"]][1] \
                    if s_["period_idx"] < len(periods) else total_frames_hint
                s_["end_frame"] = min(same_period_next or period_end, period_end)
            else:
                s_["end_frame"] = same_period_next or total_frames_hint
        return out

    # Legacy single-bbox path (kept for backward compatibility with the
    # single-segment FastAPI route still calling through run_samurai_tracking)
    bbox_raw = payload.get("bbox", {})
    frame = int(payload.get("frame", 0))
    if isinstance(bbox_raw, dict):
        x1 = float(bbox_raw.get("x1", 0)); y1 = float(bbox_raw.get("y1", 0))
        x2 = float(bbox_raw.get("x2", 0)); y2 = float(bbox_raw.get("y2", 0))
    elif isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
        x1, y1, x2, y2 = [float(v) for v in bbox_raw]
    else:
        return []
    return [{
        "start_frame": frame,
        "end_frame": total_frames_hint,
        "period_idx": 0,
        "bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
    }]


def _action_track(session_id: str, s: dict, payload: dict, sm: SessionManager) -> dict:
    """
    SAMURAI tracking (multi-segment, parallel) + global analysis (concurrent
    on main thread) → auto-generate replay.

    The two GPU jobs run in parallel — SAMURAI on extracted JPGs via
    subprocesses, the analysis on the original video via the main process.
    They only sync up right before _compute_player_summary (which needs both).
    """
    import threading

    # Try to probe total_frames so we can default the last segment's end_frame.
    total_frames_hint = 1500
    actual_fps = 25.0
    video_path_local = s.get("video_path") or ""
    if video_path_local:
        try:
            import cv2 as _cv2  # noqa: WPS433 (local import for cold-start speed)
            cap = _cv2.VideoCapture(video_path_local)
            t = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
            f = float(cap.get(_cv2.CAP_PROP_FPS))
            cap.release()
            if t > 0:
                total_frames_hint = t
            if f > 0:
                actual_fps = f
        except Exception:
            pass

    client_fps = payload.get("client_fps")

    match_periods = _parse_match_periods(payload, total_frames_hint, client_fps=client_fps, actual_fps=actual_fps)
    segments = _parse_segments(payload, total_frames_hint=total_frames_hint, periods=match_periods, client_fps=client_fps, actual_fps=actual_fps)
    if not segments:
        return {"error": "Provide either bbox+frame or segments=[{frame,bbox},...]"}

    # ── Rescale bbox coordinates if video was downscaled ──────────────
    # The frontend (Roboflow path) sends bbox coords in the ORIGINAL video
    # resolution (e.g. 2880×1800 from the R2 public URL). But the backend
    # auto-downscales to 1080p after download (e.g. 1728×1080). Without
    # rescaling, SAMURAI's clamp logic crushes the bbox to ~2px → all fail.
    if video_path_local:
        try:
            import cv2 as _cv2r
            _cap = _cv2r.VideoCapture(video_path_local)
            _actual_w = int(_cap.get(_cv2r.CAP_PROP_FRAME_WIDTH)) or 0
            _actual_h = int(_cap.get(_cv2r.CAP_PROP_FRAME_HEIGHT)) or 0
            _cap.release()
            if _actual_w > 0 and _actual_h > 0:
                # Check if any bbox exceeds the actual video dims OR if img_dims were provided
                _needs_rescale = False
                for seg in segments:
                    img_dims = seg.get("img_dims")
                    if img_dims and img_dims.get("width") and img_dims.get("height"):
                        _img_w = float(img_dims["width"])
                        if abs(_img_w - _actual_w) > 5:
                            _needs_rescale = True
                            break

                if _needs_rescale:
                    print(f"[BBOX-RESCALE] Rescaling bboxes for actual video dims ({_actual_w}x{_actual_h})", flush=True)
                    for seg in segments:
                        _scale = 1.0
                        img_dims = seg.get("img_dims")
                        if img_dims and img_dims.get("width") and img_dims.get("height"):
                            _img_w = float(img_dims["width"])
                            _img_h = float(img_dims["height"])
                            _scale = min(_actual_w / _img_w, _actual_h / _img_h)
                            seg["bbox"]["x"] *= _scale
                            seg["bbox"]["y"] *= _scale
                            seg["bbox"]["w"] *= _scale
                            seg["bbox"]["h"] *= _scale
                        else:
                            print(f"[BBOX-RESCALE] ⚠️  Missing img_dims for segment {seg.get('start_frame')}, cannot rescale!", flush=True)
                else:
                    print(f"[BBOX-RESCALE] ✅ Bboxes fit within video "
                          f"({_actual_w}x{_actual_h}), no rescale needed",
                          flush=True)
        except Exception as _e:
            print(f"[BBOX-RESCALE] ⚠️  Could not verify bbox coords: {_e}",
                  flush=True)

    # Enforce minimum period length (point E).
    # Real match videos (total ≥ 300s × 25fps = 7500 frames) keep the 30s minimum
    # so SAMURAI has enough frames to build a reliable model.
    # Short test clips (< 5 min) use a relaxed 3s minimum so the UI works during dev.
    _fps_est = 25
    _min_sec = 30 if total_frames_hint >= 300 * _fps_est else 3
    MIN_PERIOD_FRAMES = _min_sec * _fps_est
    for ps, pe in match_periods:
        if pe - ps < MIN_PERIOD_FRAMES:
            return {"error": f"Period {ps}..{pe} shorter than {_min_sec}s minimum"}

    s_merged = {**s, "start_frame": segments[0]["start_frame"]}
    samurai_cache_path = str(sm.session_output_dir(session_id) / "samurai_tracking.pkl")
    sm.update_status(session_id, "tracking", progress=1,
                     stage="samurai_multi_pending",
                     samurai_cache_path=samurai_cache_path,
                     # Pin the periods on the session so downstream code
                     # (analysis, render) can read them without re-parsing
                     # the payload.
                     match_periods_frames=[list(p) for p in match_periods])

    s_merged = sm.get_session(session_id) or s_merged

    print(f"[TRACK] launching SAMURAI ({len(segments)} segment(s) across "
          f"{len(match_periods)} period(s)) || merged analysis in parallel")

    # Event-based handoff between SAMURAI thread and analysis thread.
    # Replaces the old filesystem busy-poll (sleep 1s in a loop): zero CPU
    # while waiting, instant wake when SAMURAI finishes, and no RunPod
    # seconds wasted spinning.
    samurai_done = threading.Event()
    samurai_err: dict = {}

    def _samurai_worker():
        try:
            pipeline_tasks.run_samurai_tracking_multi(
                session_id, s_merged, segments, sm
            )
        except Exception as e:
            samurai_err["exc"] = e
            log.exception("SAMURAI multi-segment failed")
        finally:
            samurai_done.set()

    samurai_thread = threading.Thread(target=_samurai_worker, daemon=True)
    samurai_thread.start()

    # Run analysis on this thread, concurrently with SAMURAI subprocesses.
    # run_global_analysis blocks on `samurai_done` (passed via attribute on
    # session dict) right before the summary step.
    s_merged["_samurai_done_event"] = samurai_done
    try:
        pipeline_tasks.run_global_analysis(session_id, s_merged, sm)
    finally:
        # Scale the thread.join() cap with video length too — 900s (15 min)
        # was fine for 30-min clips but broke 1.5h+ matches. Use 2× video
        # duration with a 15-min floor, computed from the same total_frames
        # we probed earlier.
        join_timeout = max(900.0, 2.0 * (total_frames_hint / 25.0))
        samurai_thread.join(timeout=join_timeout)

    # Daemon thread 超时后仍可能 is_alive=True：上一版直接 return ok，等于
    # 静默丢任务，DB 里 status=analysis_done 但 SAMURAI 还在背后跑 / 没结果。
    # 现在显式检查、写入失败状态、return error。
    #
    # ⚠️ P2.4 已知缺陷（暂未修）：abandon 的 daemon thread 里 SAMURAI 是用
    # ProcessPoolExecutor 起的子进程跑的 —— Python interpreter 退出时这些
    # 子进程不一定能被回收，可能继续在 GPU 上跑直到 RunPod 杀掉整个容器。
    # 真正的修法是让 run_samurai_tracking_multi 周期性检查
    # session["_samurai_kill_event"] 并主动 shutdown executor，跨层改动较大，
    # 单次 audit 不展开。下一轮 reliability pass 处理。短期靠 RunPod 的
    # job idle timeout 兜底（worker 退出时 SIGKILL 全部子进程）。
    if samurai_thread.is_alive():
        err = f"SAMURAI exceeded {join_timeout/60:.1f} min — abandoning thread"
        log.error(err)
        log.error(
            "[samurai-leak] daemon thread still alive — subprocess workers may "
            "continue eating GPU until container shutdown. See P2.4 comment."
        )
        try:
            sm.update_status(session_id, "tracking_failed", error=err)
        except Exception:
            pass
        return {"error": err}

    if samurai_err:
        return {"error": f"SAMURAI failed: {samurai_err['exc']}"}

    _run_auto_full_replay(session_id, sm)
    return {"ok": True, "session": sm.get_session(session_id)}


def _action_analyze(session_id: str, s: dict, payload: dict, sm: SessionManager) -> dict:
    """Full analysis (assumes SAMURAI cache already present)."""
    pipeline_tasks.run_global_analysis(session_id, s, sm)
    _run_auto_full_replay(session_id, sm)
    return {"ok": True, "session": sm.get_session(session_id)}


def _action_feature(session_id: str, s: dict, payload: dict, sm: SessionManager) -> dict:
    """Generate a single feature output (heatmap, speed chart, etc.)."""
    feature = payload.get("feature")
    from .routes.analysis import FEATURE_TASKS
    fn = FEATURE_TASKS.get(feature or "")
    if not fn:
        return {"error": f"unknown feature {feature!r}"}
    task_id = sm.create_task(session_id, feature)
    fn(session_id, s, task_id, sm)
    return {"ok": True, "task": sm.get_task(session_id, task_id)}


# Adding a new action = one entry here, no main-handler changes needed.
ACTIONS = {
    "detect_frame": _action_detect_frame,
    "track":        _action_track,
    "analyze":      _action_analyze,
    "feature":      _action_feature,
}

# ── Worker-pool dispatch ────────────────────────────────────────────────────
# Two RunPod Serverless endpoints share this handler:
#   - GPU endpoint (WORKER_MODE=gpu, the default): runs every action.
#   - CPU endpoint (WORKER_MODE=cpu, Dockerfile.cpu image): runs only the
#     no-GPU feature tasks listed below. Cheaper worker hours + faster cold
#     start. Reject GPU actions fast and clear so misrouted requests don't
#     silently crash on a missing torch import.
WORKER_MODE = os.environ.get("WORKER_MODE", "gpu").strip().lower()

# Features that don't need GPU compute. They can run on the CPU endpoint.
# Keep this in sync with the routing table in frontend/api/analyze.js.
_CPU_FEATURES = frozenset({
    "ai_summary",
    "heatmap",
    "speed_chart",
    "possession",
    "sprint_analysis",
    "defensive_line",
})

# Actions a CPU worker is allowed to run at all (everything else short-circuits).
_CPU_ACTIONS = frozenset({"feature"})


def handler(event: dict[str, Any]) -> dict[str, Any]:
    """Entry point called by the RunPod Serverless runtime."""
    payload = event.get("input", {}) or {}
    action = payload.get("action", "analyze")
    session_id = payload.get("session_id")
    if not session_id:
        return {"error": "session_id is required"}

    fn = ACTIONS.get(action)
    if not fn:
        return {"error": f"unknown action {action!r}"}

    # CPU pool guard: drop anything we can't serve here so the caller knows
    # to retry on the GPU pool (Vercel proxy decides routing, but a misroute
    # shouldn't bring down the worker on an ImportError).
    if WORKER_MODE == "cpu":
        if action not in _CPU_ACTIONS:
            return {
                "error": f"action {action!r} not supported on CPU worker; "
                         f"use the GPU endpoint",
                "worker_mode": "cpu",
            }
        if action == "feature":
            feat = payload.get("feature", "")
            if feat not in _CPU_FEATURES:
                return {
                    "error": f"feature {feat!r} not supported on CPU worker; "
                             f"use the GPU endpoint",
                    "worker_mode": "cpu",
                }

    video_url = payload.get("video_url", "")
    sm = _get_sm()

    try:
        s = sm.get_session(session_id)
        if not s:
            if video_url:
                sm.create_session(session_id, video_url)
                s = sm.get_session(session_id)
            else:
                return {"error": f"no session {session_id!r} and no video_url"}

        # 大部分任务都需要本地视频文件：
        #   - GPU: detect_frame / track / analyze 直接读帧
        #   - CPU: ai_summary 用 ffmpeg 切片再上传 Gemini
        # 只有少数 CPU feature 任务（heatmap/charts 等）只读 tracks.pkl
        # 不碰视频。简单起见：除了那些纯 stats 的 feature 之外，全部走下载。
        _STATS_ONLY_FEATURES = {"heatmap", "speed_chart", "possession",
                                "sprint_analysis", "defensive_line"}
        _feat = (payload.get("feature") or "").strip()
        _needs_video = not (action == "feature" and _feat in _STATS_ONLY_FEATURES)
        if _needs_video:
            _ensure_local_video(session_id, video_url or s.get("video_url", ""), sm)
            s = sm.get_session(session_id)

        return fn(session_id, s, payload, sm)
    except Exception as exc:
        log.exception("handler failed")
        return {"error": str(exc)}


# 打印版本 + worker mode：方便从 RunPod 日志确认部署的是哪个 commit。
# 每次 git push 都会改这个常量 → 看到老值就知道 image 没 rebuild。
HANDLER_VERSION = "2026-06-04-bbox-attack-dir"   # bump this on every relevant push

# Print worker mode on import so RunPod logs make it obvious which pool we're on.
print(f"[HANDLER] WORKER_MODE={WORKER_MODE} version={HANDLER_VERSION} "
      f"({'feature-only (ai_summary + charts)' if WORKER_MODE == 'cpu' else 'all actions'})",
      flush=True)


# RunPod Serverless boilerplate (only imports when actually running serverless).
if __name__ == "__main__":
    try:
        import runpod  # type: ignore

        runpod.serverless.start({"handler": handler})
    except ImportError:
        print("runpod SDK not installed; this module is only meant to run inside RunPod Serverless.")
