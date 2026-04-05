/**
 * VideoOverlayPlayer.jsx
 * 原始视频 + Canvas 实时标注叠加
 * 直接读 /api/<sid>/overlay_data，无需等待任何视频生成
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import colab from '../services/colabService';
import './VideoOverlayPlayer.css';

const BALL_COLOR    = '#00FF00';
const TRACKED_OUTER = '#FFD700';

// ── 椭圆：模仿 cv2.ellipse(center, axes, 0, -45, 235) ─────────────────────
function drawEllipse(ctx, bbox, color, trackId, lineWidth = 2) {
    const [x1, y1, x2, y2] = bbox;
    const cx = (x1 + x2) / 2;
    const cy = y2;                          // 底部中心
    const rx = Math.max((x2 - x1) / 2, 1);
    const ry = Math.max(rx * 0.35, 1);

    // OpenCV startAngle=-45, endAngle=235（顺时针）→ Canvas 同等角度（rad）
    const startA = (-45  * Math.PI) / 180;
    const endA   = (235  * Math.PI) / 180;

    ctx.save();
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, startA, endA, false); // false = 顺时针
    ctx.strokeStyle = color;
    ctx.lineWidth   = lineWidth;
    ctx.stroke();

    // ID 标签
    if (trackId !== null && trackId !== undefined) {
        const label = String(trackId);
        ctx.font = 'bold 11px Arial';
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(cx - tw / 2 - 4, cy - 9, tw + 8, 14);
        ctx.fillStyle = '#000';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, cx, cy - 2);
    }
    ctx.restore();
}

// ── 倒三角（球 / 持球标记） ────────────────────────────────────────────────
function drawTriangle(ctx, bbox, color) {
    const [x1, y1, x2, y2] = bbox;
    const cx = (x1 + x2) / 2;
    const cy = y1;
    const s  = 9;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(cx, cy - s);
    ctx.lineTo(cx - s / 2, cy);
    ctx.lineTo(cx + s / 2, cy);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    ctx.restore();
}

// ── 顶部控球率横条 ─────────────────────────────────────────────────────────
function drawPossessionBar(ctx, cw, t1frac, t2frac, t1color, t2color) {
    const bh = 22, bw = 220, bx = (cw - bw) / 2, by = 8;
    ctx.save();
    ctx.globalAlpha = 0.75;
    ctx.fillStyle = '#111';
    ctx.fillRect(bx, by, bw, bh);
    ctx.fillStyle = t1color;
    ctx.fillRect(bx, by, bw * t1frac, bh);
    ctx.fillStyle = t2color;
    ctx.fillRect(bx + bw * t1frac, by, bw * t2frac, bh);
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 11px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(
        `T1 ${Math.round(t1frac * 100)}%  |  T2 ${Math.round(t2frac * 100)}%`,
        cw / 2, by + 11
    );
    ctx.restore();
}

// ── 计算视频在 canvas 上的实际显示区域（letterbox 补偿） ────────────────────
function getVideoRect(canvasW, canvasH, videoW, videoH) {
    const canvasAspect = canvasW / canvasH;
    const videoAspect  = videoW  / videoH;
    let dispW, dispH, offsetX, offsetY;
    if (canvasAspect > videoAspect) {
        dispH   = canvasH;
        dispW   = canvasH * videoAspect;
        offsetX = (canvasW - dispW) / 2;
        offsetY = 0;
    } else {
        dispW   = canvasW;
        dispH   = canvasW / videoAspect;
        offsetX = 0;
        offsetY = (canvasH - dispH) / 2;
    }
    return { dispW, dispH, offsetX, offsetY };
}

export default function VideoOverlayPlayer({ sessionId }) {
    const videoRef   = useRef(null);
    const canvasRef  = useRef(null);
    const overlayRef = useRef(null);
    const rafRef     = useRef(null);
    const possRef    = useRef({ t1: 0, t2: 0, lastFrame: -1 });

    const [loadState, setLoadState] = useState('loading');
    const [errMsg, setErrMsg]       = useState('');

    // ── 加载 overlay 数据 ─────────────────────────────────────────────────
    useEffect(() => {
        if (!sessionId) return;
        setLoadState('loading');
        const base = colab.getUrl();
        fetch(`${base}/api/${sessionId}/overlay_data`)
            .then(r => r.ok ? r.json() : r.json().then(e => { throw new Error(e.error || r.status) }))
            .then(data => { overlayRef.current = data; setLoadState('ready'); })
            .catch(e => { setErrMsg(e.message); setLoadState('error'); });
    }, [sessionId]);

    // ── Canvas 渲染（每 rAF 触发一次） ───────────────────────────────────
    const renderFrame = useCallback(() => {
        const video  = videoRef.current;
        const canvas = canvasRef.current;
        const data   = overlayRef.current;
        if (!video || !canvas || !data) return;

        const ctx = canvas.getContext('2d');
        const cw  = canvas.width;
        const ch  = canvas.height;
        ctx.clearRect(0, 0, cw, ch);

        const frameIdx = Math.min(
            Math.floor(video.currentTime * data.fps),
            data.frames.length - 1
        );
        if (frameIdx < 0) return;

        const f = data.frames[frameIdx];
        if (!f) return;

        // ── 控球率累计（每帧只累计一次）──────────────────────────────────
        const pos = possRef.current;
        if (frameIdx !== pos.lastFrame) {
            pos.lastFrame = frameIdx;
            if (f.ctrl === 1) pos.t1++;
            else if (f.ctrl === 2) pos.t2++;
        }
        const tot = pos.t1 + pos.t2 || 1;

        // ── 计算视频实际显示区域（处理 letterbox） ────────────────────────
        const { dispW, dispH, offsetX, offsetY } =
            getVideoRect(cw, ch, data.video_w, data.video_h);
        const scaleX = dispW / data.video_w;
        const scaleY = dispH / data.video_h;

        // 坐标转换：原始分辨率 → canvas 像素（含 letterbox 偏移）
        const sc = ([ax1, ay1, ax2, ay2]) => [
            ax1 * scaleX + offsetX,
            ay1 * scaleY + offsetY,
            ax2 * scaleX + offsetX,
            ay2 * scaleY + offsetY,
        ];

        // ── 找追踪目标对应的 YOLO player id ──────────────────────────────
        let trackedId = null;
        if (f.t) {
            const [tx1, ty1, tx2, ty2] = f.t;
            const tcx = (tx1 + tx2) / 2, tcy = (ty1 + ty2) / 2;
            let minD = 80; // 原始分辨率像素阈值
            for (const p of f.p) {
                const pcx = (p[1] + p[3]) / 2, pcy = (p[2] + p[4]) / 2;
                const d = Math.hypot(pcx - tcx, pcy - tcy);
                if (d < minD) { minD = d; trackedId = p[0]; }
            }
        }

        // ── 画普通球员 ─────────────────────────────────────────────────
        const t1c = data.t1 || '#00BFFF';
        const t2c = data.t2 || '#FF1493';
        for (const p of f.p) {
            if (p[0] === trackedId) continue;
            const color = p[5] === 1 ? t1c : t2c;
            drawEllipse(ctx, sc([p[1], p[2], p[3], p[4]]), color, p[0]);
            if (p[6]) drawTriangle(ctx, sc([p[1], p[2], p[3], p[4]]), '#0000FF');
        }

        // ── 画追踪目标（金色外圈 + 队伍色内圈） ─────────────────────────
        if (f.t) {
            const tBbox = sc(f.t);
            drawEllipse(ctx, tBbox, TRACKED_OUTER, null, 5);

            const tp = f.p.find(p => p[0] === trackedId);
            const innerColor = tp ? (tp[5] === 1 ? t1c : t2c) : TRACKED_OUTER;
            // 内圈稍微缩小（4px 四周）
            const inner = [tBbox[0]+4, tBbox[1]+4, tBbox[2]-4, tBbox[3]-4];
            drawEllipse(ctx, inner, innerColor, trackedId, 2);

            // 速度标签
            if (tp && tp[7] > 0) {
                const spd = tp[7].toFixed(1) + ' km/h';
                ctx.save();
                ctx.font = 'bold 12px Arial';
                const tw = ctx.measureText(spd).width;
                const lx = tBbox[0], ly = tBbox[3] + 4;
                ctx.fillStyle = 'rgba(180,235,255,0.9)';
                ctx.fillRect(lx, ly, tw + 8, 16);
                ctx.strokeStyle = '#000'; ctx.lineWidth = 0.5;
                ctx.strokeRect(lx, ly, tw + 8, 16);
                ctx.fillStyle = '#000';
                ctx.textAlign = 'left'; ctx.textBaseline = 'top';
                ctx.fillText(spd, lx + 4, ly + 2);
                ctx.restore();
            }
        }

        // ── 球 ─────────────────────────────────────────────────────────
        if (f.b) drawTriangle(ctx, sc(f.b), BALL_COLOR);

        // ── 控球率横条 ─────────────────────────────────────────────────
        drawPossessionBar(ctx, cw, pos.t1 / tot, pos.t2 / tot, t1c, t2c);

    }, []);

    // ── rAF 循环 ─────────────────────────────────────────────────────────
    useEffect(() => {
        if (loadState !== 'ready') return;
        const loop = () => { renderFrame(); rafRef.current = requestAnimationFrame(loop); };
        rafRef.current = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(rafRef.current);
    }, [loadState, renderFrame]);

    // ── 视频 / 窗口尺寸变化时同步 canvas 大小 ────────────────────────────
    useEffect(() => {
        if (loadState !== 'ready') return;
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas) return;
        const sync = () => {
            canvas.width  = video.clientWidth  || video.offsetWidth  || 640;
            canvas.height = video.clientHeight || video.offsetHeight || 360;
        };
        sync();
        const ro = new ResizeObserver(sync);
        ro.observe(video);
        return () => ro.disconnect();
    }, [loadState]);

    const videoSrc = `${colab.getUrl()}/api/${sessionId}/raw_video`;

    return (
        <div className="vop">
            {loadState === 'loading' && (
                <div className="vop__loading">
                    <div className="vop__spinner" />
                    <span>Loading overlay data...</span>
                </div>
            )}
            {loadState === 'error' && (
                <div className="vop__error">❌ {errMsg}</div>
            )}
            {loadState === 'ready' && (
                <div className="vop__wrapper">
                    <video
                        ref={videoRef}
                        src={videoSrc}
                        controls
                        className="vop__video"
                        onSeeked={() => { possRef.current = { t1: 0, t2: 0, lastFrame: -1 }; }}
                    />
                    <canvas ref={canvasRef} className="vop__canvas" />
                </div>
            )}
        </div>
    );
}
