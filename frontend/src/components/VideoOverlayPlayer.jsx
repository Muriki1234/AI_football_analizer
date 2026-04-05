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
const POSS_WINDOW   = 120;  // 控球率：滚动最近 N 帧

// ── 椭圆：模仿 cv2.ellipse(center=(cx,y2), axes, 0, -45, 235) ────────────
function drawEllipse(ctx, bbox, color, lineWidth = 3) {
    const [x1, y1, x2, y2] = bbox;
    const cx = (x1 + x2) / 2;
    const cy = y2;
    const rx = Math.max((x2 - x1) / 2, 1);
    const ry = Math.max(rx * 0.35, 1);

    ctx.save();
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0,
        (-45  * Math.PI) / 180,
        (235  * Math.PI) / 180,
        false   // clockwise（与 OpenCV 一致）
    );
    ctx.strokeStyle = color;
    ctx.lineWidth   = lineWidth;
    ctx.stroke();
    ctx.restore();
}

// ── 追踪目标专用：金色外圈 + 队伍色内圈 + 速度标签 ──────────────────────
function drawTrackedPlayer(ctx, outerBbox, innerBbox, teamColor, trackId, speedKmh) {
    drawEllipse(ctx, outerBbox, TRACKED_OUTER, 5);
    drawEllipse(ctx, innerBbox, teamColor, 3);

    // ID 标签（只追踪目标显示）
    if (trackId !== null && trackId !== undefined) {
        const [x1, , x2, y2] = outerBbox;
        const cx = (x1 + x2) / 2;
        const label = String(trackId);
        ctx.save();
        ctx.font = 'bold 11px Arial';
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = TRACKED_OUTER;
        ctx.fillRect(cx - tw / 2 - 4, y2 - 9, tw + 8, 14);
        ctx.fillStyle = '#000';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, cx, y2 - 2);
        ctx.restore();
    }

    // 速度标签
    if (speedKmh > 0) {
        const [x1, , , y2] = outerBbox;
        const spd = speedKmh.toFixed(1) + ' km/h';
        ctx.save();
        ctx.font = 'bold 12px Arial';
        const tw = ctx.measureText(spd).width;
        ctx.fillStyle = 'rgba(180,235,255,0.9)';
        ctx.fillRect(x1, y2 + 4, tw + 8, 16);
        ctx.strokeStyle = '#000'; ctx.lineWidth = 0.5;
        ctx.strokeRect(x1, y2 + 4, tw + 8, 16);
        ctx.fillStyle = '#000';
        ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText(spd, x1 + 4, y2 + 6);
        ctx.restore();
    }
}

// ── 倒三角（球 / 持球标记） ───────────────────────────────────────────────
function drawTriangle(ctx, bbox, color) {
    const [x1, y1, x2] = bbox;
    const cx = (x1 + x2) / 2;
    const s  = 9;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(cx, y1 - s);
    ctx.lineTo(cx - s / 2, y1);
    ctx.lineTo(cx + s / 2, y1);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    ctx.restore();
}

// ── 控球率横条（滚动窗口） ─────────────────────────────────────────────────
function drawPossessionBar(ctx, cw, t1frac, t2frac, t1color, t2color) {
    const bh = 22, bw = 220, bx = (cw - bw) / 2, by = 8;
    ctx.save();
    ctx.globalAlpha = 0.8;
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

// ── 计算 letterbox 显示区域 ────────────────────────────────────────────────
function getVideoRect(cw, ch, vw, vh) {
    const ca = cw / ch, va = vw / vh;
    let dispW, dispH, ox, oy;
    if (ca > va) {
        dispH = ch; dispW = ch * va; ox = (cw - dispW) / 2; oy = 0;
    } else {
        dispW = cw; dispH = cw / va; ox = 0; oy = (ch - dispH) / 2;
    }
    return { dispW, dispH, ox, oy };
}

export default function VideoOverlayPlayer({ sessionId }) {
    const videoRef    = useRef(null);
    const canvasRef   = useRef(null);
    const overlayRef  = useRef(null);
    const rafRef      = useRef(null);
    const historyRef  = useRef([]);   // 最近 POSS_WINDOW 帧的 ctrl 值

    const [loadState, setLoadState] = useState('loading');
    const [errMsg, setErrMsg]       = useState('');

    // ── 加载 overlay 数据 ──────────────────────────────────────────────
    useEffect(() => {
        if (!sessionId) return;
        setLoadState('loading');
        const base = colab.getUrl();
        fetch(`${base}/api/${sessionId}/overlay_data`)
            .then(r => r.ok ? r.json() : r.json().then(e => { throw new Error(e.error || r.status) }))
            .then(data => { overlayRef.current = data; setLoadState('ready'); })
            .catch(e => { setErrMsg(e.message); setLoadState('error'); });
    }, [sessionId]);

    // ── 渲染单帧 ──────────────────────────────────────────────────────
    const renderFrame = useCallback(() => {
        const video  = videoRef.current;
        const canvas = canvasRef.current;
        const data   = overlayRef.current;
        if (!video || !canvas || !data) return;

        const ctx = canvas.getContext('2d');
        const cw = canvas.width, ch = canvas.height;
        ctx.clearRect(0, 0, cw, ch);

        const frameIdx = Math.min(
            Math.floor(video.currentTime * data.fps),
            data.frames.length - 1
        );
        if (frameIdx < 0) return;

        const f = data.frames[frameIdx];
        if (!f) return;

        // ── 滚动窗口控球率 ─────────────────────────────────────────────
        const hist = historyRef.current;
        hist.push(f.ctrl);
        if (hist.length > POSS_WINDOW) hist.shift();
        const t1c_count = hist.filter(v => v === 1).length;
        const t2c_count = hist.filter(v => v === 2).length;
        const tot = t1c_count + t2c_count || 1;

        // ── letterbox 坐标映射 ─────────────────────────────────────────
        const { dispW, dispH, ox, oy } = getVideoRect(cw, ch, data.video_w, data.video_h);
        const sx = dispW / data.video_w, sy = dispH / data.video_h;
        const sc = ([ax1, ay1, ax2, ay2]) => [
            ax1 * sx + ox, ay1 * sy + oy,
            ax2 * sx + ox, ay2 * sy + oy,
        ];

        // ── 找追踪目标 player_id ───────────────────────────────────────
        let trackedId = null;
        if (f.t) {
            const [tx1, ty1, tx2, ty2] = f.t;
            const tcx = (tx1 + tx2) / 2, tcy = (ty1 + ty2) / 2;
            let minD = 80;
            for (const p of f.p) {
                const d = Math.hypot((p[1]+p[3])/2 - tcx, (p[2]+p[4])/2 - tcy);
                if (d < minD) { minD = d; trackedId = p[0]; }
            }
        }

        const t1color = data.t1 || '#00BFFF';
        const t2color = data.t2 || '#FF1493';

        // ── 普通球员：只画彩色椭圆弧，不显示 ID ──────────────────────
        for (const p of f.p) {
            if (p[0] === trackedId) continue;
            const color = p[5] === 1 ? t1color : t2color;
            drawEllipse(ctx, sc([p[1], p[2], p[3], p[4]]), color);
            if (p[6]) drawTriangle(ctx, sc([p[1], p[2], p[3], p[4]]), '#0000FF');
        }

        // ── 追踪目标（金色外圈 + 队伍色内圈 + 速度 + ID） ──────────────
        if (f.t) {
            const tBbox  = sc(f.t);
            const inner  = [tBbox[0]+4, tBbox[1]+4, tBbox[2]-4, tBbox[3]-4];
            const tp     = f.p.find(p => p[0] === trackedId);
            const tcolor = tp ? (tp[5] === 1 ? t1color : t2color) : TRACKED_OUTER;
            const speed  = tp?.at(7) ?? 0;
            drawTrackedPlayer(ctx, tBbox, inner, tcolor, trackedId, speed);
        }

        // ── 球 ─────────────────────────────────────────────────────────
        if (f.b) drawTriangle(ctx, sc(f.b), BALL_COLOR);

        // ── 控球率横条 ─────────────────────────────────────────────────
        drawPossessionBar(ctx, cw, t1c_count / tot, t2c_count / tot, t1color, t2color);

    }, []);

    // ── rAF 渲染循环 ──────────────────────────────────────────────────
    useEffect(() => {
        if (loadState !== 'ready') return;
        const loop = () => { renderFrame(); rafRef.current = requestAnimationFrame(loop); };
        rafRef.current = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(rafRef.current);
    }, [loadState, renderFrame]);

    // ── Canvas 尺寸跟随视频 ───────────────────────────────────────────
    useEffect(() => {
        if (loadState !== 'ready') return;
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas) return;
        const sync = () => {
            canvas.width  = video.clientWidth  || 640;
            canvas.height = video.clientHeight || 360;
        };
        sync();
        const ro = new ResizeObserver(sync);
        ro.observe(video);
        return () => ro.disconnect();
    }, [loadState]);

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
                        src={`${colab.getUrl()}/api/${sessionId}/raw_video`}
                        controls
                        className="vop__video"
                        onSeeked={() => { historyRef.current = []; }}
                    />
                    <canvas ref={canvasRef} className="vop__canvas" />
                </div>
            )}
        </div>
    );
}
