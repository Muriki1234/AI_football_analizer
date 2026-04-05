/**
 * VideoOverlayPlayer.jsx
 * 原始视频 + Canvas 实时标注叠加
 * 直接读 /api/<sid>/overlay_data，无需等待任何视频生成
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import colab from '../services/colabService';
import './VideoOverlayPlayer.css';

const T1_COLOR = '#00BFFF';
const T2_COLOR = '#FF1493';
const BALL_COLOR = '#00FF00';
const TRACKED_OUTER = '#FFD700';

function hexToRgba(hex, alpha = 1) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}

/** 在 bbox 底部中心画椭圆（模仿 cv2.ellipse 效果） */
function drawEllipse(ctx, bbox, color, trackId, isTracked, lineWidth = 2) {
    const [x1, y1, x2, y2] = bbox;
    const cx = (x1 + x2) / 2;
    const cy = y2;
    const rx = Math.max((x2 - x1) / 2, 1);
    const ry = Math.max(rx * 0.35, 1);

    // 只画下半段（-45° ~ 235°）
    const startA = (225 * Math.PI) / 180;
    const endA   = (315 * Math.PI) / 180;

    ctx.save();
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, startA, endA, false);
    ctx.strokeStyle = color;
    ctx.lineWidth   = lineWidth;
    ctx.stroke();

    // ID 标签
    if (trackId !== null && trackId !== undefined) {
        const label = String(trackId);
        ctx.font = 'bold 11px Arial';
        const tw = ctx.measureText(label).width;
        const bx = cx - tw / 2 - 4;
        const by = cy - 8;
        ctx.fillStyle = color;
        ctx.fillRect(bx, by, tw + 8, 14);
        ctx.fillStyle = '#000';
        ctx.textAlign = 'center';
        ctx.fillText(label, cx, cy + 2);
    }
    ctx.restore();
}

/** 球 — 倒三角 */
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

/** 顶部控球率横条 */
function drawPossessionBar(ctx, cw, ctrl, t1frac, t2frac, t1color, t2color) {
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
    ctx.fillText(`T1 ${Math.round(t1frac * 100)}%  |  T2 ${Math.round(t2frac * 100)}%`, cw / 2, by + 15);
    ctx.restore();
}

export default function VideoOverlayPlayer({ sessionId }) {
    const videoRef  = useRef(null);
    const canvasRef = useRef(null);
    const overlayRef = useRef(null);   // 存放 overlay 数据
    const rafRef    = useRef(null);
    const possRef   = useRef({ t1: 0, t2: 0 });  // 累计控球帧

    const [loadState, setLoadState] = useState('loading'); // loading | ready | error
    const [errMsg, setErrMsg]       = useState('');

    // 加载 overlay 数据
    useEffect(() => {
        if (!sessionId) return;
        setLoadState('loading');
        const base = colab.getUrl();
        fetch(`${base}/api/${sessionId}/overlay_data`)
            .then(r => r.ok ? r.json() : r.json().then(e => { throw new Error(e.error || r.status) }))
            .then(data => {
                overlayRef.current = data;
                setLoadState('ready');
            })
            .catch(e => {
                setErrMsg(e.message);
                setLoadState('error');
            });
    }, [sessionId]);

    // Canvas 渲染循环
    const renderFrame = useCallback(() => {
        const video  = videoRef.current;
        const canvas = canvasRef.current;
        const data   = overlayRef.current;
        if (!video || !canvas || !data) return;

        const ctx = canvas.getContext('2d');
        const cw  = canvas.width;
        const ch  = canvas.height;
        ctx.clearRect(0, 0, cw, ch);

        // 缩放比（视频实际渲染尺寸 vs 原始分辨率）
        const scaleX = cw / data.video_w;
        const scaleY = ch / data.video_h;

        const frameIdx = Math.min(
            Math.floor(video.currentTime * data.fps),
            data.frames.length - 1
        );
        if (frameIdx < 0) return;

        const f = data.frames[frameIdx];
        if (!f) return;

        // 累计控球
        const pos = possRef.current;
        if (f.ctrl === 1) pos.t1++;
        else if (f.ctrl === 2) pos.t2++;
        const tot = pos.t1 + pos.t2 || 1;

        // 找追踪目标对应的 player id
        let trackedId = null;
        if (f.t) {
            const [tx1, ty1, tx2, ty2] = f.t;
            const tcx = (tx1 + tx2) / 2, tcy = (ty1 + ty2) / 2;
            let minD = 100 / scaleX; // 像素阈值（原始分辨率）
            for (const p of f.p) {
                const pcx = (p[1] + p[3]) / 2, pcy = (p[2] + p[4]) / 2;
                const d = Math.hypot(pcx - tcx, pcy - tcy);
                if (d < minD) { minD = d; trackedId = p[0]; }
            }
        }

        // 坐标转换 helper
        const sc = ([x1, y1, x2, y2]) => [x1 * scaleX, y1 * scaleY, x2 * scaleX, y2 * scaleY];

        // 画普通球员
        for (const p of f.p) {
            if (p[0] === trackedId) continue;
            const color = p[5] === 1 ? (data.t1 || T1_COLOR) : (data.t2 || T2_COLOR);
            drawEllipse(ctx, sc([p[1], p[2], p[3], p[4]]), color, p[0], false);
            // 持球三角
            if (p[6]) drawTriangle(ctx, sc([p[1], p[2], p[3], p[4]]), '#0000FF');
        }

        // 画追踪目标
        if (f.t && trackedId !== null) {
            const tBbox = sc(f.t);
            // 外圈金色（厚）
            drawEllipse(ctx, tBbox, TRACKED_OUTER, null, true, 5);
            // 内圈队伍色（薄）
            const tp = f.p.find(p => p[0] === trackedId);
            const innerColor = tp ? (tp[5] === 1 ? (data.t1 || T1_COLOR) : (data.t2 || T2_COLOR)) : TRACKED_OUTER;
            const inner = tBbox.map((v, i) => i < 2 ? v + 4 : v - 4); // 略微缩小
            drawEllipse(ctx, inner, innerColor, trackedId, true, 2);
            // 速度标签
            if (tp && tp[7] > 0) {
                const spd = tp[7].toFixed(1) + ' km/h';
                ctx.save();
                ctx.font = 'bold 12px Arial';
                const tw = ctx.measureText(spd).width;
                ctx.fillStyle = 'rgba(180,235,255,0.9)';
                ctx.fillRect(tBbox[0], tBbox[3] + 4, tw + 8, 16);
                ctx.strokeStyle = '#000';
                ctx.lineWidth = 0.5;
                ctx.strokeRect(tBbox[0], tBbox[3] + 4, tw + 8, 16);
                ctx.fillStyle = '#000';
                ctx.textAlign = 'left';
                ctx.fillText(spd, tBbox[0] + 4, tBbox[3] + 16);
                ctx.restore();
            }
        } else if (f.t) {
            // SAMURAI bbox 存在但没匹配到 YOLO player
            drawEllipse(ctx, sc(f.t), TRACKED_OUTER, null, true, 5);
        }

        // 球
        if (f.b) drawTriangle(ctx, sc(f.b), BALL_COLOR);

        // 控球率横条
        drawPossessionBar(ctx, cw, f.ctrl, pos.t1 / tot, pos.t2 / tot, data.t1 || T1_COLOR, data.t2 || T2_COLOR);

    }, []);

    // rAF 循环
    useEffect(() => {
        if (loadState !== 'ready') return;
        const loop = () => {
            renderFrame();
            rafRef.current = requestAnimationFrame(loop);
        };
        rafRef.current = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(rafRef.current);
    }, [loadState, renderFrame]);

    // 视频尺寸变化时同步 canvas
    useEffect(() => {
        if (loadState !== 'ready') return;
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas) return;
        const sync = () => {
            canvas.width  = video.clientWidth;
            canvas.height = video.clientHeight;
        };
        sync();
        const ro = new ResizeObserver(sync);
        ro.observe(video);
        return () => ro.disconnect();
    }, [loadState]);

    const base = colab.getUrl();
    const videoSrc = `${base}/api/${sessionId}/raw_video`;

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
                        onSeeked={() => { possRef.current = { t1: 0, t2: 0 }; }}
                    />
                    <canvas ref={canvasRef} className="vop__canvas" />
                </div>
            )}
        </div>
    );
}
