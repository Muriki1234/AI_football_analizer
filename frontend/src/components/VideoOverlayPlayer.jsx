/**
 * VideoOverlayPlayer.jsx
 * 完全对齐 football_project.ipynb 的视觉效果
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import colab from '../services/colabService';
import './VideoOverlayPlayer.css';

const TRACKED_GOLD = '#00D7FF';  // OpenCV (0,215,255) → #00D7FF
const BALL_CYAN = '#00FFFF';

// ─────────────────────────────────────────────────────────────────────────────
// 基础绘图工具
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 椭圆弧 - 完全对齐 cv2.ellipse(center=(cx,y2), axes=(w,0.35w), 0, -45, 235)
 * notebook: thickness=2 正常 / thickness=4 追踪
 */
function drawEllipse(ctx, bbox, color, lineWidth = 2) {
    const [x1, , x2, y2] = bbox;
    const cx = (x1 + x2) / 2;
    const rx = Math.max((x2 - x1) / 2, 1);
    const ry = Math.max(rx * 0.35, 1);
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(cx, y2, rx, ry, 0, (-45 * Math.PI) / 180, (235 * Math.PI) / 180, false);
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
    ctx.restore();
}

/**
 * ID 标签框 - 对齐 notebook：背景色=color，黑色文字
 * 追踪目标：背景=#00D7FF，后面加 ★
 */
function drawIdLabel(ctx, bbox, color, trackId, isTracked) {
    if (trackId === null || trackId === undefined) return;
    const [x1, , x2, y2] = bbox;
    const cx = (x1 + x2) / 2;
    const label = String(trackId);
    const star  = isTracked ? ' ★' : '';

    ctx.save();
    ctx.font = 'bold 12px Arial';
    const tw = ctx.measureText(label + star).width;
    const bx = cx - tw / 2 - 5;
    const by = y2 - 10;
    const bh = 20;

    // 背景框（追踪=金色，普通=队伍色）
    ctx.fillStyle = isTracked ? TRACKED_GOLD : color;
    ctx.fillRect(bx, by, tw + 10, bh);

    // 文字
    ctx.fillStyle = '#000';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label + star, cx, by + bh / 2);
    ctx.restore();
}

/**
 * 三角形 - 对齐 notebook draw_triangle:
 *   pts = [[x, y-10], [x-10, y-30], [x+10, y-30]]  ← 上方朝上三角，基在上顶在下
 *   填充 color，黑色描边 thickness=2
 */
function drawTriangle(ctx, bbox, color) {
    const [x1, y1, x2] = bbox;
    const cx = (x1 + x2) / 2;
    const tipY  = y1 - 10;   // 下顶点
    const baseY = y1 - 30;   // 上底边
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(cx,      tipY);
    ctx.lineTo(cx - 10, baseY);
    ctx.lineTo(cx + 10, baseY);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();
}

function drawBallBox(ctx, bbox, conf) {
    const [x1, y1, x2, y2] = bbox;
    const label = conf !== null && conf !== undefined ? `Ball ${conf.toFixed(2)}` : 'Ball';

    ctx.save();
    ctx.strokeStyle = BALL_CYAN;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, Math.max(x2 - x1, 1), Math.max(y2 - y1, 1));

    ctx.font = 'bold 12px Arial';
    const tw = ctx.measureText(label).width;
    const bh = 18;
    const by = Math.max(0, y1 - bh - 4);
    ctx.fillStyle = BALL_CYAN;
    ctx.fillRect(x1, by, tw + 8, bh);
    ctx.fillStyle = '#000';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(label, x1 + 4, by + 3);
    ctx.restore();
}

/**
 * 控球率面板 - 对齐 notebook draw_team_ball_control:
 *   右下角，半透明白色背景，队伍色小方块，百分比文字
 *   响应式：位置根据 canvas 尺寸计算
 */
function drawPossessionPanel(ctx, cw, ch, t1count, t2count, t1color, t2color) {
    const total = t1count + t2count || 1;
    const t1pct = (t1count / total * 100).toFixed(1);
    const t2pct = (t2count / total * 100).toFixed(1);

    const pw = 260, ph = 64;
    const px = cw - pw - 10;
    const py = ch - ph - 10;

    // 半透明白色背景（alpha=0.6 对齐 notebook addWeighted 0.6）
    ctx.save();
    ctx.globalAlpha = 0.82;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(px, py, pw, ph);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = '#323232';
    ctx.lineWidth = 2;
    ctx.strokeRect(px, py, pw, ph);

    // 标题
    ctx.fillStyle = '#000';
    ctx.font = 'bold 13px Arial';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('Ball Control', px + 8, py + 6);

    // Team 1：颜色方块 + 百分比
    ctx.fillStyle = t1color;
    ctx.fillRect(px + 8, py + 28, 18, 18);
    ctx.fillStyle = '#000';
    ctx.font = '12px Arial';
    ctx.fillText(`Team 1:  ${t1pct}%`, px + 32, py + 30);

    // Team 2：颜色方块 + 百分比
    ctx.fillStyle = t2color;
    ctx.fillRect(px + 138, py + 28, 18, 18);
    ctx.fillStyle = '#000';
    ctx.fillText(`Team 2:  ${t2pct}%`, px + 162, py + 30);

    ctx.restore();
}

/**
 * 追踪目标：金色外圈(thickness=6) + 队伍色内圈(thickness=3)
 * 对齐 notebook 的双重 cv2.ellipse 调用
 */
function drawTrackedEllipse(ctx, bbox, teamColor) {
    const [x1, y1, x2, y2] = bbox;
    const cx  = (x1 + x2) / 2;
    const rx  = Math.max((x2 - x1) / 2, 1);
    const ry  = Math.max(rx * 0.35, 1);
    const ang = [(-45 * Math.PI) / 180, (235 * Math.PI) / 180];

    // 外圈：金色，thickness=6
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(cx, y2, rx, ry, 0, ...ang, false);
    ctx.strokeStyle = TRACKED_GOLD;
    ctx.lineWidth = 6;
    ctx.stroke();

    // 内圈：队伍色，axes=(0.9w, 0.315w)，thickness=3
    ctx.beginPath();
    ctx.ellipse(cx, y2, rx * 0.9, ry * 0.9, 0, ...ang, false);
    ctx.strokeStyle = teamColor;
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.restore();
}

/**
 * 速度标签 - 对齐 notebook:
 *   位置：bbox 底部 +32px
 *   背景：(180,235,255) → #B4EBFF，黑色边框，黑色文字
 */
function drawSpeedLabel(ctx, bbox, speedKmh) {
    if (speedKmh <= 0) return;
    const text = speedKmh.toFixed(1) + ' km/h';
    const [x1, , , y2] = bbox;
    ctx.save();
    ctx.font = 'bold 12px Arial';
    const tw = ctx.measureText(text).width;
    const lx = x1, ly = y2 + 20;
    ctx.fillStyle = '#B4EBFF';
    ctx.fillRect(lx, ly, tw + 6, 16);
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    ctx.strokeRect(lx, ly, tw + 6, 16);
    ctx.fillStyle = '#000';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(text, lx + 3, ly + 2);
    ctx.restore();
}

// ─────────────────────────────────────────────────────────────────────────────
// letterbox 补偿
// ─────────────────────────────────────────────────────────────────────────────
function getVideoRect(cw, ch, vw, vh) {
    const ca = cw / ch, va = vw / vh;
    if (ca > va) {
        const dh = ch, dw = ch * va;
        return { dw, dh, ox: (cw - dw) / 2, oy: 0 };
    }
    const dw = cw, dh = cw / va;
    return { dw, dh, ox: 0, oy: (ch - dh) / 2 };
}

// ─────────────────────────────────────────────────────────────────────────────
// 组件
// ─────────────────────────────────────────────────────────────────────────────
export default function VideoOverlayPlayer({ sessionId }) {
    const videoRef   = useRef(null);
    const canvasRef  = useRef(null);
    const overlayRef = useRef(null);
    const rafRef     = useRef(null);
    // 累计控球帧数（对齐 notebook 的 cumulative team_ball_control）
    const possRef    = useRef({ t1: 0, t2: 0, lastFrame: -1 });

    const [loadState, setLoadState] = useState('loading');
    const [errMsg, setErrMsg]       = useState('');

    // 加载 overlay 数据
    useEffect(() => {
        if (!sessionId) return;
        setLoadState('loading');
        const base = colab.getUrl();
        fetch(`${base}/api/${sessionId}/overlay_data`)
            .then(r => r.ok ? r.json() : r.json().then(e => { throw new Error(e.error || r.status) }))
            .then(data => { overlayRef.current = data; setLoadState('ready'); })
            .catch(e => { setErrMsg(e.message); setLoadState('error'); });
    }, [sessionId]);

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

        // 累计控球（每帧只算一次）
        const pos = possRef.current;
        if (frameIdx !== pos.lastFrame) {
            pos.lastFrame = frameIdx;
            if (f.ctrl === 1) pos.t1++;
            else if (f.ctrl === 2) pos.t2++;
        }

        // letterbox 坐标映射
        const { dw, dh, ox, oy } = getVideoRect(cw, ch, data.video_w, data.video_h);
        const sx = dw / data.video_w, sy = dh / data.video_h;
        const sc = ([ax1, ay1, ax2, ay2]) => [
            ax1 * sx + ox, ay1 * sy + oy,
            ax2 * sx + ox, ay2 * sy + oy,
        ];

        // 找追踪目标 YOLO player_id
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

        const t1c = data.t1 || '#00BFFF';
        const t2c = data.t2 || '#FF1493';

        // ── 普通球员（排除追踪目标） ──────────────────────────────────
        for (const p of f.p) {
            if (p[0] === trackedId) continue;
            const bbox  = sc([p[1], p[2], p[3], p[4]]);
            const color = p[5] === 1 ? t1c : t2c;

            drawEllipse(ctx, bbox, color, 2);              // 椭圆弧 thickness=2
            drawIdLabel(ctx, bbox, color, p[0], false);    // ID 标签（队伍色背景）

            // 持球三角（红色，置信度>0.6）
            if (p[6]) drawTriangle(ctx, bbox, '#0000FF');
        }

        // ── 追踪目标 ────────────────────────────────────────────────
        if (f.t) {
            const tBbox  = sc(f.t);
            const tp     = f.p.find(p => p[0] === trackedId);
            const tcolor = tp ? (tp[5] === 1 ? t1c : t2c) : TRACKED_GOLD;
            const speed  = tp ? (tp[7] ?? 0) : 0;

            drawTrackedEllipse(ctx, tBbox, tcolor);             // 双圈
            drawIdLabel(ctx, tBbox, TRACKED_GOLD, trackedId, true); // 金色 ID + ★
            drawSpeedLabel(ctx, tBbox, speed);                   // 速度标签

            // 持球三角
            if (tp?.at(6)) drawTriangle(ctx, tBbox, '#0000FF');
        }

        // ── 球：青色 bbox + 置信度标签 ───────────────────────────────
        if (f.b) {
            const bbox = sc([f.b[0], f.b[1], f.b[2], f.b[3]]);
            drawBallBox(ctx, bbox, f.b[4]);
        }

        // ── 控球率面板（右下角，累计） ───────────────────────────────
        drawPossessionPanel(ctx, cw, ch, pos.t1, pos.t2, t1c, t2c);

    }, []);

    // rAF 循环
    useEffect(() => {
        if (loadState !== 'ready') return;
        const loop = () => { renderFrame(); rafRef.current = requestAnimationFrame(loop); };
        rafRef.current = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(rafRef.current);
    }, [loadState, renderFrame]);

    // 同步 canvas 尺寸
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
                        onSeeked={() => {
                            possRef.current = { t1: 0, t2: 0, lastFrame: -1 };
                        }}
                    />
                    <canvas ref={canvasRef} className="vop__canvas" />
                </div>
            )}
        </div>
    );
}
