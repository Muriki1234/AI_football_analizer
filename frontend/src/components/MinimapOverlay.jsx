import { useEffect, useRef, useState } from 'react';

/**
 * Canvas overlay drawn on top of the main video, synced to currentTime.
 * Loads minimap_positions.json once and indexes by frame on every rAF.
 *
 * Coordinate system: pitch is `length × width` in cm. We render to a fixed
 * 240×140 px box (bottom-right of the video).
 */
export default function MinimapOverlay({ dataUrl, videoRef, visible }) {
    const canvasRef = useRef(null);
    const [data, setData] = useState(null);

    // Load JSON once
    useEffect(() => {
        if (!dataUrl) return;
        let cancelled = false;
        fetch(dataUrl)
            .then((r) => r.json())
            .then((d) => { if (!cancelled) setData(d); })
            .catch(() => { });
        return () => { cancelled = true; };
    }, [dataUrl]);

    // rAF loop: read video.currentTime → draw frame
    useEffect(() => {
        if (!visible || !data || !canvasRef.current) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;

        const W = 260, H = 156;
        canvas.width = W * dpr;
        canvas.height = H * dpr;
        canvas.style.width = `${W}px`;
        canvas.style.height = `${H}px`;
        ctx.scale(dpr, dpr);

        const pitchLen = data.pitch?.length || 12000;
        const pitchWid = data.pitch?.width || 7000;

        // Margins inside the canvas
        const PAD = 10;
        const drawW = W - PAD * 2;
        const drawH = H - PAD * 2;

        const toPx = (x, y) => [
            PAD + (x / pitchLen) * drawW,
            PAD + (y / pitchWid) * drawH,
        ];

        const drawPitch = () => {
            // Background
            ctx.fillStyle = 'rgba(15, 23, 42, 0.92)';
            ctx.fillRect(0, 0, W, H);
            // Pitch fill
            ctx.fillStyle = '#1a472a';
            ctx.fillRect(PAD, PAD, drawW, drawH);
            // Outline + halfway + center circle
            ctx.strokeStyle = 'rgba(255,255,255,0.55)';
            ctx.lineWidth = 1.2;
            ctx.strokeRect(PAD, PAD, drawW, drawH);
            ctx.beginPath();
            ctx.moveTo(PAD + drawW / 2, PAD);
            ctx.lineTo(PAD + drawW / 2, PAD + drawH);
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(PAD + drawW / 2, PAD + drawH / 2, drawH * 0.13, 0, Math.PI * 2);
            ctx.stroke();
            // Penalty boxes (rough proportions, ~16.5m / 70m wide pitch)
            const pbW = drawW * (2015 / pitchLen);
            const pbH = drawH * (4100 / pitchWid);
            const pbY = PAD + (drawH - pbH) / 2;
            ctx.strokeRect(PAD, pbY, pbW, pbH);
            ctx.strokeRect(PAD + drawW - pbW, pbY, pbW, pbH);
        };

        let raf;
        const tick = () => {
            const v = videoRef?.current;
            if (!v) { raf = requestAnimationFrame(tick); return; }
            const fps = data.fps || 25;
            const frameIdx = Math.min(
                Math.max(0, Math.floor(v.currentTime * fps)),
                (data.total_frames || data.frames.length) - 1,
            );
            const frame = data.frames[frameIdx] || [];
            const ball = data.ball?.[frameIdx];
            const teamColors = data.team_colors || {};

            ctx.clearRect(0, 0, W, H);
            drawPitch();

            // Players
            for (const p of frame) {
                const [px, py] = toPx(p.x, p.y);
                const color = teamColors[String(p.t)] || (p.t === 1 ? '#3498db' : p.t === 2 ? '#e74c3c' : '#94a3b8');
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(px, py, p.tr ? 5 : 3.2, 0, Math.PI * 2);
                ctx.fill();
                if (p.tr) {
                    // Tracked player ring (magenta/cyan to match the video marker)
                    ctx.strokeStyle = '#ff00ff';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.arc(px, py, 7, 0, Math.PI * 2);
                    ctx.stroke();
                }
                if (p.b) {
                    // Has-ball halo
                    ctx.strokeStyle = '#fde047';
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    ctx.arc(px, py, 8, 0, Math.PI * 2);
                    ctx.stroke();
                }
            }

            // Ball
            if (ball) {
                const [bx, by] = toPx(ball[0], ball[1]);
                ctx.fillStyle = '#fde047';
                ctx.beginPath();
                ctx.arc(bx, by, 2.6, 0, Math.PI * 2);
                ctx.fill();
            }

            raf = requestAnimationFrame(tick);
        };
        raf = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(raf);
    }, [visible, data, videoRef]);

    if (!visible || !dataUrl) return null;
    return <canvas ref={canvasRef} className="minimap-overlay" />;
}
