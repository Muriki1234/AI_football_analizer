import { useEffect, useRef, useState } from 'react';

/**
 * Canvas overlay drawn on top of the main video, synced to currentTime.
 * - Loads minimap_positions.json once and indexes by frame on every rAF.
 * - Draggable: click + drag to move to any corner (or anywhere). Position
 *   persists in localStorage per session.
 * - Visual style: plain solid-colour dots. The tracked player gets a thin
 *   purple ring; the ball is a clear white-with-black-outline circle.
 */
const STORAGE_KEY = 'pitchlogic.minimapPos';

export default function MinimapOverlay({ dataUrl, videoRef, visible }) {
    const canvasRef = useRef(null);
    const dragRef = useRef(null);
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

    // ── Position + drag state ────────────────────────────────────────────
    // null pos means "use default bottom-right corner".
    const [pos, setPos] = useState(() => {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            return raw ? JSON.parse(raw) : null;
        } catch { return null; }
    });
    const [dragging, setDragging] = useState(false);

    useEffect(() => {
        if (!dragging) return;
        const onMove = (e) => {
            if (!dragRef.current) return;
            const { startX, startY, posX, posY, parent } = dragRef.current;
            const dx = e.clientX - startX;
            const dy = e.clientY - startY;
            // Clamp inside the parent (video wrapper) bounds
            const newX = Math.max(0, Math.min(parent.width - 270, posX + dx));
            const newY = Math.max(0, Math.min(parent.height - 165, posY + dy));
            setPos({ x: newX, y: newY });
        };
        const onUp = () => {
            setDragging(false);
            // Persist to localStorage
            try {
                if (pos) localStorage.setItem(STORAGE_KEY, JSON.stringify(pos));
            } catch { /* localStorage disabled */ }
        };
        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', onUp);
        return () => {
            window.removeEventListener('mousemove', onMove);
            window.removeEventListener('mouseup', onUp);
        };
    }, [dragging, pos]);

    const handleMouseDown = (e) => {
        e.preventDefault();
        const canvas = canvasRef.current;
        if (!canvas) return;
        const canvasRect = canvas.getBoundingClientRect();
        // Find the parent video wrapper for clamping bounds
        const parent = canvas.parentElement?.getBoundingClientRect()
            || { left: 0, top: 0, width: 9999, height: 9999 };
        dragRef.current = {
            startX: e.clientX,
            startY: e.clientY,
            posX: canvasRect.left - parent.left,
            posY: canvasRect.top - parent.top,
            parent: { width: parent.width, height: parent.height },
        };
        setDragging(true);
    };

    // ── rAF loop: read video.currentTime → draw frame ────────────────────
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

        const PAD = 10;
        const drawW = W - PAD * 2;
        const drawH = H - PAD * 2;

        const toPx = (x, y) => [
            PAD + (x / pitchLen) * drawW,
            PAD + (y / pitchWid) * drawH,
        ];

        const drawPitch = () => {
            ctx.fillStyle = 'rgba(15, 23, 42, 0.92)';
            ctx.fillRect(0, 0, W, H);
            ctx.fillStyle = '#1a472a';
            ctx.fillRect(PAD, PAD, drawW, drawH);
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

            // Players — plain solid-colour dots. Draw the tracked one last
            // so its purple ring lands on top of everybody else.
            let trackedPlayer = null;
            for (const p of frame) {
                if (p.tr) {
                    trackedPlayer = p;
                    continue;
                }
                const [px, py] = toPx(p.x, p.y);
                const color = teamColors[String(p.t)]
                    || (p.t === 1 ? '#3498db' : p.t === 2 ? '#e74c3c' : '#94a3b8');
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(px, py, 3.2, 0, Math.PI * 2);
                ctx.fill();
            }

            // Tracked player: same colored dot, plus a thin purple border
            if (trackedPlayer) {
                const [px, py] = toPx(trackedPlayer.x, trackedPlayer.y);
                const color = teamColors[String(trackedPlayer.t)]
                    || (trackedPlayer.t === 1 ? '#3498db'
                        : trackedPlayer.t === 2 ? '#e74c3c' : '#94a3b8');
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(px, py, 3.6, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = '#a855f7';  // purple ring = the player we're tracking
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(px, py, 6, 0, Math.PI * 2);
                ctx.stroke();
            }

            // Ball — clear white circle with black outline so it stands out
            // against both the pitch green AND the player dots.
            if (ball) {
                const [bx, by] = toPx(ball[0], ball[1]);
                ctx.fillStyle = '#ffffff';
                ctx.strokeStyle = '#000000';
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.arc(bx, by, 3.2, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
            }

            raf = requestAnimationFrame(tick);
        };
        raf = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(raf);
    }, [visible, data, videoRef]);

    if (!visible || !dataUrl) return null;

    // Style: free positioning if user dragged, else default bottom-right
    const style = pos
        ? { left: `${pos.x}px`, top: `${pos.y}px`, right: 'auto', bottom: 'auto' }
        : {};

    return (
        <canvas
            ref={canvasRef}
            className={`minimap-overlay ${dragging ? 'is-dragging' : ''}`}
            style={style}
            onMouseDown={handleMouseDown}
            title="Drag to move"
        />
    );
}
