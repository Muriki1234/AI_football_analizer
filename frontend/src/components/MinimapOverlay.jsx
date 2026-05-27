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

    // Re-clamp the saved position whenever the video wrapper resizes.
    // Without this, dragging the minimap to the bottom-right corner and
    // then shrinking the window leaves the minimap stranded outside the
    // visible area.
    useEffect(() => {
        if (!visible || !pos) return;
        const el = canvasRef.current;
        if (!el?.parentElement) return;
        const parent = el.parentElement;
        const ro = new ResizeObserver(() => {
            const r = parent.getBoundingClientRect();
            const maxX = Math.max(0, r.width - 270);
            const maxY = Math.max(0, r.height - 165);
            if (pos.x > maxX || pos.y > maxY) {
                setPos({
                    x: Math.min(pos.x, maxX),
                    y: Math.min(pos.y, maxY),
                });
            }
        });
        ro.observe(parent);
        return () => ro.disconnect();
    }, [visible, pos]);

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

        // JSON 后端按 5Hz 降采样（每 200ms 一帧），如果按"最近样本"取，
        // 球员位置就会每 200ms 跳一下 = 卡顿感。
        // 改成：找 targetFrame 前后两个样本 + 插值比例 alpha ∈ [0, 1]，
        // 让前端用 60fps 的 rAF 在两个样本之间平滑插值球员位置。
        const stride = data.sample_stride || 1;
        const sampleIndices = data.sample_indices;
        const pickInterp = (targetFrame) => {
            const N = data.frames.length;
            if (!sampleIndices || sampleIndices.length === 0) {
                const idx = Math.min(Math.max(0, targetFrame), N - 1);
                return { a: idx, b: idx, alpha: 0 };
            }
            // 用 stride 估计一个起点，然后线性扫描找到 targetFrame 所在的区间。
            // 因为 sample_indices 单调递增，常数次内必中。
            let i = Math.min(
                Math.max(0, Math.floor(targetFrame / stride)),
                sampleIndices.length - 1,
            );
            // 调整 i 使 sampleIndices[i] <= targetFrame < sampleIndices[i+1]
            while (i > 0 && sampleIndices[i] > targetFrame) i--;
            while (i < sampleIndices.length - 1 && sampleIndices[i + 1] <= targetFrame) i++;
            const a = i;
            const b = Math.min(i + 1, sampleIndices.length - 1);
            const fa = sampleIndices[a];
            const fb = sampleIndices[b];
            const alpha = (fa === fb)
                ? 0
                : Math.max(0, Math.min(1, (targetFrame - fa) / (fb - fa)));
            return { a, b, alpha };
        };

        // ── Period-aware time mapping ──────────────────────────────────────
        // The rendered mp4 has non-match frames stripped out, so its
        // duration ≠ original video duration. video.currentTime maps to an
        // OUTPUT frame, which we need to translate back to the original
        // frame index that sample_indices uses.
        //
        // Algorithm: walk match_periods accumulating their lengths until
        // we've covered enough output frames to reach the requested point.
        const matchPeriods = Array.isArray(data.match_periods) && data.match_periods.length > 0
            ? data.match_periods : null;
        const outputFrameToOriginal = (outFrame) => {
            if (!matchPeriods) return outFrame;
            let acc = 0;
            for (const [ps, pe] of matchPeriods) {
                const periodLen = pe - ps;
                if (outFrame < acc + periodLen) {
                    return ps + (outFrame - acc);
                }
                acc += periodLen;
            }
            // Past the end of all periods — clamp to the last frame
            const last = matchPeriods[matchPeriods.length - 1];
            return last[1] - 1;
        };

        // Helper: 按 id 把帧的球员数组转 Map，方便后面 O(1) 找配对
        const toMap = (arr) => {
            const m = new Map();
            for (const p of arr || []) m.set(p.id, p);
            return m;
        };

        let raf;
        const tick = () => {
            const v = videoRef?.current;
            if (!v) { raf = requestAnimationFrame(tick); return; }
            const fps = data.fps || 25;
            const outputFrame = Math.max(0, Math.floor(v.currentTime * fps));
            const targetFrame = outputFrameToOriginal(outputFrame);

            // 取前后两帧 + 插值比例
            const { a, b, alpha } = pickInterp(targetFrame);
            const frameA = data.frames[a] || [];
            const frameB = data.frames[b] || [];
            const mapB = toMap(frameB);
            const ballA = data.ball?.[a];
            const ballB = data.ball?.[b];
            const teamColors = data.team_colors || {};

            ctx.clearRect(0, 0, W, H);
            drawPitch();

            // ── Players ─────────────────────────────────────────────────
            // 用 frameA 里的球员作为基准（每个 id 在 frameA 都有就插值，
            // 没有就只用 frameA 的位置不插值）。同样把追踪的球员留到最后
            // 画，让紫圈盖在最上层。
            let trackedPa = null, trackedPb = null;
            for (const pa of frameA) {
                if (pa.tr) {
                    trackedPa = pa;
                    trackedPb = mapB.get(pa.id) || null;
                    continue;
                }
                const pb = mapB.get(pa.id);
                // 球员在 b 帧消失：原地停住而不是闪一下，更自然
                const x = pb ? pa.x + (pb.x - pa.x) * alpha : pa.x;
                const y = pb ? pa.y + (pb.y - pa.y) * alpha : pa.y;
                const [px, py] = toPx(x, y);
                const color = teamColors[String(pa.t)]
                    || (pa.t === 1 ? '#3498db' : pa.t === 2 ? '#e74c3c' : '#94a3b8');
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(px, py, 3.2, 0, Math.PI * 2);
                ctx.fill();
            }

            // Tracked player + 紫色追踪环
            if (trackedPa) {
                const pb = trackedPb;
                const x = pb ? trackedPa.x + (pb.x - trackedPa.x) * alpha : trackedPa.x;
                const y = pb ? trackedPa.y + (pb.y - trackedPa.y) * alpha : trackedPa.y;
                const [px, py] = toPx(x, y);
                const color = teamColors[String(trackedPa.t)]
                    || (trackedPa.t === 1 ? '#3498db'
                        : trackedPa.t === 2 ? '#e74c3c' : '#94a3b8');
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(px, py, 3.6, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = '#a855f7';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(px, py, 6, 0, Math.PI * 2);
                ctx.stroke();
            }

            // ── Ball ────────────────────────────────────────────────────
            // 同样插值。如果 b 帧没有球（识别丢失），就用 a 帧的位置
            if (ballA) {
                const x = ballB ? ballA[0] + (ballB[0] - ballA[0]) * alpha : ballA[0];
                const y = ballB ? ballA[1] + (ballB[1] - ballA[1]) * alpha : ballA[1];
                const [bx, by] = toPx(x, y);
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
