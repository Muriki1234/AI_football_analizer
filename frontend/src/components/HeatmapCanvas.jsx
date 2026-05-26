import { useEffect, useRef, useState } from 'react';

/**
 * Canvas heatmap of the tracked player's pitch positions.
 *
 * Uses an offscreen "intensity" canvas where each position bumps an alpha
 * channel via radial gradients, then maps the intensity to a colour ramp.
 * Pure browser, no external lib.
 *
 * Width is taken from the parent container so it never overflows the
 * (narrow) drawer. Height is derived from the pitch aspect ratio.
 */
export default function HeatmapCanvas({ dataUrl }) {
    const wrapRef = useRef(null);
    const canvasRef = useRef(null);
    const [data, setData] = useState(null);
    const [err, setErr] = useState(null);
    const [size, setSize] = useState({ w: 0, h: 0 });

    useEffect(() => {
        if (!dataUrl) return;
        let cancelled = false;
        fetch(dataUrl)
            .then((r) => r.json())
            .then((d) => { if (!cancelled) setData(d); })
            .catch((e) => { if (!cancelled) setErr(String(e)); });
        return () => { cancelled = true; };
    }, [dataUrl]);

    // Observe wrapper size so the canvas matches the drawer width
    useEffect(() => {
        if (!wrapRef.current) return;
        const el = wrapRef.current;
        const pitchLen = data?.pitch?.length || 12000;
        const pitchWid = data?.pitch?.width || 7000;
        const aspect = pitchWid / pitchLen;
        const ro = new ResizeObserver(([entry]) => {
            const w = Math.floor(entry.contentRect.width);
            if (!w) return;
            const h = Math.floor(w * aspect);
            setSize({ w, h });
        });
        ro.observe(el);
        return () => ro.disconnect();
    }, [data]);

    useEffect(() => {
        if (!data || !canvasRef.current || !size.w) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const width = size.w;
        const height = size.h;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
        ctx.scale(dpr, dpr);

        const pitchLen = data.pitch?.length || 12000;
        const pitchWid = data.pitch?.width || 7000;
        const PAD = 14;
        const drawW = width - PAD * 2;
        const drawH = height - PAD * 2;

        const toPx = (x, y) => [
            PAD + (x / pitchLen) * drawW,
            PAD + (y / pitchWid) * drawH,
        ];

        // Pitch background
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = '#1a472a';
        ctx.fillRect(PAD, PAD, drawW, drawH);
        ctx.strokeStyle = 'rgba(255,255,255,0.55)';
        ctx.lineWidth = 1.4;
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

        const positions = data.positions || [];
        if (positions.length === 0) {
            ctx.fillStyle = 'rgba(255,255,255,0.6)';
            ctx.font = '13px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No tracked positions found', width / 2, height / 2);
            return;
        }

        // 1. Build alpha intensity on a DPR-scaled offscreen canvas.
        //
        // BUG WE'RE FIXING: previously the offscreen was sized at (width,
        // height) while the main canvas was DPR-scaled (width*dpr × height*dpr).
        // putImageData ignores transforms and uses raw pixel coords — so the
        // smaller image only filled the top-left quadrant of the main canvas
        // (the rest was leftover pitch-background pixels). Match both
        // canvases at physical-pixel size and the heatmap covers the full pitch.
        const off = document.createElement('canvas');
        off.width = width * dpr;
        off.height = height * dpr;
        const octx = off.getContext('2d');
        octx.scale(dpr, dpr);
        const radius = Math.max(18, Math.min(drawW, drawH) * 0.06);
        octx.globalCompositeOperation = 'lighter';
        for (const [x, y] of positions) {
            const [px, py] = toPx(x, y);
            const grad = octx.createRadialGradient(px, py, 0, px, py, radius);
            grad.addColorStop(0, 'rgba(255,255,255,0.18)');
            grad.addColorStop(1, 'rgba(255,255,255,0)');
            octx.fillStyle = grad;
            octx.fillRect(px - radius, py - radius, radius * 2, radius * 2);
        }

        // 2. Read pixels (in physical-pixel size) and remap intensity → heat colour
        const physW = width * dpr;
        const physH = height * dpr;
        const img = octx.getImageData(0, 0, physW, physH);
        const px = img.data;
        for (let i = 0; i < px.length; i += 4) {
            const a = px[i + 3];
            if (a < 8) { px[i + 3] = 0; continue; }
            // Colour ramp: blue → cyan → green → yellow → red
            const t = Math.min(1, a / 200);
            let r, g, b;
            if (t < 0.25)      { r = 0;   g = Math.round(t * 4 * 255);             b = 255; }
            else if (t < 0.5)  { r = 0;   g = 255;                                  b = Math.round((1 - (t - 0.25) * 4) * 255); }
            else if (t < 0.75) { r = Math.round((t - 0.5) * 4 * 255); g = 255;     b = 0; }
            else               { r = 255; g = Math.round((1 - (t - 0.75) * 4) * 255); b = 0; }
            px[i] = r; px[i + 1] = g; px[i + 2] = b;
            px[i + 3] = Math.min(220, a * 1.8);
        }
        // Put remapped pixels back to the offscreen canvas, then composite
        // onto the main canvas with drawImage so alpha-blending preserves
        // the green pitch background underneath.
        // (putImageData on the main canvas directly would overwrite pitch
        // pixels with the transparent regions of the heat layer — that is
        // why the green pitch was disappearing.)
        octx.putImageData(img, 0, 0);
        ctx.drawImage(off, 0, 0, width, height);
    }, [data, size]);

    if (err) return <div className="heatmap-error">Failed to load heatmap: {err}</div>;
    if (!dataUrl) return <div className="heatmap-error">Heatmap data not yet available.</div>;
    return (
        <div ref={wrapRef} className="heatmap-wrap">
            <canvas ref={canvasRef} className="heatmap-canvas" />
        </div>
    );
}
