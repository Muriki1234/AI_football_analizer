import { useRef, useState, useCallback, useEffect } from 'react';

/**
 * TelestrationCanvas — freehand + arrow drawing overlay for video or minimap.
 *
 * Props:
 *   active     : boolean — when true, canvas captures pointer events
 *   canvasRef  : React ref — parent can read this to do toDataURL() for screenshots
 *   width      : number — logical pixel width (matches video or minimap)
 *   height     : number — logical pixel height
 */
const COLORS = ['#ef4444', '#f59e0b', '#22c55e', '#3b82f6', '#f8fafc'];

export default function TelestrationCanvas({ active, parentRef, width, height }) {
    const canvasRef = useRef(null);
    const [color, setColor] = useState(COLORS[0]);
    const [tool, setTool] = useState('pen');      // 'pen' | 'arrow'
    const [drawing, setDrawing] = useState(false);
    const [strokes, setStrokes] = useState([]);    // finished strokes
    const currentStroke = useRef([]);
    const arrowStart = useRef(null);

    // Expose canvas ref to parent
    useEffect(() => {
        if (parentRef) parentRef.current = canvasRef.current;
    }, [parentRef]);

    // Redraw all strokes whenever they change
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        canvas.width = (width || 800) * dpr;
        canvas.height = (height || 450) * dpr;
        canvas.style.width = `${width || 800}px`;
        canvas.style.height = `${height || 450}px`;
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, width || 800, height || 450);

        for (const s of strokes) {
            if (s.type === 'pen') {
                drawPenStroke(ctx, s.points, s.color);
            } else if (s.type === 'arrow') {
                drawArrow(ctx, s.from, s.to, s.color);
            }
        }
    }, [strokes, width, height]);

    const getPos = useCallback((e) => {
        const canvas = canvasRef.current;
        if (!canvas) return { x: 0, y: 0 };
        const rect = canvas.getBoundingClientRect();
        const scaleX = (width || 800) / rect.width;
        const scaleY = (height || 450) / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY,
        };
    }, [width, height]);

    const handlePointerDown = useCallback((e) => {
        if (!active) return;
        e.preventDefault();
        setDrawing(true);
        const pos = getPos(e);
        if (tool === 'pen') {
            currentStroke.current = [pos];
        } else {
            arrowStart.current = pos;
        }
    }, [active, tool, getPos]);

    const handlePointerMove = useCallback((e) => {
        if (!drawing || !active) return;
        e.preventDefault();
        const pos = getPos(e);
        if (tool === 'pen') {
            currentStroke.current.push(pos);
            // Live preview: draw on canvas directly
            const canvas = canvasRef.current;
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            const dpr = window.devicePixelRatio || 1;
            ctx.save();
            ctx.scale(dpr, dpr);
            const pts = currentStroke.current;
            if (pts.length >= 2) {
                ctx.strokeStyle = color;
                ctx.lineWidth = 3;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                ctx.beginPath();
                ctx.moveTo(pts[pts.length - 2].x, pts[pts.length - 2].y);
                ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
                ctx.stroke();
            }
            ctx.restore();
        }
    }, [drawing, active, tool, color, getPos]);

    const handlePointerUp = useCallback((e) => {
        if (!drawing) return;
        setDrawing(false);
        const pos = getPos(e);
        if (tool === 'pen' && currentStroke.current.length > 1) {
            const pts = [...currentStroke.current];
            setStrokes(prev => [...prev, {
                type: 'pen',
                points: pts,
                color,
            }]);
        } else if (tool === 'arrow' && arrowStart.current) {
            const start = { ...arrowStart.current };
            setStrokes(prev => [...prev, {
                type: 'arrow',
                from: start,
                to: pos,
                color,
            }]);
        }
        currentStroke.current = [];
        arrowStart.current = null;
    }, [drawing, tool, color, getPos]);

    const handleClear = useCallback(() => {
        setStrokes([]);
    }, []);

    const handleScreenshot = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const link = document.createElement('a');
        link.download = `tactical_board_${Date.now()}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
    }, []);

    return (
        <>
            <canvas
                ref={canvasRef}
                className={`telestration-canvas ${!active ? 'telestration-canvas--inactive' : ''}`}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onPointerLeave={handlePointerUp}
            />
            <div className={`draw-toolbar ${active ? 'is-visible' : ''}`}>
                <button
                    className={`draw-toolbar__btn ${tool === 'pen' ? 'is-active' : ''}`}
                    onClick={() => setTool('pen')}
                    title="画笔"
                >✏️</button>
                <button
                    className={`draw-toolbar__btn ${tool === 'arrow' ? 'is-active' : ''}`}
                    onClick={() => setTool('arrow')}
                    title="箭头"
                >➡️</button>
                <div className="draw-toolbar__sep" />
                {COLORS.map(c => (
                    <button
                        key={c}
                        className={`draw-toolbar__color ${color === c ? 'is-active' : ''}`}
                        style={{ background: c }}
                        onClick={() => setColor(c)}
                    />
                ))}
                <div className="draw-toolbar__sep" />
                <button className="draw-toolbar__btn" onClick={handleClear} title="清除">🗑️</button>
                <button className="draw-toolbar__btn" onClick={handleScreenshot} title="截图保存">📸</button>
            </div>
        </>
    );
}

// ── Drawing helpers ──────────────────────────────────────────────────────────

function drawPenStroke(ctx, points, color) {
    if (points.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.stroke();
}

function drawArrow(ctx, from, to, color) {
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const angle = Math.atan2(dy, dx);
    const headLen = 18;

    // Shaft
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();

    // Arrowhead
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(to.x, to.y);
    ctx.lineTo(
        to.x - headLen * Math.cos(angle - Math.PI / 6),
        to.y - headLen * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
        to.x - headLen * Math.cos(angle + Math.PI / 6),
        to.y - headLen * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fill();
}
