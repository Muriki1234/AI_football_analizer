import { useRef, useState, useCallback, useEffect, useImperativeHandle } from 'react';
import PropTypes from 'prop-types';
import { HiPencil, HiArrowUpRight, HiArrowUturnLeft, HiTrash, HiCamera, HiMinus } from 'react-icons/hi2';

/**
 * TelestrationCanvas — freehand + arrow drawing overlay for video or minimap.
 *
 * Props:
 *   active             : boolean — when true, canvas captures pointer events
 *   parentRef          : React ref — parent can read this to do toDataURL() for screenshots
 *   width              : number — logical pixel width (matches video or minimap)
 *   height             : number — logical pixel height
 *   onInteractionStart : function — called when drawing begins, useful for pausing video
 */
const COLORS = ['#ef4444', '#f59e0b', '#22c55e', '#3b82f6', '#f8fafc'];

export default function TelestrationCanvas({ active, parentRef, width, height, onInteractionStart, initialStrokes = [] }) {
    const canvasRef = useRef(null);
    const [color, setColor] = useState(COLORS[0]);
    const [tool, setTool] = useState('arrow');      // 'pen' | 'arrow' | 'dashed-arrow'
    const [lineWidth, setLineWidth] = useState(4);
    const [drawing, setDrawing] = useState(false);
    const [strokes, setStrokes] = useState(initialStrokes);    // finished strokes
    const currentStroke = useRef([]);

    useEffect(() => {
        setStrokes(initialStrokes);
    }, [initialStrokes]);


    // Keyboard shortcut for Undo (Ctrl+Z or Cmd+Z)
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (!active) return;
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
                e.preventDefault();
                setStrokes((prev) => prev.slice(0, -1));
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [active]);

    // Redraw all strokes whenever they change
    const redraw = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        canvas.width = (width || 800) * dpr;
        canvas.height = (height || 450) * dpr;
        canvas.style.width = `${width || 800}px`;
        canvas.style.height = `${height || 450}px`;
        ctx.scale(dpr, dpr);
        
        const render = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            for (const s of strokes) {
                drawFreehand(ctx, s.points, s.color, s.type, s.lineWidth || 4);
            }
            if (drawing) {
                drawFreehand(ctx, currentStroke.current, color, tool, lineWidth);
            }
        };
        requestAnimationFrame(render);
    }, [strokes, width, height, drawing, color, tool, lineWidth]);

    useEffect(() => {
        redraw();
    }, [redraw]);

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
        if (onInteractionStart) onInteractionStart();
        
        setDrawing(true);
        const pos = getPos(e);
        currentStroke.current = [pos];
    }, [active, getPos, onInteractionStart]);

    const handlePointerMove = useCallback((e) => {
        if (!drawing || !active) return;
        e.preventDefault();
        const pos = getPos(e);
        currentStroke.current.push(pos);
        
        // Live preview: redraw all committed strokes, then draw current
        redraw();
        
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        
        const pts = currentStroke.current;
        if (pts.length >= 2) {
            drawFreehand(ctx, pts, color, tool, lineWidth);
        }
    }, [drawing, active, tool, color, lineWidth, getPos, redraw]);

    const handlePointerUp = useCallback((e) => {
        if (!drawing) return;
        setDrawing(false);
        const pts = [...currentStroke.current];
        
        // Only save if stroke has some length
        if (pts.length >= 2) {
            setStrokes(prev => [...prev, {
                type: tool,
                points: pts,
                color,
                lineWidth,
            }]);
        }
        currentStroke.current = [];
        redraw();
    }, [drawing, tool, color, lineWidth, redraw]);

    const handleUndo = useCallback(() => {
        setStrokes(prev => prev.slice(0, -1));
    }, []);

    const handleClear = useCallback(() => {
        setStrokes([]);
        redraw();
    }, [redraw]);

    // Expose methods to parent
    useImperativeHandle(parentRef, () => ({
        clearCanvas: handleClear,
        getStrokes: () => strokes,
    }), [handleClear, strokes]);

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
                style={{ cursor: active ? 'crosshair' : 'default' }}
            />
            <div className={`draw-toolbar ${active ? 'is-visible' : ''}`}>
                <button
                    className={`draw-toolbar__btn ${tool === 'pen' ? 'is-active' : ''}`}
                    onClick={() => setTool('pen')}
                    title="画笔"
                ><HiPencil /></button>
                <button
                    className={`draw-toolbar__btn ${tool === 'arrow' ? 'is-active' : ''}`}
                    onClick={() => setTool('arrow')}
                    title="实线箭头"
                ><HiArrowUpRight /></button>
                <button
                    className={`draw-toolbar__btn ${tool === 'dashed-arrow' ? 'is-active' : ''}`}
                    onClick={() => setTool('dashed-arrow')}
                    title="虚线箭头"
                    style={{ position: 'relative' }}
                >
                    <HiArrowUpRight />
                    <div style={{ position: 'absolute', bottom: 2, left: 6, right: 6, height: 2, borderBottom: '2px dashed currentColor', opacity: 0.7 }} />
                </button>
                
                <div className="draw-toolbar__sep" />
                
                {/* Line width selectors */}
                <button
                    className={`draw-toolbar__btn ${lineWidth === 2 ? 'is-active' : ''}`}
                    onClick={() => setLineWidth(2)}
                    title="细线条"
                ><HiMinus style={{ transform: 'scaleY(0.5)' }} /></button>
                <button
                    className={`draw-toolbar__btn ${lineWidth === 4 ? 'is-active' : ''}`}
                    onClick={() => setLineWidth(4)}
                    title="中等线条"
                ><HiMinus /></button>
                <button
                    className={`draw-toolbar__btn ${lineWidth === 8 ? 'is-active' : ''}`}
                    onClick={() => setLineWidth(8)}
                    title="粗线条"
                ><HiMinus style={{ transform: 'scaleY(2)' }} /></button>

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
                
                <button className="draw-toolbar__btn" onClick={handleUndo} title="撤销 (Ctrl+Z)"><HiArrowUturnLeft /></button>
                <button className="draw-toolbar__btn" onClick={handleClear} title="全部清除"><HiTrash /></button>
                <button className="draw-toolbar__btn" onClick={handleScreenshot} title="截图保存"><HiCamera /></button>
            </div>
        </>
    );
}

// ── Drawing helpers ──────────────────────────────────────────────────────────

function drawFreehand(ctx, points, color, type, lineWidth = 4) {
    if (!points || points.length < 2) return;
    
    const isArrow = type === 'arrow' || type === 'dashed-arrow';
    
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // Add professional drop shadow
    ctx.shadowColor = 'rgba(0, 0, 0, 0.7)';
    ctx.shadowBlur = 6;
    ctx.shadowOffsetX = 1;
    ctx.shadowOffsetY = 3;

    if (type === 'dashed-arrow') {
        ctx.setLineDash([12, 8]); // Dashed lines for arrows (tactical style)
    } else {
        ctx.setLineDash([]);
    }

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    
    // Smooth quadratic curve drawing
    for (let i = 1; i < points.length - 1; i++) {
        const xc = (points[i].x + points[i + 1].x) / 2;
        const yc = (points[i].y + points[i + 1].y) / 2;
        ctx.quadraticCurveTo(points[i].x, points[i].y, xc, yc);
    }
    // Line to the final point
    if (points.length > 2) {
        ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);
    }
    
    ctx.stroke();

    if (isArrow) {
        ctx.setLineDash([]); // Reset line dash for the arrowhead
        
        // Draw arrowhead at the very end
        const end = points[points.length - 1];
        // Calculate tangent using a point slightly before the end for stability
        const p2 = points[Math.max(0, points.length - 5)];
        const dx = end.x - p2.x;
        const dy = end.y - p2.y;
        
        // If the stroke is too short, don't draw arrow head
        if (Math.hypot(dx, dy) < 5) return;
        
        const angle = Math.atan2(dy, dx);
        const headLen = 18;
        
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(end.x, end.y);
        ctx.lineTo(
            end.x - headLen * Math.cos(angle - Math.PI / 6),
            end.y - headLen * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(
            end.x - headLen * Math.cos(angle + Math.PI / 6),
            end.y - headLen * Math.sin(angle + Math.PI / 6)
        );
        ctx.closePath();
        ctx.fill();
    }
    
    // Reset shadow & dash so it doesn't affect other things
    ctx.shadowColor = 'transparent';
    ctx.setLineDash([]);
}

TelestrationCanvas.propTypes = {
    active: PropTypes.bool,
    parentRef: PropTypes.shape({ current: PropTypes.any }),
    width: PropTypes.number,
    height: PropTypes.number,
    onInteractionStart: PropTypes.func,
    initialStrokes: PropTypes.array
};
