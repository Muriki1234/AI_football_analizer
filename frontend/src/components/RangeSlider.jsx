import { useState, useRef, useCallback, useEffect } from 'react';
import './RangeSlider.css';

export default function RangeSlider({ min = 0, max = 100, onChange, formatLabel }) {
    const [low, setLow] = useState(min);
    const [high, setHigh] = useState(max);
    const trackRef = useRef(null);
    const dragging = useRef(null);

    const pct = (v) => ((v - min) / (max - min)) * 100;

    const getValueFromEvent = useCallback((e) => {
        const rect = trackRef.current.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        return Math.round(min + ratio * (max - min));
    }, [min, max]);

    const onPointerDown = useCallback((handle) => (e) => {
        e.preventDefault();
        dragging.current = handle;
    }, []);

    const onPointerMove = useCallback((e) => {
        if (!dragging.current) return;
        const val = getValueFromEvent(e);
        if (dragging.current === 'low') {
            const newLow = Math.min(val, high - 1);
            setLow(newLow);
            onChange?.([newLow, high]);
        } else {
            const newHigh = Math.max(val, low + 1);
            setHigh(newHigh);
            onChange?.([low, newHigh]);
        }
    }, [low, high, getValueFromEvent, onChange]);

    const onPointerUp = useCallback(() => {
        dragging.current = null;
    }, []);

    useEffect(() => {
        window.addEventListener('mousemove', onPointerMove);
        window.addEventListener('mouseup', onPointerUp);
        window.addEventListener('touchmove', onPointerMove);
        window.addEventListener('touchend', onPointerUp);
        return () => {
            window.removeEventListener('mousemove', onPointerMove);
            window.removeEventListener('mouseup', onPointerUp);
            window.removeEventListener('touchmove', onPointerMove);
            window.removeEventListener('touchend', onPointerUp);
        };
    }, [onPointerMove, onPointerUp]);

    const fmt = formatLabel || ((v) => v);

    return (
        <div className="range-slider">
            <div className="range-slider__labels">
                <span className="range-slider__label">{fmt(low)}</span>
                <span className="range-slider__label">{fmt(high)}</span>
            </div>
            <div className="range-slider__track" ref={trackRef}>
                <div className="range-slider__rail" />
                <div
                    className="range-slider__fill"
                    style={{ left: `${pct(low)}%`, width: `${pct(high) - pct(low)}%` }}
                />
                <div
                    className="range-slider__handle"
                    style={{ left: `${pct(low)}%` }}
                    onMouseDown={onPointerDown('low')}
                    onTouchStart={onPointerDown('low')}
                />
                <div
                    className="range-slider__handle"
                    style={{ left: `${pct(high)}%` }}
                    onMouseDown={onPointerDown('high')}
                    onTouchStart={onPointerDown('high')}
                />
            </div>
        </div>
    );
}
