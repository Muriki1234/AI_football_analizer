import { useMemo, useRef } from 'react';

const SEGMENT_STYLES = {
    first_half: { color: '#3498db', label: 'First Half' },
    second_half: { color: '#e74c3c', label: 'Second Half' },
    halftime: { color: '#94a3b8', label: 'Halftime' },
};

const formatTime = (sec) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
};

/**
 * Renders a segment strip below the video that lets the user jump to
 * scene boundaries (first_half / halftime / second_half).
 *
 * Falls back to nothing if segments / fps / total_frames aren't available
 * (e.g. session created before scene_segmentation ran).
 */
export default function VideoTimelineMarkers({ segments, fps, totalFrames, videoRef }) {
    const containerRef = useRef(null);

    const items = useMemo(() => {
        if (!segments?.length || !fps || !totalFrames) return [];
        const totalSec = totalFrames / fps;
        return segments.map((seg) => {
            const startSec = seg.start_frame / fps;
            const endSec = seg.end_frame / fps;
            return {
                ...seg,
                style: SEGMENT_STYLES[seg.type] || { color: '#64748b', label: seg.type },
                startSec,
                endSec,
                widthPct: ((endSec - startSec) / totalSec) * 100,
                leftPct: (startSec / totalSec) * 100,
            };
        });
    }, [segments, fps, totalFrames]);

    if (items.length === 0) return null;

    const seekTo = (sec) => {
        const v = videoRef?.current;
        if (!v) return;
        v.currentTime = Math.max(0, Math.min(sec, v.duration || sec));
        v.play?.().catch(() => { });
    };

    return (
        <div className="video-markers" ref={containerRef}>
            <div className="video-markers__strip">
                {items.map((seg, i) => (
                    <button
                        key={`${seg.type}-${i}`}
                        type="button"
                        className="video-markers__seg"
                        style={{
                            left: `${seg.leftPct}%`,
                            width: `${seg.widthPct}%`,
                            background: seg.style.color,
                        }}
                        onClick={() => seekTo(seg.startSec)}
                        title={`${seg.style.label} • ${formatTime(seg.startSec)}–${formatTime(seg.endSec)}`}
                    >
                        <span className="video-markers__seg-label">{seg.style.label}</span>
                    </button>
                ))}
            </div>
            <div className="video-markers__legend">
                {items.map((seg, i) => (
                    <button
                        key={`legend-${i}`}
                        type="button"
                        className="video-markers__legend-item"
                        onClick={() => seekTo(seg.startSec)}
                    >
                        <span
                            className="video-markers__legend-dot"
                            style={{ background: seg.style.color }}
                        />
                        <span>{seg.style.label}</span>
                        <span className="video-markers__legend-time">
                            {formatTime(seg.startSec)}
                        </span>
                    </button>
                ))}
            </div>
        </div>
    );
}
