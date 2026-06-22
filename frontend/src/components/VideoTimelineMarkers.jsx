import { useEffect, useMemo, useRef, useState } from 'react';
import { HiPause, HiPlay } from 'react-icons/hi2';

const SEGMENT_STYLES = {
    first_half: { color: '#3498db', label: 'First Half' },
    second_half: { color: '#e74c3c', label: 'Second Half' },
    halftime: { color: '#94a3b8', label: 'Halftime' },
    pre_match: { color: '#64748b', label: 'Pre Match' },
    post_match: { color: '#64748b', label: 'Post Match' },
    match: { color: '#3498db', label: 'Full Match' },
    full_match: { color: '#3498db', label: 'Full Match' },
};

const formatTime = (sec) => {
    const safeSec = Number.isFinite(sec) ? Math.max(0, sec) : 0;
    const m = Math.floor(safeSec / 60);
    const s = Math.floor(safeSec % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
};

export default function VideoTimelineMarkers({ segments, matchPeriods, fps, totalFrames, videoRef }) {
    const containerRef = useRef(null);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);

    useEffect(() => {
        const video = videoRef?.current;
        if (!video) return undefined;

        const syncTime = () => {
            setCurrentTime(video.currentTime || 0);
            setDuration(video.duration || 0);
        };
        const syncPlaying = () => setIsPlaying(!video.paused && !video.ended);

        syncTime();
        syncPlaying();
        video.addEventListener('timeupdate', syncTime);
        video.addEventListener('loadedmetadata', syncTime);
        video.addEventListener('durationchange', syncTime);
        video.addEventListener('play', syncPlaying);
        video.addEventListener('pause', syncPlaying);
        video.addEventListener('ended', syncPlaying);

        return () => {
            video.removeEventListener('timeupdate', syncTime);
            video.removeEventListener('loadedmetadata', syncTime);
            video.removeEventListener('durationchange', syncTime);
            video.removeEventListener('play', syncPlaying);
            video.removeEventListener('pause', syncPlaying);
            video.removeEventListener('ended', syncPlaying);
        };
    }, [videoRef]);

    const items = useMemo(() => {
        // Map original frame to stitched frame
        const mapToStitchedFrame = (origFrame) => {
            if (!matchPeriods || matchPeriods.length === 0) return origFrame;
            let curStitched = 0;
            for (const [ps, pe] of matchPeriods) {
                if (origFrame >= ps && origFrame < pe) {
                    return curStitched + (origFrame - ps);
                }
                if (origFrame >= pe) {
                    curStitched += (pe - ps);
                }
            }
            return curStitched;
        };

        const source = segments?.length
            ? segments
            : [{
                type: 'full_match',
                start_frame: 0,
                end_frame: totalFrames || Math.round((duration || 1) * (fps || 1)),
                start_sec: 0,
                end_sec: duration || 1,
            }];

        // Stitched total frames is the sum of all match periods, or just totalFrames
        let stitchedTotalFrames = totalFrames;
        if (matchPeriods && matchPeriods.length > 0) {
            stitchedTotalFrames = matchPeriods.reduce((acc, [ps, pe]) => acc + (pe - ps), 0);
        }
        
        const stitchedTotalSec = fps && stitchedTotalFrames ? stitchedTotalFrames / fps : duration;
        if (!stitchedTotalSec) return [];

        return source.map((seg) => {
            const stitchedStartFrame = mapToStitchedFrame(seg.start_frame);
            const stitchedEndFrame = mapToStitchedFrame(seg.end_frame);
            
            const startSec = stitchedStartFrame / fps;
            const endSec = stitchedEndFrame / fps;
            
            const fallbackStart = Number(seg.start_sec ?? 0);
            const fallbackEnd = Number(seg.end_sec ?? stitchedTotalSec);
            
            const safeStart = Number.isFinite(startSec) ? startSec : fallbackStart;
            const safeEnd = Number.isFinite(endSec) ? endSec : fallbackEnd;
            
            return {
                ...seg,
                style: SEGMENT_STYLES[seg.type] || { color: '#64748b', label: seg.type },
                startSec: safeStart,
                endSec: safeEnd,
                widthPct: Math.max(0, ((safeEnd - safeStart) / stitchedTotalSec) * 100),
                leftPct: Math.max(0, (safeStart / stitchedTotalSec) * 100),
            };
        }).filter(seg => seg.widthPct > 0); // Hide dropped segments (like halftime)
    }, [segments, matchPeriods, fps, totalFrames, duration]);

    if (items.length === 0) return null;

    const seekTo = (sec) => {
        const v = videoRef?.current;
        if (!v) return;
        v.currentTime = Math.max(0, Math.min(sec, v.duration || sec));
        v.play?.().catch(() => { });
    };

    const togglePlay = () => {
        const v = videoRef?.current;
        if (!v) return;
        if (v.paused) v.play?.().catch(() => { });
        else v.pause?.();
    };

    const seekFromPointer = (event) => {
        const v = videoRef?.current;
        const track = event.currentTarget;
        if (!v || !track || !duration) return;
        const rect = track.getBoundingClientRect();
        const pct = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
        v.currentTime = pct * duration;
    };

    const progressPct = duration ? Math.max(0, Math.min(100, (currentTime / duration) * 100)) : 0;

    return (
        <div className="video-markers" ref={containerRef}>
            <div className="video-markers__controls">
                <button
                    type="button"
                    className="video-markers__play"
                    onClick={togglePlay}
                    aria-label={isPlaying ? 'Pause replay' : 'Play replay'}
                    title={isPlaying ? 'Pause' : 'Play'}
                >
                    {isPlaying ? <HiPause /> : <HiPlay />}
                </button>
                <button
                    type="button"
                    className="video-markers__track"
                    onClick={seekFromPointer}
                    aria-label="Replay progress"
                >
                    <span className="video-markers__segments">
                        {items.map((seg, i) => (
                            <span
                                key={`${seg.type}-${i}`}
                                className="video-markers__seg"
                                style={{
                                    left: `${seg.leftPct}%`,
                                    width: `${seg.widthPct}%`,
                                    background: seg.style.color,
                                }}
                                title={`${seg.style.label} • ${formatTime(seg.startSec)}-${formatTime(seg.endSec)}`}
                            >
                                <span className="video-markers__seg-label">{seg.style.label}</span>
                            </span>
                        ))}
                    </span>
                    <span className="video-markers__progress" style={{ width: `${progressPct}%` }} />
                    <span className="video-markers__thumb" style={{ left: `${progressPct}%` }} />
                </button>
                <span className="video-markers__time">
                    {formatTime(currentTime)} / {formatTime(duration || items.at(-1)?.endSec || 0)}
                </span>
            </div>
            <div className="video-markers__chapters">
                {items.map((seg, i) => (
                    <button
                        key={`chapter-${seg.type}-${i}`}
                        type="button"
                        className="video-markers__chapter"
                        onClick={() => seekTo(seg.startSec)}
                        title={`${seg.style.label} • ${formatTime(seg.startSec)}`}
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
