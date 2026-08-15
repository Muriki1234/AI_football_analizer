import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { HiPause, HiPlay, HiSpeakerWave, HiSpeakerXMark, HiPencil } from 'react-icons/hi2';

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

export default function VideoTimelineMarkers({ segments, matchPeriods, fps, totalFrames, videoRef, drawMode, onToggleDraw, highlights }) {
    const containerRef = useRef(null);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [isMuted, setIsMuted] = useState(true);

    useEffect(() => {
        const video = videoRef?.current;
        if (!video) return undefined;

        const syncTime = () => {
            setCurrentTime(video.currentTime || 0);
            setDuration(video.duration || 0);
        };
        const syncPlaying = () => setIsPlaying(!video.paused && !video.ended);
        const syncVolume = () => setIsMuted(video.muted || video.volume === 0);

        syncTime();
        syncPlaying();
        syncVolume();
        video.addEventListener('timeupdate', syncTime);
        video.addEventListener('loadedmetadata', syncTime);
        video.addEventListener('durationchange', syncTime);
        video.addEventListener('play', syncPlaying);
        video.addEventListener('pause', syncPlaying);
        video.addEventListener('ended', syncPlaying);
        video.addEventListener('volumechange', syncVolume);

        return () => {
            video.removeEventListener('timeupdate', syncTime);
            video.removeEventListener('loadedmetadata', syncTime);
            video.removeEventListener('durationchange', syncTime);
            video.removeEventListener('play', syncPlaying);
            video.removeEventListener('pause', syncPlaying);
            video.removeEventListener('ended', syncPlaying);
            video.removeEventListener('volumechange', syncVolume);
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

        // Non-match segment types that get stripped from the video
        const NON_MATCH_TYPES = new Set(['halftime', 'pre_match', 'post_match']);

        return source.map((seg) => {
            const isNonMatch = NON_MATCH_TYPES.has(seg.type);
            const stitchedStartFrame = mapToStitchedFrame(seg.start_frame);
            const stitchedEndFrame = mapToStitchedFrame(seg.end_frame);
            
            const startSec = stitchedStartFrame / fps;
            const endSec = stitchedEndFrame / fps;
            
            const fallbackStart = Number(seg.start_sec ?? 0);
            const fallbackEnd = Number(seg.end_sec ?? stitchedTotalSec);
            
            const safeStart = Number.isFinite(startSec) ? startSec : fallbackStart;
            const safeEnd = Number.isFinite(endSec) ? endSec : fallbackEnd;
            
            // Bug 3 fix: Non-match segments (halftime etc.) get a thin separator
            // instead of being filtered out, so users can still see the label
            const rawWidth = (safeEnd - safeStart) / stitchedTotalSec * 100;
            const widthPct = isNonMatch && rawWidth <= 0 ? 0.5 : Math.max(0, rawWidth);
            
            return {
                ...seg,
                style: SEGMENT_STYLES[seg.type] || { color: '#64748b', label: seg.type },
                startSec: safeStart,
                endSec: safeEnd,
                widthPct,
                leftPct: Math.max(0, (safeStart / stitchedTotalSec) * 100),
                isSeparator: isNonMatch && rawWidth <= 0,
            };
        });
    }, [segments, matchPeriods, fps, totalFrames, duration]);

    const globalMaxSpeed = useMemo(() => {
        let m = 20; // fallback scale
        items.forEach(seg => {
            const speed = Number(seg.max_speed_kmh);
            if (!isNaN(speed) && speed > m) m = speed;
        });
        return m;
    }, [items]);

    if (items.length === 0) return null;

    const seekTo = (sec) => {
        const v = videoRef?.current;
        if (!v) return;
        const target = Math.max(0, Math.min(sec, v.duration || sec));
        v.currentTime = target;
        setCurrentTime(target);
        v.play?.().catch(() => { });
    };

    const togglePlay = () => {
        const v = videoRef?.current;
        if (!v) return;
        
        // If we hit Play and drawing is active, immediately turn off drawing mode
        if (v.paused && drawMode && onToggleDraw) {
            onToggleDraw(false);
        }
        
        if (v.paused) v.play?.().catch(() => { });
        else v.pause?.();
    };

    const toggleMute = () => {
        const v = videoRef?.current;
        if (!v) return;
        v.muted = !v.muted;
    };

    const computedDuration = useMemo(() => {
        if (!fps) return duration;
        let stitchedTotalFrames = totalFrames;
        if (matchPeriods && matchPeriods.length > 0) {
            stitchedTotalFrames = matchPeriods.reduce((acc, [ps, pe]) => acc + (pe - ps), 0);
        }
        return stitchedTotalFrames ? stitchedTotalFrames / fps : duration;
    }, [fps, totalFrames, matchPeriods, duration]);

    const activeDuration = computedDuration || duration;
    const progressPct = activeDuration ? Math.max(0, Math.min(100, (currentTime / activeDuration) * 100)) : 0;

    const [isScrubbing, setIsScrubbing] = useState(false);
    const [hoverPos, setHoverPos] = useState(null);
    const trackRef = useRef(null);

    const updateSeek = useCallback((clientX) => {
        const v = videoRef?.current;
        const track = trackRef.current;
        if (!v || !track || !activeDuration) return;
        const rect = track.getBoundingClientRect();
        const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        const target = pct * activeDuration;
        v.currentTime = target;
        setCurrentTime(target);
    }, [videoRef, activeDuration]);

    const handlePointerDown = (e) => {
        // e.preventDefault(); // allow focus
        setIsScrubbing(true);
        updateSeek(e.clientX);
    };

    const handlePointerMove = (e) => {
        if (!trackRef.current || !activeDuration) return;
        const rect = trackRef.current.getBoundingClientRect();
        const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        setHoverPos({ pct, time: pct * activeDuration });
    };

    const handlePointerLeave = () => {
        setHoverPos(null);
    };

    useEffect(() => {
        if (!isScrubbing) return;
        const onMove = (e) => updateSeek(e.clientX);
        const onUp = () => setIsScrubbing(false);
        window.addEventListener('pointermove', onMove);
        window.addEventListener('pointerup', onUp);
        return () => {
            window.removeEventListener('pointermove', onMove);
            window.removeEventListener('pointerup', onUp);
        };
    }, [isScrubbing, updateSeek]);

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
                    className="video-markers__play"
                    onClick={toggleMute}
                    aria-label={isMuted ? 'Unmute' : 'Mute'}
                    title={isMuted ? 'Unmute' : 'Mute'}
                >
                    {isMuted ? <HiSpeakerXMark /> : <HiSpeakerWave />}
                </button>
                <button
                    type="button"
                    className={`video-markers__play ${drawMode ? 'is-active-draw' : ''}`}
                    onClick={() => {
                        // Click pen -> toggle draw mode. If turning ON, auto-pause video
                        const nextState = !drawMode;
                        if (onToggleDraw) onToggleDraw(nextState);
                        const v = videoRef?.current;
                        if (v && nextState && !v.paused) v.pause?.();
                    }}
                    aria-label={drawMode ? 'Exit Drawing Mode' : 'Enter Drawing Mode'}
                    title="战术画笔"
                    style={drawMode ? { color: '#38bdf8', background: 'rgba(56, 189, 248, 0.15)' } : {}}
                >
                    <HiPencil />
                </button>
                <div
                    className="video-markers__track"
                    ref={trackRef}
                    onPointerDown={handlePointerDown}
                    onPointerMove={handlePointerMove}
                    onPointerLeave={handlePointerLeave}
                    aria-label="Replay progress"
                    style={{ position: 'relative', cursor: 'pointer' }}
                >
                    <span className="video-markers__segments">
                        {items.map((seg, i) => (
                            <span
                                key={`${seg.type}-${i}`}
                                className={`video-markers__seg${seg.isSeparator ? ' video-markers__seg--separator' : ''}`}
                                style={{
                                    left: `${seg.leftPct}%`,
                                    width: `${seg.widthPct}%`,
                                    background: seg.isSeparator ? 'rgba(148, 163, 184, 0.6)' : seg.style.color,
                                    ...(seg.isSeparator ? { borderLeft: '1px dashed #64748b', borderRight: '1px dashed #64748b' } : {}),
                                }}
                                title={`${seg.style.label}${seg.isSeparator ? '' : ` • ${formatTime(seg.startSec)}-${formatTime(seg.endSec)}`}`}
                            >
                                {!seg.isSeparator && <span className="video-markers__seg-label">{seg.style.label}</span>}
                            </span>
                        ))}
                    </span>
                    
                    {/* Heatmap overlay */}
                    <span className="video-markers__heatmap">
                        {items.map((seg, i) => {
                            if (seg.isSeparator) return null;
                            const speed = Number(seg.max_speed_kmh);
                            if (isNaN(speed) || speed <= 0) return null;
                            
                            const heightPct = Math.min(100, Math.max(10, (speed / globalMaxSpeed) * 100));
                            // Color logic: Red for high intensity, orange/yellow for medium
                            const color = heightPct > 80 ? '#ef4444' : (heightPct > 50 ? '#f59e0b' : '#38bdf8');
                            return (
                                <span
                                    key={`heat-${i}`}
                                    className="video-markers__heatbar"
                                    style={{
                                        left: `${seg.leftPct}%`,
                                        width: `${seg.widthPct}%`,
                                        height: `${heightPct}%`,
                                        background: color,
                                    }}
                                    title={`Intensity/Speed: ${speed} km/h`}
                                />
                            );
                        })}
                    </span>

                    <span className="video-markers__progress" style={{ width: `${progressPct}%` }} />
                    <span className="video-markers__thumb" style={{ left: `${progressPct}%`, pointerEvents: 'none' }} />
                    
                    {/* Highlight Dots */}
                    {highlights && highlights.map((hl, i) => {
                        const pct = duration ? (hl.time / duration) * 100 : 0;
                        if (pct < 0 || pct > 100) return null;
                        return (
                            <div
                                key={`hl-${i}`}
                                className="video-markers__highlight-dot"
                                style={{ left: `${pct}%` }}
                                title={`Highlight: ${hl.label}`}
                                onClick={(e) => {
                                    e.stopPropagation();
                                    seekTo(hl.time);
                                }}
                            />
                        );
                    })}
                    
                    {hoverPos && !isScrubbing && (
                        <div 
                            className="video-markers__hover-tooltip" 
                            style={{
                                position: 'absolute',
                                left: `${hoverPos.pct * 100}%`,
                                top: '-30px',
                                transform: 'translateX(-50%)',
                                background: '#1e293b',
                                color: 'white',
                                padding: '4px 8px',
                                borderRadius: '4px',
                                fontSize: '12px',
                                pointerEvents: 'none',
                                whiteSpace: 'nowrap',
                                zIndex: 10,
                                boxShadow: '0 2px 4px rgba(0,0,0,0.3)'
                            }}
                        >
                            {formatTime(hoverPos.time)}
                        </div>
                    )}
                </div>
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
