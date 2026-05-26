/**
 * MatchPeriods page — formerly Trimmer.
 *
 * Sits between Upload and MultiSegmentConfig. User defines 1+ valid match
 * intervals; the gaps between them are "skip" regions that the backend
 * leaves out of analysis, tracking, and the rendered replay.
 *
 * Default state is one period covering the whole video — so a user who
 * just hits "Continue" gets the previous behaviour unchanged.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import {
    HiArrowLeft, HiArrowRight, HiPlus, HiMinus, HiArrowPath,
} from 'react-icons/hi2';
import { getSession, saveMatchPeriods } from '../services/api';
import StepNav from '../components/StepNav';
import './Trimmer.css';

// For real match videos (≥5 min) keep 30s minimum so the backend is happy.
// For short test clips we drop to 3s so handles and Add-Break still work.
// Backend has a matching check — keep these in sync with handler.py.
const REAL_MATCH_THRESHOLD_SEC = 300;  // 5 minutes
const MIN_PERIOD_SEC_REAL = 30;
const MIN_PERIOD_SEC_TEST = 3;
const MAX_PERIODS = 4;           // 4 periods × 4 segs = 16 picks ceiling
const HANDLE_PX = 7;             // half the handle CSS width (14px) — for centering

function fmt(sec) {
    if (!Number.isFinite(sec) || sec < 0) return '00:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function parseTimecode(str, max) {
    const m = /^(\d{1,3}):(\d{2})$/.exec(str.trim());
    if (!m) return null;
    const sec = parseInt(m[1], 10) * 60 + parseInt(m[2], 10);
    return Math.max(0, Math.min(max, sec));
}

export default function Trimmer() {
    const navigate = useNavigate();
    const location = useLocation();
    const query = new URLSearchParams(location.search);
    const sessionId =
        location.state?.sessionId || location.state?.videoId || query.get('sessionId');

    const [duration, setDuration] = useState(0);   // seconds
    const [videoUrl, setVideoUrl] = useState(null);
    const [previewTime, setPreviewTime] = useState(0);

    // Dynamic minimum — short test clips use 3s so handles/Add-Break still work.
    // Real match videos (≥5 min) use the full 30s that the backend enforces.
    const minPeriodSec = duration >= REAL_MATCH_THRESHOLD_SEC
        ? MIN_PERIOD_SEC_REAL
        : MIN_PERIOD_SEC_TEST;
    const videoRef = useRef(null);

    // periods: sorted by start, non-overlapping, in SECONDS (we send frames to backend)
    const [periods, setPeriods] = useState([{ start: 0, end: 0 }]);
    const [dragging, setDragging] = useState(null);   // { idx, edge: 'start'|'end' }
    const trackRef = useRef(null);

    // Load session + initial periods (from session.extra.match_periods_sec
    // if user is coming back via "Back" from MultiSegmentConfig — point D)
    useEffect(() => {
        if (!sessionId) return;
        (async () => {
            try {
                const s = await getSession(sessionId);
                setVideoUrl(s?.video_url || null);

                const dur = await new Promise((resolve) => {
                    if (!s?.video_url) return resolve(0);
                    const v = document.createElement('video');
                    v.preload = 'metadata';
                    v.src = s.video_url;
                    v.addEventListener('loadedmetadata',
                        () => resolve(v.duration || 0), { once: true });
                    v.addEventListener('error', () => resolve(0), { once: true });
                });
                setDuration(dur);

                const saved = Array.isArray(s?.match_periods_sec) ? s.match_periods_sec : null;
                if (saved && saved.length > 0) {
                    setPeriods(saved);
                } else {
                    setPeriods([{ start: 0, end: dur }]);
                }
            } catch (e) {
                toast.error('Failed to load video: ' + e.message);
            }
        })();
    }, [sessionId]);

    // Scrub video preview when user drags a handle
    useEffect(() => {
        if (videoRef.current && Number.isFinite(previewTime)) {
            videoRef.current.currentTime = previewTime;
        }
    }, [previewTime]);

    // ── Add / remove / reset ──────────────────────────────────────────────
    const addBreak = () => {
        if (periods.length >= MAX_PERIODS) {
            toast(`Maximum ${MAX_PERIODS} periods`);
            return;
        }
        const next = [...periods];
        let longestIdx = 0;
        for (let i = 0; i < next.length; i++) {
            if (next[i].end - next[i].start > next[longestIdx].end - next[longestIdx].start) {
                longestIdx = i;
            }
        }
        const p = next[longestIdx];
        const mid = (p.start + p.end) / 2;
        // Gap between the two resulting periods — scale with minPeriodSec so
        // short test clips still get a visible gap without violating the minimum.
        const gap = Math.max(2, minPeriodSec * 0.5);
        if (mid - p.start < minPeriodSec + gap / 2
            || p.end - mid < minPeriodSec + gap / 2) {
            toast(`Period too short to split (need ≥ ${Math.ceil(2 * minPeriodSec + gap)}s)`);
            return;
        }
        next.splice(longestIdx, 1,
            { start: p.start, end: mid - gap / 2 },
            { start: mid + gap / 2, end: p.end },
        );
        setPeriods(next);
    };

    const removeBreak = () => {
        if (periods.length <= 1) return;
        const next = periods.slice(0, -2);
        const a = periods[periods.length - 2];
        const b = periods[periods.length - 1];
        next.push({ start: a.start, end: b.end });
        setPeriods(next);
    };

    const reset = () => {
        setPeriods([{ start: 0, end: duration }]);
    };

    // ── Handle dragging ───────────────────────────────────────────────────
    const onTrackPointerDown = (idx, edge) => (e) => {
        e.preventDefault();
        setDragging({ idx, edge });
    };

    useEffect(() => {
        if (!dragging) return;
        const onMove = (e) => {
            if (!trackRef.current) return;
            const rect = trackRef.current.getBoundingClientRect();
            const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
            const ratio = Math.max(0, Math.min(1, x / rect.width));
            const newSec = ratio * duration;

            setPeriods((prev) => {
                const next = prev.map((p) => ({ ...p }));
                const { idx, edge } = dragging;
                const prevBound = idx > 0 ? next[idx - 1].end : 0;
                const nextBound = idx < next.length - 1 ? next[idx + 1].start : duration;
                if (edge === 'start') {
                    next[idx].start = Math.max(prevBound,
                        Math.min(newSec, next[idx].end - minPeriodSec));
                } else {
                    next[idx].end = Math.min(nextBound,
                        Math.max(newSec, next[idx].start + minPeriodSec));
                }
                return next;
            });
            setPreviewTime(newSec);
        };
        const onUp = () => setDragging(null);
        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', onUp);
        window.addEventListener('touchmove', onMove);
        window.addEventListener('touchend', onUp);
        return () => {
            window.removeEventListener('mousemove', onMove);
            window.removeEventListener('mouseup', onUp);
            window.removeEventListener('touchmove', onMove);
            window.removeEventListener('touchend', onUp);
        };
    }, [dragging, duration]);

    // ── Numeric mm:ss inputs ──────────────────────────────────────────────
    const setEdge = (idx, edge, sec) => {
        setPeriods((prev) => {
            const next = prev.map((p) => ({ ...p }));
            const prevBound = idx > 0 ? next[idx - 1].end : 0;
            const nextBound = idx < next.length - 1 ? next[idx + 1].start : duration;
            if (edge === 'start') {
                next[idx].start = Math.max(prevBound,
                    Math.min(sec, next[idx].end - minPeriodSec));
            } else {
                next[idx].end = Math.min(nextBound,
                    Math.max(sec, next[idx].start + minPeriodSec));
            }
            return next;
        });
    };

    const totalMatchSec = useMemo(
        () => periods.reduce((sum, p) => sum + (p.end - p.start), 0),
        [periods],
    );
    const skippedSec = Math.max(0, duration - totalMatchSec);

    const handleContinue = async () => {
        if (!sessionId) {
            navigate('/upload');
            return;
        }
        for (let i = 0; i < periods.length; i++) {
            if (periods[i].end - periods[i].start < minPeriodSec) {
                toast.error(`Period ${i + 1} is shorter than ${minPeriodSec}s`);
                return;
            }
        }
        // Persist so "Back" works + re-pick flow remembers (point D)
        try {
            await saveMatchPeriods(sessionId, periods);
        } catch (e) {
            console.warn('Failed to persist match periods (continuing anyway):', e);
        }
        navigate(`/configure-multi?sessionId=${encodeURIComponent(sessionId)}`, {
            state: {
                sessionId,
                videoId: sessionId,
                matchPeriods: periods,
            },
        });
    };

    if (!sessionId) {
        return (
            <div className="page-container">
                <p style={{ padding: 24 }}>No session — go back to upload.</p>
                <button className="btn btn-primary" onClick={() => navigate('/upload')}>
                    Go to Upload
                </button>
            </div>
        );
    }

    return (
        <div className="page-container trim-page">
            <div className="bg-grid" />

            <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                style={{ padding: '16px 24px' }}
            >
                <button className="btn btn-ghost" onClick={() => navigate('/upload')}>
                    <HiArrowLeft /> Back
                </button>
            </motion.div>

            <StepNav />

            <motion.div
                className="config-page__header"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
            >
                <h1>Mark Match Periods</h1>
                <p>
                    Drag the handles to skip pre-game intro, halftime,
                    and post-game cooldown. Click <strong>+ Add break</strong>
                    {' '}for a halftime split.
                </p>
            </motion.div>

            <div className="trim-page__layout">
                {videoUrl && (
                    <video
                        ref={videoRef}
                        src={videoUrl}
                        className="trim-page__video"
                        controls
                        muted
                    />
                )}

                <div className="periods-track" ref={trackRef}>
                    <div className="periods-track__bg" />
                    {duration > 0 && periods.map((p, i) => {
                        const left = (p.start / duration) * 100;
                        const width = Math.max(0, ((p.end - p.start) / duration) * 100);
                        return (
                            <div key={i}>
                                <div
                                    className="periods-track__band"
                                    style={{ left: `${left}%`, width: `${width}%` }}
                                    title={`Period ${i + 1}: ${fmt(p.start)} – ${fmt(p.end)}`}
                                />
                                <div
                                    className={`periods-track__handle ${dragging?.idx === i && dragging.edge === 'start' ? 'is-dragging' : ''}`}
                                    style={{ left: `${left}%`, marginLeft: `-${HANDLE_PX}px` }}
                                    onMouseDown={onTrackPointerDown(i, 'start')}
                                    onTouchStart={onTrackPointerDown(i, 'start')}
                                />
                                <div
                                    className={`periods-track__handle ${dragging?.idx === i && dragging.edge === 'end' ? 'is-dragging' : ''}`}
                                    style={{ left: `${left + width}%`, marginLeft: `-${HANDLE_PX}px` }}
                                    onMouseDown={onTrackPointerDown(i, 'end')}
                                    onTouchStart={onTrackPointerDown(i, 'end')}
                                />
                            </div>
                        );
                    })}
                </div>

                <div className="periods-track__time-labels">
                    <span>{fmt(0)}</span>
                    <span>{fmt(duration / 2)}</span>
                    <span>{fmt(duration)}</span>
                </div>

                <div className="periods-list">
                    {periods.map((p, i) => (
                        <div key={i} className="periods-list__row">
                            <span className="periods-list__label">
                                Period {i + 1}{periods.length > 1 ? ` of ${periods.length}` : ''}
                            </span>
                            <PeriodInput
                                value={p.start}
                                onChange={(s) => setEdge(i, 'start', s)}
                                max={duration}
                                label="Start"
                            />
                            <span style={{ color: '#64748b' }}>→</span>
                            <PeriodInput
                                value={p.end}
                                onChange={(s) => setEdge(i, 'end', s)}
                                max={duration}
                                label="End"
                            />
                            <span className="periods-list__dur">
                                = {fmt(p.end - p.start)}
                            </span>
                        </div>
                    ))}
                </div>

                <div className="periods-actions">
                    <button
                        className="btn btn-secondary"
                        onClick={addBreak}
                        disabled={periods.length >= MAX_PERIODS}
                        title="Insert a skip gap inside the longest period"
                    >
                        <HiPlus /> Add break
                    </button>
                    <button
                        className="btn btn-secondary"
                        onClick={removeBreak}
                        disabled={periods.length <= 1}
                        title="Remove the last skip gap"
                    >
                        <HiMinus /> Remove break
                    </button>
                    <button
                        className="btn btn-ghost"
                        onClick={reset}
                        title="Back to one full-length period"
                    >
                        <HiArrowPath /> Reset
                    </button>
                </div>

                <div className="periods-summary">
                    <span>
                        ⏱ Match time:{' '}
                        <strong style={{ color: '#4ade80' }}>{fmt(totalMatchSec)}</strong>
                        {skippedSec > 0.5 && (
                            <span style={{ color: '#94a3b8' }}>
                                {' '}(skipping {fmt(skippedSec)})
                            </span>
                        )}
                        {periods.length > 1 && (
                            <span style={{ color: '#94a3b8' }}>
                                {' '}— {periods.length}× 4 = {periods.length * 4} player picks coming up
                            </span>
                        )}
                    </span>
                </div>

                <div className="periods-cta">
                    <button
                        className="btn btn-primary btn-lg"
                        onClick={handleContinue}
                        disabled={!duration}
                    >
                        Continue <HiArrowRight />
                    </button>
                </div>
            </div>
        </div>
    );
}

function PeriodInput({ value, onChange, max, label }) {
    const [draft, setDraft] = useState(fmt(value));
    useEffect(() => { setDraft(fmt(value)); }, [value]);
    const commit = () => {
        const parsed = parseTimecode(draft, max);
        if (parsed != null) onChange(parsed);
        else setDraft(fmt(value));
    };
    return (
        <input
            type="text"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={commit}
            onKeyDown={(e) => {
                if (e.key === 'Enter') {
                    e.currentTarget.blur();
                } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
                    e.preventDefault();
                    const delta = e.key === 'ArrowUp' ? 1 : -1;
                    const step = e.shiftKey ? 5 : 1;
                    onChange(Math.max(0, Math.min(max, value + delta * step)));
                }
            }}
            className="periods-list__input"
            placeholder="mm:ss"
            aria-label={label}
            title={label}
        />
    );
}
