import { useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import { HiArrowLeft, HiArrowRight, HiArrowPath, HiCheckCircle } from 'react-icons/hi2';
import { analyzeFrame, getSession } from '../services/api';
import StepNav from '../components/StepNav';
import './Configuration.css';

// Auto segment count per period. 4 is the sweet spot on 24GB GPUs:
//   1 period  × 4 = 4 picks total
//   2 periods × 4 = 8 picks (e.g. first + second half)
//   N periods × 4 = N*4 picks  (SAMURAI cap=8 means more queue up)
//
// 之前阈值是 20s — 30秒视频拆 2 段 (15s each) 反而每段只给 1 segment，用户
// 加了 break 后预期 4→8 结果变 1→2，反直觉。降到 5s：
//   - 5秒以下：1 segment（SAMURAI 在 100帧内开 4 进程并行 overhead 大于收益）
//   - 5秒以上：4 segments
// 30 秒视频 / 2 period × 15s = 各 4 segments = 8 picks，符合预期。
const SEGS_PER_PERIOD = 4;
const MIN_PERIOD_FOR_MULTI_SEG = 5;   // 秒
function segCountForPeriod(periodSec) {
    if (!Number.isFinite(periodSec) || periodSec < MIN_PERIOD_FOR_MULTI_SEG) return 1;
    return SEGS_PER_PERIOD;
}

/**
 * Distribute N keyframes across a (start, end) frame range.
 * For N=1 → just the start. For N>1 → evenly-spaced 0, 1/N, 2/N, ..., (N-1)/N.
 */
function keyframesIn(startFrame, endFrame, n) {
    if (n <= 1) return [startFrame];
    const span = endFrame - startFrame;
    return Array.from({ length: n }, (_, i) =>
        Math.floor(startFrame + (i / n) * span)
    );
}

export default function MultiSegmentConfig() {
    const navigate = useNavigate();
    const location = useLocation();
    const query = new URLSearchParams(location.search);
    const sessionId =
        location.state?.sessionId || location.state?.videoId || query.get('sessionId');

    // Match periods come from /trim → location.state.matchPeriods (in seconds),
    // or get restored from session.extra.match_periods_sec on direct navigation.
    const matchPeriodsFromNav = location.state?.matchPeriods || null;

    // Frame indices for the N segments (filled once session is loaded)
    const [frameIndices, setFrameIndices] = useState([]);
    // Cached video total — needed to recompute indices when user +/- segments
    const [totalFrames, setTotalFrames] = useState(0);
    // [{startFrame, endFrame}] in original-video frames
    const [periodsFrames, setPeriodsFrames] = useState([]);
    // Derived
    const segmentCount = frameIndices.length;
    // Per-segment state: { detecting, error, players, frameUrl, imgDims, selectedBbox }
    const [segments, setSegments] = useState([]);
    const [activeIdx, setActiveIdx] = useState(0);
    const [starting, setStarting] = useState(false);
    const detectedSegs = useRef(new Set());

    // 1. Compute keyframe indices.
    //    Sources for the match periods, in priority order:
    //      a. location.state.matchPeriods  (just came from /trim)
    //      b. session.extra.match_periods_sec  (saved by /trim earlier)
    //      c. [{start:0, end:fullDuration}]  (single-period default — same
    //         behaviour as before the Trim page existed)
    //    For each period we drop SEGS_PER_PERIOD evenly-spaced keyframes,
    //    or 1 keyframe if the period is < 20s (single-pick fallback).
    useEffect(() => {
        if (!sessionId) return;
        (async () => {
            try {
                const s = await getSession(sessionId);
                const fps = Number(s?.video_fps) || 25;
                let total = Number(s?.total_frames) || 0;
                let durationSec = total > 0 ? total / fps : 0;

                // Probe the <video> element if backend hasn't filled fps/total yet
                if (!total && s?.video_url) {
                    const probed = await new Promise((resolve) => {
                        const v = document.createElement('video');
                        v.preload = 'metadata';
                        v.src = s.video_url;
                        v.addEventListener('loadedmetadata', () => {
                            const d = v.duration || 0;
                            v.removeAttribute('src'); v.load();
                            resolve(d);
                        }, { once: true });
                        v.addEventListener('error', () => resolve(0), { once: true });
                    });
                    durationSec = probed;
                    total = Math.floor(probed * fps);
                }
                if (!total) { total = 1500; durationSec = 60; }
                setTotalFrames(total);

                // Resolve match periods (priority: nav state → DB → default)
                const periodsSec = matchPeriodsFromNav
                    || (Array.isArray(s?.match_periods_sec) ? s.match_periods_sec : null)
                    || [{ start: 0, end: durationSec }];

                // Convert sec → frames, clamp inside the video
                const periodsFr = periodsSec.map((p) => ({
                    startFrame: Math.max(0, Math.floor(p.start * fps)),
                    endFrame:   Math.min(total, Math.floor(p.end   * fps)),
                }));
                setPeriodsFrames(periodsFr);

                // Build the segment list — each period contributes up to
                // SEGS_PER_PERIOD keyframes, tagged with their periodIdx so
                // the nudge buttons can clamp inside the right range.
                const newSegments = [];
                const indices = [];
                periodsFr.forEach((pr, periodIdx) => {
                    const periodSec = (pr.endFrame - pr.startFrame) / fps;
                    const n = segCountForPeriod(periodSec);
                    keyframesIn(pr.startFrame, pr.endFrame, n).forEach((f) => {
                        indices.push(f);
                        newSegments.push({
                            frame: f,
                            periodIdx,
                            periodStartFrame: pr.startFrame,
                            periodEndFrame: pr.endFrame,
                            detecting: false,
                            error: null,
                            players: [],
                            frameUrl: null,
                            imgDims: null,
                            selectedBbox: null,
                        });
                    });
                });
                setFrameIndices(indices);
                setSegments(newSegments);
            } catch (e) {
                toast.error('Failed to load session: ' + e.message);
            }
        })();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [sessionId]);

    // 2. Lazy-load detection for the active segment. Re-runs whenever the
    // active segment's frame value changes (e.g. from a nudge button).
    // Use a per-(idx, frame) detection key so each unique frame is fetched
    // exactly once even with React StrictMode double-invoke.
    const activeSegFrame = segments[activeIdx]?.frame;
    useEffect(() => {
        if (segments.length === 0 || activeSegFrame == null) return;
        const detectKey = `${activeIdx}:${activeSegFrame}`;
        if (detectedSegs.current.has(detectKey)) return;
        detectedSegs.current.add(detectKey);

        setSegments((prev) => prev.map((s, i) =>
            i === activeIdx ? { ...s, detecting: true, error: null } : s
        ));

        analyzeFrame(sessionId, activeSegFrame)
            .then((data) => {
                const players = (data.players_data || []).map((p, i) => ({
                    id: p.id || i + 1,
                    bbox: p.bbox,
                }));
                setSegments((prev) => prev.map((s, i) =>
                    i === activeIdx
                        ? {
                            ...s,
                            detecting: false,
                            players,
                            frameUrl: data.annotated_frame_url,
                            imgDims: data.image_dimensions,
                            error: players.length === 0
                                ? 'No players detected here — nudge to a different frame'
                                : null,
                          }
                        : s
                ));
            })
            .catch((e) => {
                detectedSegs.current.delete(detectKey);  // allow retry
                setSegments((prev) => prev.map((s, i) =>
                    i === activeIdx
                        ? { ...s, detecting: false, error: e.message || 'Detection failed' }
                        : s
                ));
            });
    }, [activeIdx, activeSegFrame, segments.length, sessionId]);

    /**
     * Shift one segment's keyframe by ±N frames. Useful when the auto-split
     * lands on the player out-of-frame / occluded — user nudges a few frames
     * forward, hits Retry-detect (auto-triggered by frame change), picks again.
     */
    const nudgeFrame = (idx, delta) => {
        setSegments((prev) => prev.map((s, i) => {
            if (i !== idx) return s;
            // Clamp inside this segment's PERIOD (not the whole video).
            // Prevents user from nudging into a skipped halftime range.
            const lo = s.periodStartFrame ?? 0;
            const hi = (s.periodEndFrame ?? totalFrames) - 1;
            const newFrame = Math.max(lo, Math.min(hi, (s.frame || lo) + delta));
            if (newFrame === s.frame) return s;
            return {
                ...s,
                frame: newFrame,
                detecting: false,
                error: null,
                players: [],
                frameUrl: null,
                imgDims: null,
                selectedBbox: null,
                selectedPlayerId: null,
            };
        }));
    };

    const setSelectedFor = (idx, bbox, playerId) => {
        setSegments((prev) => prev.map((s, i) =>
            i === idx ? { ...s, selectedBbox: bbox, selectedPlayerId: playerId } : s
        ));
    };

    const allPicked = segments.length > 0 && segments.every((s) => s.selectedBbox);

    const startAnalysis = async () => {
        if (!sessionId || !allPicked) return;
        setStarting(true);
        // Tag each segment with its periodIdx so the backend knows which
        // (start_frame, end_frame) range it belongs to. Also send the
        // periods themselves so the backend can skip non-match frames.
        const payload = segments.map((s) => ({
            frame: s.frame,
            bbox: s.selectedBbox,
            period_idx: s.periodIdx ?? 0,
            img_dims: s.imgDims,
        }));
        navigate(`/dashboard?sessionId=${encodeURIComponent(sessionId)}`, {
            state: {
                sessionId,
                videoId: sessionId,
                multiSegments: payload,
                matchPeriodsFrames: periodsFrames,
                clientFps: session.video_fps || 25,
            },
        });
    };

    const active = segments[activeIdx];

    if (!sessionId) {
        return (
            <div className="page-container config-page">
                <div className="bg-grid" />
                <StepNav />
                <p style={{ padding: 24 }}>No session — go back to upload.</p>
                <button className="btn btn-primary" onClick={() => navigate('/upload')}>
                    Go to Upload
                </button>
            </div>
        );
    }

    return (
        <div className="page-container config-page">
            <div className="bg-grid" />

            <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                style={{ padding: '16px 24px' }}
            >
                <button
                    className="btn btn-ghost"
                    onClick={() => navigate(`/trim?sessionId=${encodeURIComponent(sessionId)}`, {
                        state: { sessionId, videoId: sessionId },
                    })}
                    disabled={starting}
                >
                    <HiArrowLeft /> Back to Match Periods
                </button>
            </motion.div>

            <StepNav />

            <motion.div
                className="config-page__header"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
            >
                <h1>
                    {segmentCount === 1 ? 'Pick a Player' : 'Multi-segment Tracking'}
                </h1>
                <p>
                    {segmentCount === 1
                        ? 'Short clip — one pick is enough.'
                        : `Pick the same player at ${segmentCount} points across the video. Each segment runs in parallel — much faster for long videos.`}
                </p>
            </motion.div>

            {/* Segment progress dots — count auto-decided from video length */}
            <div className="mseg__dots">
                {segments.map((s, i) => {
                    const done = !!s.selectedBbox;
                    const isActive = i === activeIdx;
                    return (
                        <button
                            key={i}
                            type="button"
                            className={`mseg__dot ${isActive ? 'is-active' : ''} ${done ? 'is-done' : ''}`}
                            onClick={() => setActiveIdx(i)}
                        >
                            {done ? <HiCheckCircle /> : i + 1}
                            <span className="mseg__dot-label">Seg {i + 1}</span>
                        </button>
                    );
                })}
            </div>

            <div className="config-split-view">
                <motion.div
                    className="config-frame-container"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    key={activeIdx}   // force fresh mount on segment change
                >
                    {!active ? (
                        <div className="config-frame-placeholder">Loading…</div>
                    ) : active.detecting ? (
                        <div className="config-frame-placeholder">
                            <div className="config-loading-spinner" />
                            <span>Detecting players at frame {active.frame}…</span>
                        </div>
                    ) : active.error ? (
                        <div className="config-frame-placeholder">
                            <span style={{ color: '#f87171' }}>{active.error}</span>
                            <button
                                className="btn btn-ghost"
                                style={{ marginTop: 12 }}
                                onClick={() => {
                                    detectedSegs.current.delete(activeIdx);
                                    setActiveIdx(activeIdx);  // re-trigger
                                }}
                            >
                                <HiArrowPath /> Retry
                            </button>
                        </div>
                    ) : active.frameUrl ? (
                        <div className="config-frame-wrapper">
                            <img
                                src={active.frameUrl}
                                alt={`Segment ${activeIdx + 1} frame`}
                                className="config-frame-img"
                                onError={() => toast.error('Failed to load frame image')}
                            />
                            {active.imgDims && (
                                <svg
                                    viewBox={`0 0 ${active.imgDims.width} ${active.imgDims.height}`}
                                    className="config-frame-svg"
                                    preserveAspectRatio="none"
                                >
                                    {active.players
                                        .filter((p) => p.bbox)
                                        .map((player) => {
                                            const [x1, y1, x2, y2] = player.bbox;
                                            const isSel = active.selectedPlayerId === player.id;
                                            return (
                                                <rect
                                                    key={player.id}
                                                    x={x1}
                                                    y={y1}
                                                    width={x2 - x1}
                                                    height={y2 - y1}
                                                    className={`config-bbox ${isSel ? 'config-bbox--selected' : ''}`}
                                                    onClick={() => setSelectedFor(activeIdx, player.bbox, player.id)}
                                                    rx="4"
                                                />
                                            );
                                        })}
                                </svg>
                            )}
                        </div>
                    ) : (
                        <div className="config-frame-placeholder">
                            <span>No frame data available</span>
                        </div>
                    )}
                </motion.div>

                <div className="config-sidebar">
                    <div className="mseg__sidebar-status">
                        <strong>Segment {activeIdx + 1} of {segmentCount}</strong>

                        {/* Frame nudge — only meaningful for multi-segment
                            mode where the auto-extracted frame may not show
                            the player. For single-segment mode at frame 0
                            we hide it to keep the UI clean. */}
                        {segmentCount > 1 && (
                            <>
                                <p className="mseg__nudge-hint">
                                    Player not in this frame? Nudge to a nearby moment:
                                </p>
                                <div className="mseg__frame-nudge">
                                    {/* Nudge bounds are the segment's PERIOD (not full video),
                                        so the user can never land on a frame inside a skipped halftime. */}
                                    <button
                                        type="button"
                                        className="mseg__nudge-btn"
                                        onClick={() => nudgeFrame(activeIdx, -30)}
                                        disabled={!active || active.frame - 30 < (active.periodStartFrame ?? 0)}
                                        title="−30 frames (≈1 second back)"
                                    >−30</button>
                                    <button
                                        type="button"
                                        className="mseg__nudge-btn"
                                        onClick={() => nudgeFrame(activeIdx, -5)}
                                        disabled={!active || active.frame - 5 < (active.periodStartFrame ?? 0)}
                                        title="−5 frames"
                                    >−5</button>
                                    <span className="mseg__frame-display">
                                        Frame {active?.frame ?? 0}
                                    </span>
                                    <button
                                        type="button"
                                        className="mseg__nudge-btn"
                                        onClick={() => nudgeFrame(activeIdx, 5)}
                                        disabled={!active || active.frame + 5 >= (active.periodEndFrame ?? totalFrames)}
                                        title="+5 frames"
                                    >+5</button>
                                    <button
                                        type="button"
                                        className="mseg__nudge-btn"
                                        onClick={() => nudgeFrame(activeIdx, 30)}
                                        disabled={!active || active.frame + 30 >= (active.periodEndFrame ?? totalFrames)}
                                        title="+30 frames (≈1 second forward)"
                                    >+30</button>
                                </div>
                            </>
                        )}

                        <p style={{ color: '#94a3b8', margin: '0.4rem 0 1rem', fontSize: '0.85rem' }}>
                            {active?.selectedBbox
                                ? '✓ Player chosen'
                                : 'Click the player on the left.'}
                        </p>
                    </div>

                    <div className="mseg__nav-row">
                        <button
                            className="btn btn-secondary"
                            disabled={activeIdx === 0}
                            onClick={() => setActiveIdx(activeIdx - 1)}
                        >
                            <HiArrowLeft /> Prev
                        </button>
                        <button
                            className="btn btn-secondary"
                            disabled={activeIdx === segmentCount - 1}
                            onClick={() => setActiveIdx(activeIdx + 1)}
                        >
                            Next <HiArrowRight />
                        </button>
                    </div>

                    <motion.div
                        className="config-cta"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.3 }}
                    >
                        <button
                            className={`btn btn-primary btn-lg w-full ${starting ? 'btn--loading' : ''}`}
                            onClick={startAnalysis}
                            disabled={!allPicked || starting}
                        >
                            {starting ? (
                                <><span className="config-spinner" />Starting…</>
                            ) : (
                                <>Start Analysis<HiArrowRight /></>
                            )}
                        </button>
                        <p className="config-cta__hint">
                            {allPicked
                                ? `All ${segmentCount} segments picked — ready`
                                : `${segments.filter((s) => s.selectedBbox).length}/${segmentCount} segments picked`}
                        </p>
                    </motion.div>
                </div>
            </div>
        </div>
    );
}
