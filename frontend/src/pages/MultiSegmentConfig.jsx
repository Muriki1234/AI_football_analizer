import { useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import { HiArrowLeft, HiArrowRight, HiArrowPath, HiCheckCircle } from 'react-icons/hi2';
import { analyzeFrame, getSession } from '../services/api';
import StepNav from '../components/StepNav';
import './Configuration.css';

// Auto-suggest how many segments to use based on the clip's length. The
// trade-off: each segment costs a CUDA-context (~700MB GPU) + 1 click for
// the user. For short clips a single SAMURAI run is already fast, so multi
// just adds friction. For long matches we want as many segments as the
// 16GB GPU can comfortably parallelize.
function suggestedSegmentCount(durationSec) {
    if (!Number.isFinite(durationSec) || durationSec <= 0) return 4;
    if (durationSec < 60)   return 1;   // < 1 min  → single pick
    if (durationSec < 180)  return 2;   // 1-3 min  → 2-way split
    if (durationSec < 600)  return 3;   // 3-10 min → 3-way split
    return 4;                            // 10 min+  → 4-way split (max)
}

export default function MultiSegmentConfig() {
    const navigate = useNavigate();
    const location = useLocation();
    const query = new URLSearchParams(location.search);
    const sessionId =
        location.state?.sessionId || location.state?.videoId || query.get('sessionId');

    // segmentCount is decided after we know the video duration. Query param
    // overrides the auto-suggestion (e.g. ?segments=8 for power users).
    const segmentCountOverride = Number(query.get('segments') || 0);
    const [segmentCount, setSegmentCount] = useState(
        segmentCountOverride > 0 ? Math.min(8, segmentCountOverride) : 4
    );

    // Frame indices for the N segments (filled once session is loaded)
    const [frameIndices, setFrameIndices] = useState([]);
    // Per-segment state: { detecting, error, players, frameUrl, imgDims, selectedBbox }
    const [segments, setSegments] = useState([]);
    const [activeIdx, setActiveIdx] = useState(0);
    const [starting, setStarting] = useState(false);
    const detectedSegs = useRef(new Set());

    // 1. Compute keyframe indices from session metadata + auto-pick segment count
    useEffect(() => {
        if (!sessionId) return;
        (async () => {
            try {
                const s = await getSession(sessionId);
                const fps = Number(s?.video_fps) || 25;
                let total = Number(s?.total_frames) || 0;
                let durationSec = total > 0 ? total / fps : 0;

                // Fallback: probe the <video> element ourselves (faster than
                // waiting for the analysis pipeline to fill video_fps)
                if (!total && s?.video_url) {
                    const probed = await new Promise((resolve) => {
                        const v = document.createElement('video');
                        v.preload = 'metadata';
                        v.src = s.video_url;
                        v.addEventListener('loadedmetadata', () => {
                            const d = v.duration || 0;
                            v.removeAttribute('src');
                            v.load();
                            resolve(d);
                        }, { once: true });
                        v.addEventListener('error', () => resolve(0), { once: true });
                    });
                    durationSec = probed;
                    total = Math.floor(probed * fps);
                }
                if (!total) {
                    total = 1500;
                    durationSec = 60;
                }

                // Decide final segment count — unless URL forced an override
                const finalCount = segmentCountOverride > 0
                    ? Math.min(8, segmentCountOverride)
                    : suggestedSegmentCount(durationSec);
                if (finalCount !== segmentCount) setSegmentCount(finalCount);

                // Evenly-spaced keyframes: e.g. for N=4 → 0%, 25%, 50%, 75%
                const indices = Array.from({ length: finalCount }, (_, i) =>
                    Math.floor((i / finalCount) * total)
                );
                setFrameIndices(indices);
                setSegments(indices.map((f) => ({
                    frame: f,
                    detecting: false,
                    error: null,
                    players: [],
                    frameUrl: null,
                    imgDims: null,
                    selectedBbox: null,
                })));
            } catch (e) {
                toast.error('Failed to load session: ' + e.message);
            }
        })();
        // segmentCount is intentionally omitted — we set it inside this effect
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [sessionId, segmentCountOverride]);

    // 2. Lazy-load detection for the active segment on first visit
    useEffect(() => {
        if (segments.length === 0) return;
        if (detectedSegs.current.has(activeIdx)) return;
        detectedSegs.current.add(activeIdx);

        const frameIdx = frameIndices[activeIdx];
        setSegments((prev) => prev.map((s, i) =>
            i === activeIdx ? { ...s, detecting: true, error: null } : s
        ));

        analyzeFrame(sessionId, frameIdx)
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
                                ? 'No players detected here — try a different frame'
                                : null,
                          }
                        : s
                ));
            })
            .catch((e) => {
                detectedSegs.current.delete(activeIdx);  // allow retry
                setSegments((prev) => prev.map((s, i) =>
                    i === activeIdx
                        ? { ...s, detecting: false, error: e.message || 'Detection failed' }
                        : s
                ));
            });
    }, [activeIdx, segments.length, frameIndices, sessionId]);

    const setSelectedFor = (idx, bbox, playerId) => {
        setSegments((prev) => prev.map((s, i) =>
            i === idx ? { ...s, selectedBbox: bbox, selectedPlayerId: playerId } : s
        ));
    };

    const allPicked = segments.length > 0 && segments.every((s) => s.selectedBbox);

    const startAnalysis = async () => {
        if (!sessionId || !allPicked) return;
        setStarting(true);
        const payload = segments.map((s) => ({
            frame: s.frame,
            bbox: s.selectedBbox,
        }));
        // Dashboard will dispatch via startTrackingMulti when it sees multiSegments in state
        navigate(`/dashboard?sessionId=${encodeURIComponent(sessionId)}`, {
            state: {
                sessionId,
                videoId: sessionId,
                multiSegments: payload,
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
                    onClick={() => navigate(`/configure?sessionId=${encodeURIComponent(sessionId)}`, {
                        state: { sessionId, videoId: sessionId }
                    })}
                    disabled={starting}
                >
                    <HiArrowLeft /> Single-pick mode
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

            {/* Segment progress dots */}
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
                        <p style={{ color: '#94a3b8', margin: '0.4rem 0 1rem' }}>
                            Frame {active?.frame || 0}
                            {active?.selectedBbox
                                ? ' — ✓ Player chosen'
                                : ' — click a box on the left'}
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
