import { useState, useEffect, useMemo, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { marked } from 'marked';
import toast from 'react-hot-toast';
import {
    HiHome, HiArrowPath, HiBars3, HiXMark, HiExclamationCircle,
    HiUserGroup, HiSparkles, HiChartBar, HiMapPin, HiFire,
    HiPlayCircle,
} from 'react-icons/hi2';
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
    PieChart, Pie, Cell,
} from 'recharts';
import {
    startAnalysis,
    startTracking,
    startTrackingMulti,
    queueFeature,
    getSession,
    getSummary,
    listTasks,
    artifactUrl,
    subscribeSession,
} from '../services/api';
import { absUrl, API_KEY } from '../services/config';
import StepNav from '../components/StepNav';
import VideoTimelineMarkers from '../components/VideoTimelineMarkers';
import MinimapOverlay from '../components/MinimapOverlay';
import HeatmapCanvas from '../components/HeatmapCanvas';
import './Dashboard.css';

const PHASE_LABELS = {
    uploaded: 'Ready to analyze.',
    uploading: 'Waiting for upload to finish…',
    queued: 'Queued for analysis…',
    analyzing: 'Running analysis…',
    analysis_done: 'Analysis complete.',
    tracking: 'Tracking selected player (SAMURAI)…',
    tracking_done: 'Tracking complete — starting analysis…',
    analysis_failed: 'Analysis failed.',
    tracking_failed: 'Tracking failed.',
};

const STAGE_LABELS = {
    samurai_queued: 'Queued for SAMURAI…',
    extracting_frames: 'Extracting frames for SAMURAI…',
    samurai_running: 'SAMURAI tracking the selected player…',
    samurai_done: 'SAMURAI tracking finished.',
    loading_video: 'Loading video metadata…',
    yolo_detection: 'YOLO detection…',
    camera_motion: 'Camera motion compensation…',
    keypoint_detection: 'Detecting field keypoints…',
    perspective: 'Perspective transform…',
    speed_calc: 'Computing speed & distance…',
    speed_calculation: 'Computing speed & distance…',
    team_colors: 'Resolving team colors…',
    team_assignment: 'Resolving team colors…',
    team_color_init: 'Resolving team colors…',
    team_voting: 'Assigning team colors…',
    possession_detection: 'Computing possession…',
    possession: 'Computing possession…',
    scene_segmentation: 'Detecting scene segments…',
    computing_summary: 'Building summary…',
    summary: 'Building summary…',
    done: 'Analysis complete.',
    analysis_error: 'Analysis failed.',
};

const taskResultUrl = (sessionId, rawUrl) => {
    if (!rawUrl) return null;
    if (/^https?:\/\//i.test(rawUrl)) return rawUrl;
    if (rawUrl.startsWith('/api/sessions/')) {
        const full = absUrl(rawUrl);
        return API_KEY ? `${full}${full.includes('?') ? '&' : '?'}key=${encodeURIComponent(API_KEY)}` : full;
    }
    return artifactUrl(sessionId, rawUrl.replace(/^\//, ''));
};

const taskTextResult = (result) => {
    if (!result) return '';
    if (typeof result === 'string') return result;
    return result.report_markdown || result.summary || '';
};

// ── Helpers for the data-analysis panel ────────────────────────────────────
const StatRow = ({ icon, label, value, sub }) => (
    <div className="stat-row">
        <span className="stat-row__icon">{icon}</span>
        <div className="stat-row__main">
            <div className="stat-row__value">{value}</div>
            <div className="stat-row__label">{label}{sub ? <span className="stat-row__sub"> · {sub}</span> : null}</div>
        </div>
    </div>
);

const PossessionBar = ({ team1, team2 }) => {
    const t1 = Math.max(0, Math.min(100, team1 ?? 0));
    const t2 = Math.max(0, Math.min(100, team2 ?? 0));
    return (
        <div className="poss-bar">
            <div className="poss-bar__header">
                <span><span className="poss-dot poss-dot--t1" /> Team 1 · {t1.toFixed(1)}%</span>
                <span><span className="poss-dot poss-dot--t2" /> Team 2 · {t2.toFixed(1)}%</span>
            </div>
            <div className="poss-bar__track">
                <div className="poss-bar__fill poss-bar__fill--t1" style={{ width: `${t1}%` }} />
                <div className="poss-bar__fill poss-bar__fill--t2" style={{ width: `${t2}%` }} />
            </div>
        </div>
    );
};

const DataAnalysisPanel = ({ playerSummary }) => {
    if (!playerSummary) {
        return <p className="drawer__empty">Stats will appear once analysis finishes.</p>;
    }
    const overall = playerSummary.overall || playerSummary;
    const segments = playerSummary.by_segment || [];

    const t1 = Number(overall.team1_possession_pct ?? 0);
    const t2 = Number(overall.team2_possession_pct ?? 0);
    const possessionData = [
        { name: 'Team 1', value: t1, fill: '#3498db' },
        { name: 'Team 2', value: t2, fill: '#e74c3c' },
    ];

    const speedData = [
        { name: 'Avg', value: Number(overall.avg_speed_kmh ?? 0), fill: '#60a5fa' },
        { name: 'Max', value: Number(overall.max_speed_kmh ?? 0), fill: '#f59e0b' },
    ];

    const periodData = segments.map((seg, i) => ({
        name: (seg.segment_type || `Seg ${i + 1}`).replace('_', ' '),
        distance: Number(seg.total_distance_m ?? 0),
        avg: Number(seg.avg_speed_kmh ?? 0),
        max: Number(seg.max_speed_kmh ?? 0),
    }));

    return (
        <div className="drawer__section-body">
            <div className="stat-grid">
                <StatRow icon="⚡" label="Max Speed" value={`${overall.max_speed_kmh ?? '-'} km/h`} />
                <StatRow icon="🏃" label="Avg Speed" value={`${overall.avg_speed_kmh ?? '-'} km/h`} />
                <StatRow icon="📏" label="Distance" value={`${overall.total_distance_m ?? '-'} m`} />
                <StatRow icon="⚽" label="Possession" value={`${overall.possession_seconds ?? '-'} s`} />
                <StatRow icon="🔄" label="Switches" value={overall.possession_switches ?? '-'} />
            </div>

            <h4 className="drawer__subhead">Speed (km/h)</h4>
            <div className="chart-wrap" style={{ height: 140 }}>
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={speedData} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                        <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
                        <YAxis stroke="#94a3b8" fontSize={11} />
                        <Tooltip contentStyle={chartTooltipStyle} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
                        <Bar dataKey="value" radius={[6, 6, 0, 0]} />
                    </BarChart>
                </ResponsiveContainer>
            </div>

            <h4 className="drawer__subhead">Team Possession</h4>
            <div className="poss-row">
                <div className="chart-wrap chart-wrap--donut">
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={possessionData}
                                innerRadius={32}
                                outerRadius={56}
                                paddingAngle={2}
                                dataKey="value"
                                stroke="none"
                            >
                                {possessionData.map((entry, i) => (
                                    <Cell key={i} fill={entry.fill} />
                                ))}
                            </Pie>
                            <Tooltip
                                contentStyle={chartTooltipStyle}
                                formatter={(v) => `${Number(v).toFixed(1)}%`}
                            />
                        </PieChart>
                    </ResponsiveContainer>
                </div>
                <div className="poss-row__legend">
                    <div><span className="poss-dot poss-dot--t1" /> Team 1 <strong>{t1.toFixed(1)}%</strong></div>
                    <div><span className="poss-dot poss-dot--t2" /> Team 2 <strong>{t2.toFixed(1)}%</strong></div>
                </div>
            </div>
            <PossessionBar team1={t1} team2={t2} />

            {periodData.length > 0 && (
                <>
                    <h4 className="drawer__subhead">By Period — Distance (m)</h4>
                    <div className="chart-wrap" style={{ height: 140 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={periodData} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                                <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} />
                                <YAxis stroke="#94a3b8" fontSize={11} />
                                <Tooltip contentStyle={chartTooltipStyle} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
                                <Bar dataKey="distance" fill="#22d3ee" radius={[6, 6, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                    <table className="seg-table">
                        <thead>
                            <tr><th>Period</th><th>Dist</th><th>Avg</th><th>Max</th></tr>
                        </thead>
                        <tbody>
                            {periodData.map((seg, i) => (
                                <tr key={i}>
                                    <td>{seg.name}</td>
                                    <td>{seg.distance} m</td>
                                    <td>{seg.avg} km/h</td>
                                    <td>{seg.max} km/h</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </>
            )}
        </div>
    );
};

const chartTooltipStyle = {
    background: 'rgba(15, 23, 42, 0.95)',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    borderRadius: 8,
    fontSize: 12,
    color: '#e2e8f0',
};

export default function Dashboard() {
    const location = useLocation();
    const navigate = useNavigate();

    const query = new URLSearchParams(location.search);
    const sessionId = location.state?.sessionId || location.state?.videoId || query.get('sessionId');
    const selectedBbox = location.state?.selectedBbox || null;
    const multiSegments = location.state?.multiSegments || null;
    const playerName = location.state?.playerName || null;
    const startWithoutSelection = location.state?.startAnalysis === true;
    const isFreshAnalysis = Boolean(
        (selectedBbox && Array.isArray(selectedBbox) && selectedBbox.length === 4) ||
        (multiSegments && multiSegments.length > 0) ||
        startWithoutSelection
    );

    const [session, setSession] = useState(null);
    const [aiSummary, setAiSummary] = useState(null);
    const [fullReplay, setFullReplay] = useState({ status: 'locked', url: null, progress: 0, error: null });
    const [error, setError] = useState(null);

    const [drawerOpen, setDrawerOpen] = useState(false);
    const [minimapOn, setMinimapOn] = useState(false);
    const [aiGenerating, setAiGenerating] = useState(false);

    const analysisKicked = useRef(false);
    const summaryFetched = useRef(false);
    const heroVideoRef = useRef(null);
    const realtimeEvents = useRef(0);

    const phase = session?.status || 'uploaded';
    const progress = session?.progress ?? 0;
    const stage = session?.stage || '';

    const isAnalyzing = ['queued', 'analyzing', 'tracking', 'tracking_done'].includes(phase);
    const isDone = phase === 'analysis_done';
    const isFailed = phase === 'analysis_failed' || phase === 'tracking_failed';

    const isColdStart = isAnalyzing && progress < 5 && !stage;
    const [coldStartSec, setColdStartSec] = useState(0);
    useEffect(() => {
        if (!isColdStart) { setColdStartSec(0); return; }
        const t0 = Date.now();
        const id = setInterval(() => setColdStartSec(Math.floor((Date.now() - t0) / 1000)), 1000);
        return () => clearInterval(id);
    }, [isColdStart]);

    const phaseLabel = isColdStart
        ? `Warming up GPU… (cold start ~30s, elapsed ${coldStartSec}s)`
        : PHASE_LABELS[phase] || STAGE_LABELS[stage] || stage || phase;
    const stageLabel = STAGE_LABELS[stage] || stage;

    // Smoothed progress
    const [smoothProgress, setSmoothProgress] = useState(0);
    useEffect(() => {
        if (!isAnalyzing) { setSmoothProgress(isDone ? 100 : 0); return; }
        const id = setInterval(() => {
            setSmoothProgress((prev) => {
                const target = progress;
                const ceiling = Math.min(target + 5, 99);
                if (prev < target) return Math.min(target, prev + Math.max(1, (target - prev) * 0.3));
                if (prev < ceiling) return Math.min(ceiling, prev + 0.3);
                return prev;
            });
        }, 200);
        return () => clearInterval(id);
    }, [isAnalyzing, isDone, progress]);
    const displayProgress = Math.round(smoothProgress);

    // Reset on sessionId change
    useEffect(() => {
        setSession(null);
        setAiSummary(null);
        setFullReplay({ status: 'locked', url: null, progress: 0, error: null });
        setError(null);
        setMinimapOn(false);
        setAiGenerating(false);
        analysisKicked.current = false;
        summaryFetched.current = false;
    }, [sessionId]);

    // Kick off pipeline on mount
    useEffect(() => {
        if (!sessionId) return;
        if (analysisKicked.current) return;
        analysisKicked.current = true;
        (async () => {
            try {
                if (multiSegments && multiSegments.length > 0) {
                    // Multi-segment path: bboxes already in {x1,y1,x2,y2} form
                    const segments = multiSegments.map((seg) => ({
                        frame: seg.frame,
                        bbox: seg.bbox,
                    }));
                    await startTrackingMulti(sessionId, segments);
                    toast.success(`Tracking across ${segments.length} segments in parallel…`);
                } else if (selectedBbox && Array.isArray(selectedBbox) && selectedBbox.length === 4) {
                    const [x1, y1, x2, y2] = selectedBbox;
                    await startTracking(sessionId, { x1, y1, x2, y2 }, 0);
                    if (playerName) toast.success(`Tracking ${playerName}…`);
                } else if (startWithoutSelection) {
                    await startAnalysis(sessionId);
                }
            } catch (e) {
                const msg = e?.response?.data?.detail || e?.message || 'Failed to start analysis';
                setError(msg); toast.error(msg);
            }
        })();
    }, [sessionId, selectedBbox, multiSegments, playerName, startWithoutSelection]);

    // Subscribe to live updates + initial fetch + polling fallback
    useEffect(() => {
        if (!sessionId) return;
        let cancelled = false;

        const applyTasks = (tasks = []) => {
            for (const t of tasks) {
                if (t.task_type === 'full_replay') {
                    setFullReplay({
                        status: t.status === 'done' ? 'done' : t.status === 'failed' ? 'error' : 'generating',
                        url: t.url ? taskResultUrl(sessionId, t.url) : null,
                        progress: t.progress || 0,
                        error: t.error || null,
                    });
                }
                if (t.task_type === 'ai_summary') {
                    setAiSummary(t.result || null);
                }
            }
        };

        getSession(sessionId).then((s) => { if (!cancelled) setSession(s); }).catch(() => { });
        if (!isFreshAnalysis) {
            listTasks(sessionId).then(applyTasks).catch(() => { });
        }

        realtimeEvents.current = 0;
        const pollInterval = setInterval(() => {
            if (realtimeEvents.current >= 2) { clearInterval(pollInterval); return; }
            getSession(sessionId).then((s) => {
                if (cancelled || !s) return;
                setSession(s);
                if (['analysis_done', 'analysis_failed', 'tracking_failed'].includes(s.status)) {
                    clearInterval(pollInterval);
                }
            }).catch(() => { });
            listTasks(sessionId).then(applyTasks).catch(() => { });
        }, 2000);

        const unsub = subscribeSession(sessionId, {
            onSession: (s) => { realtimeEvents.current += 1; setSession((prev) => ({ ...prev, ...s })); },
            onTask: (t) => {
                realtimeEvents.current += 1;
                if (t.task_type === 'full_replay') {
                    setFullReplay({
                        status: t.status === 'done' ? 'done' : t.status === 'failed' ? 'error' : 'generating',
                        url: t.url ? taskResultUrl(sessionId, t.url) : null,
                        progress: t.progress || 0,
                        error: t.error || null,
                    });
                }
                if (t.task_type === 'ai_summary') setAiSummary(t.result || null);
            },
        });

        return () => { cancelled = true; clearInterval(pollInterval); unsub(); };
    }, [sessionId, isFreshAnalysis]);

    // Fetch summary once analysis_done
    useEffect(() => {
        if (!isDone || summaryFetched.current) return;
        summaryFetched.current = true;
        getSummary(sessionId).then((s) => {
            if (s) setAiSummary((prev) => prev || s);
        }).catch(() => { });
    }, [isDone, sessionId]);

    const minimapDataUrl = session?.minimap_data_url || null;
    const heatmapDataUrl = session?.heatmap_data_url || null;

    const playerSummary = session?.player_summary || null;

    const aiMarkdown = useMemo(() => {
        const txt = taskTextResult(aiSummary);
        if (!txt) return '';
        try { return marked.parse(txt); } catch { return txt; }
    }, [aiSummary]);

    const handleGenerateAI = async () => {
        if (aiGenerating || !sessionId) return;
        setAiGenerating(true);
        try {
            await queueFeature(sessionId, 'ai_summary');
            toast.success('AI summary queued — this takes ~1 minute.');
        } catch (e) {
            toast.error(e?.message || 'Failed to queue AI summary');
            setAiGenerating(false);
        }
    };

    // Clear the local "generating" flag when the result actually arrives
    useEffect(() => {
        if (aiSummary) setAiGenerating(false);
    }, [aiSummary]);

    // (We used to lock body scroll while the drawer was open — turns out
    // that caused the drawer itself to feel frozen in some browser combos.
    // The drawer now uses overscroll-behavior: contain + its own scroll
    // container, which is enough to keep the wheel inside the drawer.)

    const handleNewPlayer = () => {
        if (!sessionId) return;
        if (isAnalyzing) {
            toast('Wait for the current analysis to finish.');
            return;
        }
        navigate(`/configure?sessionId=${encodeURIComponent(sessionId)}`, {
            state: { videoId: sessionId, sessionId },
        });
    };

    if (!sessionId) {
        return (
            <div className="dashboard dashboard--v2">
                <div className="bg-grid" />
                <StepNav />
                <div className="dashboard__error-banner">
                    <HiExclamationCircle /> No session. Upload a video first.
                </div>
                <button className="btn btn-primary" onClick={() => navigate('/upload')}>Go to Upload</button>
            </div>
        );
    }

    return (
        <div className="dashboard dashboard--v2">
            <div className="bg-grid" />

            {/* Top bar */}
            <div className="dashboard-v2__topbar">
                <button className="btn btn-ghost" onClick={() => navigate('/')}>
                    <HiHome /> Home
                </button>
                <div className="dashboard-v2__title">
                    <span className="dashboard-v2__title-main">Analysis</span>
                    <span className="dashboard-v2__title-sub">Session {sessionId.slice(0, 8)}…</span>
                </div>
                <button
                    className={`dashboard-v2__hamburger ${drawerOpen ? 'is-active' : ''}`}
                    onClick={() => setDrawerOpen((v) => !v)}
                    aria-label="Toggle analysis panel"
                >
                    {drawerOpen ? <HiXMark /> : <HiBars3 />}
                </button>
            </div>

            {/* Pipeline progress / errors */}
            <AnimatePresence>
                {(isAnalyzing || !session) && !isFailed && (
                    <motion.div
                        className="dashboard__pipeline-status"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                    >
                        <div className="pipeline-status__label">
                            <span>{phaseLabel}</span>
                            <span className="pipeline-status__pct">{displayProgress}%</span>
                        </div>
                        <div className="pipeline-status__bar-track">
                            <motion.div
                                className="pipeline-status__bar-fill"
                                animate={{ width: `${displayProgress}%` }}
                                transition={{ ease: 'linear', duration: 0.2 }}
                            />
                        </div>
                        {stage && <p className="pipeline-status__stage">{stageLabel}</p>}
                    </motion.div>
                )}
                {(isFailed || error) && (
                    <motion.div className="dashboard__error-banner" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                        <HiExclamationCircle /> {error || session?.error || 'Pipeline failed. Check server logs.'}
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Centerpiece: the video */}
            <main className={`dashboard-v2__stage ${drawerOpen ? 'drawer-open' : ''}`}>
              <div className="dashboard-v2__stage-inner">
                <motion.div
                    className="hero-video-card"
                    initial={{ opacity: 0, scale: 0.97 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.4 }}
                >
                    <div className="hero-video-card__header">
                        <HiPlayCircle /> <span>Annotated Replay</span>
                        {fullReplay.status === 'generating' && (
                            <span className="hero-video-card__pill">Generating · {fullReplay.progress}%</span>
                        )}
                    </div>

                    <div className="hero-video-card__body">
                        {fullReplay.status === 'done' && fullReplay.url ? (
                            <div className="hero-video-card__player-wrap">
                                <video
                                    ref={heroVideoRef}
                                    src={fullReplay.url}
                                    controls
                                    autoPlay
                                    muted
                                    loop
                                    className="hero-video-card__player"
                                />
                                <MinimapOverlay
                                    dataUrl={minimapDataUrl}
                                    videoRef={heroVideoRef}
                                    visible={minimapOn}
                                />
                            </div>
                        ) : fullReplay.status === 'generating' ? (
                            <div className="hero-video-card__placeholder">
                                <div className="feature-card__spinner" />
                                <p>Rendering replay… {fullReplay.progress}%</p>
                            </div>
                        ) : fullReplay.status === 'error' ? (
                            <div className="hero-video-card__placeholder">
                                <p className="feature-card__error">❌ {fullReplay.error || 'Replay generation failed'}</p>
                            </div>
                        ) : (
                            <div className="hero-video-card__placeholder">
                                <HiPlayCircle style={{ fontSize: 48, opacity: 0.4 }} />
                                <p>{isAnalyzing ? 'Replay will appear here once analysis finishes…' : 'No replay yet.'}</p>
                            </div>
                        )}

                        {fullReplay.status === 'done' && (
                            <VideoTimelineMarkers
                                segments={session?.segments}
                                fps={session?.video_fps}
                                totalFrames={session?.total_frames}
                                videoRef={heroVideoRef}
                            />
                        )}
                    </div>
                </motion.div>
                <StepNav />
              </div>
            </main>

            {/* Side drawer */}
            <AnimatePresence>
                {drawerOpen && (
                    <motion.aside
                        className="dashboard-v2__drawer"
                        initial={{ x: '100%' }}
                        animate={{ x: 0 }}
                        exit={{ x: '100%' }}
                        transition={{ type: 'tween', duration: 0.28, ease: 'easeOut' }}
                    >
                        <div className="drawer__list">
                            {/* Minimap toggle (stays as on/off switch) */}
                            <div className={`drawer__item ${minimapOn ? 'is-active' : ''}`}>
                                <button
                                    className="drawer__item-head"
                                    onClick={() => setMinimapOn((v) => !v)}
                                >
                                    <HiMapPin />
                                    <span>Minimap Overlay</span>
                                    <span className={`drawer__toggle ${minimapOn ? 'on' : ''}`}>
                                        {minimapOn ? 'ON' : 'OFF'}
                                    </span>
                                </button>
                            </div>

                            {/* Data Analysis — always rendered */}
                            <div className="drawer__item is-static">
                                <div className="drawer__item-head drawer__item-head--static">
                                    <HiChartBar />
                                    <span>Data Analysis</span>
                                </div>
                                <DataAnalysisPanel playerSummary={playerSummary} />
                            </div>

                            {/* AI Analysis — generate-on-demand */}
                            <div className="drawer__item is-static">
                                <div className="drawer__item-head drawer__item-head--static">
                                    <HiSparkles />
                                    <span>AI Analysis</span>
                                </div>
                                <div className="drawer__section-body">
                                    {aiMarkdown ? (
                                        <div
                                            className="markdown-body"
                                            dangerouslySetInnerHTML={{ __html: aiMarkdown }}
                                        />
                                    ) : aiGenerating ? (
                                        <div className="drawer__loading">
                                            <div className="feature-card__spinner" />
                                            <span>Generating AI summary… this takes ~1 minute</span>
                                        </div>
                                    ) : isDone ? (
                                        <div className="drawer__empty-cta">
                                            <p>Generate an AI tactical breakdown of this clip.</p>
                                            <button className="btn btn-primary" onClick={handleGenerateAI}>
                                                <HiSparkles /> Generate AI Summary
                                            </button>
                                        </div>
                                    ) : (
                                        <p className="drawer__empty">Waiting for analysis to finish…</p>
                                    )}
                                </div>
                            </div>

                            {/* Heatmap */}
                            <div className="drawer__item is-static">
                                <div className="drawer__item-head drawer__item-head--static">
                                    <HiFire />
                                    <span>Heatmap</span>
                                </div>
                                <div className="drawer__section-body">
                                    {heatmapDataUrl ? (
                                        <HeatmapCanvas dataUrl={heatmapDataUrl} />
                                    ) : isDone ? (
                                        <p className="drawer__empty">
                                            Heatmap data not exported for this session.
                                            Re-run analysis on the latest backend to enable.
                                        </p>
                                    ) : (
                                        <p className="drawer__empty">Waiting for analysis to finish…</p>
                                    )}
                                </div>
                            </div>
                        </div>

                        <div className="drawer__footer">
                            <button className="btn btn-secondary" onClick={handleNewPlayer} disabled={isAnalyzing}>
                                <HiUserGroup /> New Player
                            </button>
                            <button className="btn btn-primary" onClick={() => navigate('/upload')}>
                                <HiArrowPath /> New Video
                            </button>
                        </div>
                    </motion.aside>
                )}
            </AnimatePresence>
        </div>
    );
}
