import { useState, useEffect, useMemo, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import {
    HiHome, HiArrowPath, HiChartBar, HiBolt,
    HiUserGroup, HiMapPin, HiCheckCircle, HiExclamationCircle,
    HiArrowTrendingUp, HiPlayCircle, HiSparkles,
} from 'react-icons/hi2';
import {
    startAnalysis,
    startTracking,
    queueFeature,
    getSession,
    getSummary,
    listTasks,
    artifactUrl,
    subscribeSession,
} from '../services/api';
import { absUrl, API_KEY } from '../services/config';
import StepNav from '../components/StepNav';
import './Dashboard.css';

const FEATURES = [
    { key: 'heatmap',         label: 'Heatmap',          icon: HiMapPin,         color: '#e74c3c', type: 'image' },
    { key: 'speed_chart',     label: 'Speed & Distance', icon: HiBolt,           color: '#f39c12', type: 'image' },
    { key: 'possession',      label: 'Possession',       icon: HiChartBar,       color: '#3498db', type: 'image' },
    { key: 'minimap_replay',  label: 'Minimap Replay',   icon: HiArrowTrendingUp,color: '#00e59b', type: 'video' },
    { key: 'full_replay',     label: 'Annotated Replay', icon: HiPlayCircle,     color: '#6d28d9', type: 'video' },
    { key: 'sprint_analysis', label: 'Sprint Bursts',    icon: HiBolt,           color: '#9b59b6', type: 'image', comingSoon: true },
    { key: 'defensive_line',  label: 'Defensive Line',   icon: HiUserGroup,      color: '#00bcd4', type: 'image', comingSoon: true },
    { key: 'ai_summary',      label: 'AI Summary',       icon: HiSparkles,       color: '#ec4899', type: 'text' },
];

const PHASE_LABELS = {
    uploaded:        'Ready to analyze.',
    uploading:       'Waiting for upload to finish…',
    queued:          'Queued for analysis…',
    analyzing:       'Running analysis…',
    analysis_done:   'Analysis complete.',
    tracking:        'Tracking selected player (SAMURAI)…',
    tracking_done:   'Tracking complete — starting analysis…',
    analysis_failed: 'Analysis failed.',
    tracking_failed: 'Tracking failed.',
};

const STAGE_LABELS = {
    samurai_queued:    'Queued for SAMURAI…',
    extracting_frames: 'Extracting frames for SAMURAI…',
    samurai_running:   'SAMURAI tracking the selected player…',
    samurai_done:      'SAMURAI tracking finished.',
    loading_video:     'Loading video metadata…',
    yolo_detection:    'YOLO detection…',
    camera_motion:     'Camera motion compensation…',
    keypoint_detection:'Detecting field keypoints…',
    perspective:       'Perspective transform…',
    speed_calc:        'Computing speed & distance…',
    speed_calculation: 'Computing speed & distance…',
    team_colors:       'Resolving team colors…',
    team_assignment:   'Resolving team colors…',
    team_color_init:   'Resolving team colors…',
    team_voting:       'Assigning team colors…',
    possession_detection: 'Computing possession…',
    possession:        'Computing possession…',
    scene_segmentation:'Detecting scene segments…',
    computing_summary: 'Building summary…',
    summary:           'Building summary…',
    done:              'Analysis complete.',
    analysis_error:    'Analysis failed.',
};

const initialFeatures = Object.fromEntries(
    FEATURES.map((f) => [f.key, { status: 'locked', taskId: null, url: null, result: null, progress: 0 }])
);

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

export default function Dashboard() {
    const location = useLocation();
    const navigate = useNavigate();

    const query = new URLSearchParams(location.search);
    const sessionId = location.state?.sessionId || location.state?.videoId || query.get('sessionId');
    const selectedBbox = location.state?.selectedBbox || null;
    const playerName = location.state?.playerName || null;
    const startWithoutSelection = location.state?.startAnalysis === true;
    const isFreshAnalysis = Boolean(
        (selectedBbox && Array.isArray(selectedBbox) && selectedBbox.length === 4) ||
        startWithoutSelection
    );

    const [session, setSession] = useState(null);
    const [summary, setSummary] = useState(null);
    const [features, setFeatures] = useState(initialFeatures);
    const [error, setError] = useState(null);

    const analysisKicked = useRef(false);
    const summaryFetched = useRef(false);

    const phase = session?.status || 'uploaded';
    const progress = session?.progress ?? 0;
    const stage = session?.stage || '';

    const isAnalyzing = ['queued', 'analyzing', 'tracking', 'tracking_done'].includes(phase);
    const isDone = phase === 'analysis_done';
    const isFailed = phase === 'analysis_failed' || phase === 'tracking_failed';
    const hasGeneratingFeature = Object.values(features).some((f) => f.status === 'generating');

    const phaseLabel = PHASE_LABELS[phase] || STAGE_LABELS[stage] || stage || phase;
    const stageLabel = STAGE_LABELS[stage] || stage;

    useEffect(() => {
        setSession(null);
        setSummary(null);
        setFeatures(initialFeatures);
        setError(null);
        analysisKicked.current = false;
        summaryFetched.current = false;
    }, [sessionId]);

    // ── Kick off pipeline on mount ──────────────────────────────────────────
    // If we have a bbox from the Configure page, run SAMURAI → analysis (server
    // chains them). Otherwise fall back to plain global analysis (which will
    // currently fail because run_global_analysis depends on samurai output —
    // we surface the error so the user knows to pick a player).
    useEffect(() => {
        if (!sessionId) return;
        if (analysisKicked.current) return;
        analysisKicked.current = true;

        (async () => {
            try {
                if (selectedBbox && Array.isArray(selectedBbox) && selectedBbox.length === 4) {
                    const [x1, y1, x2, y2] = selectedBbox;
                    await startTracking(sessionId, { x1, y1, x2, y2 }, 0);
                    if (playerName) toast.success(`Tracking ${playerName}…`);
                } else if (startWithoutSelection) {
                    await startAnalysis(sessionId);
                }
            } catch (e) {
                const msg = e?.response?.data?.detail || e?.message || 'Failed to start analysis';
                setError(msg);
                toast.error(msg);
            }
        })();
    }, [sessionId, selectedBbox, playerName, startWithoutSelection]);

    // ── SSE stream for live session + task updates ──────────────────────────
    useEffect(() => {
        if (!sessionId) return;

        let cancelled = false;
        getSession(sessionId)
            .then((s) => {
                if (!cancelled) setSession(s);
            })
            .catch(() => {});

        if (!isFreshAnalysis) {
            listTasks(sessionId)
                .then((tasks = []) => {
                if (cancelled) return;
                setFeatures((prev) => {
                    const next = { ...prev };
                    for (const t of tasks) {
                        const key = t.task_type;
                        if (!(key in next)) continue;
                        next[key] = {
                            ...next[key],
                            taskId: t.task_id,
                            status:
                                t.status === 'done' ? 'done' :
                                t.status === 'failed' ? 'error' :
                                t.status === 'running' || t.status === 'queued' ? 'generating' :
                                next[key].status,
                            progress: t.progress || 0,
                            url: t.url ? taskResultUrl(sessionId, t.url) : next[key].url,
                            result: t.result ?? next[key].result,
                            error: t.error ?? next[key].error,
                        };
                    }
                    return next;
                });
            })
            .catch(() => {});
        }

        const unsub = subscribeSession(sessionId, {
            onSession: (s) => setSession((prev) => ({ ...prev, ...s })),
            onTask: (t) => {
                setFeatures((prev) => {
                    const key = t.task_type;
                    if (!(key in prev)) return prev;
                    const next = { ...prev };
                    next[key] = {
                        ...next[key],
                        taskId: t.task_id,
                        status:
                            t.status === 'done' ? 'done' :
                            t.status === 'failed' ? 'error' :
                            t.status === 'running' ? 'generating' : 'generating',
                        progress: t.progress || 0,
                        url: t.url ? taskResultUrl(sessionId, t.url) : next[key].url,
                        result: t.result ?? next[key].result,
                        error: t.error ?? next[key].error,
                    };
                    return next;
                });
            },
            onError: () => {
                // EventSource auto-reconnects; we just log and let it recover.
                console.warn('[SSE] disconnected, reconnecting…');
            },
        });
        return () => {
            cancelled = true;
            unsub();
        };
    }, [sessionId, isFreshAnalysis]);

    // Unlock feature buttons once analysis_done fires.
    useEffect(() => {
        if (!isDone) return;
        setFeatures((prev) => {
            const next = { ...prev };
            for (const k of Object.keys(next)) {
                if (next[k].status === 'locked') next[k] = { ...next[k], status: 'idle' };
            }
            return next;
        });
        if (!summaryFetched.current) {
            summaryFetched.current = true;
            getSummary(sessionId)
                .then((s) => setSummary(s?.session || s))
                .catch(() => {});
        }
    }, [isDone, sessionId]);

    const handleGenerate = async (key) => {
        setFeatures((prev) => ({
            ...prev,
            [key]: { ...prev[key], status: 'generating', progress: 0, error: null },
        }));
        try {
            await queueFeature(sessionId, key);
            // Live updates arrive via SSE; no polling needed.
        } catch (e) {
            const msg = e?.response?.data?.detail || e?.message || 'Request failed';
            setFeatures((prev) => ({
                ...prev,
                [key]: { ...prev[key], status: 'error', error: msg },
            }));
            toast.error(`${key}: ${msg}`);
        }
    };

    const handleDownload = (url, filename) => {
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.target = '_blank';
        link.rel = 'noopener';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const handleNewPlayer = () => {
        if (!sessionId) return;
        if (isAnalyzing || hasGeneratingFeature) {
            toast('Wait for the current analysis or replay generation to finish before choosing another player.');
            return;
        }
        navigate(`/configure?sessionId=${encodeURIComponent(sessionId)}`, {
            state: { videoId: sessionId, sessionId },
        });
    };

    const summaryCards = useMemo(() => {
        if (!summary) return [];
        return [
            summary.max_speed_kmh       != null && { label: 'Max Speed',   value: `${summary.max_speed_kmh} km/h`, icon: '⚡' },
            summary.avg_speed_kmh       != null && { label: 'Avg Speed',   value: `${summary.avg_speed_kmh} km/h`, icon: '🏃' },
            summary.total_distance_m    != null && { label: 'Distance',    value: `${summary.total_distance_m} m`, icon: '📏' },
            summary.possession_seconds  != null && { label: 'Possession',  value: `${summary.possession_seconds}s`, icon: '⚽' },
            summary.team1_possession_pct!= null && { label: 'Team 1 %',    value: `${summary.team1_possession_pct}%`, icon: '🔵' },
            summary.team2_possession_pct!= null && { label: 'Team 2 %',    value: `${summary.team2_possession_pct}%`, icon: '🔴' },
        ].filter(Boolean);
    }, [summary]);

    if (!sessionId) {
        return (
            <div className="dashboard">
                <div className="bg-grid" />
                <StepNav />
                <motion.div className="dashboard__error-banner" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <HiExclamationCircle /> No session. Upload a video first.
                </motion.div>
                <button className="btn btn-primary" onClick={() => navigate('/upload')}>
                    Go to Upload
                </button>
            </div>
        );
    }

    return (
        <div className="dashboard">
            <div className="bg-grid" />
            <StepNav />

            <motion.div className="dashboard__header" initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
                <div className="dashboard__header-left">
                    <div>
                        <h1 className="dashboard__title">Analysis Dashboard</h1>
                        <p className="dashboard__subtitle">Session {sessionId}</p>
                    </div>
                </div>
                <div className="dashboard__header-actions">
                    <button className="btn btn-ghost" onClick={() => navigate('/')}>
                        <HiHome /> Home
                    </button>
                </div>
            </motion.div>

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
                            <span className="pipeline-status__pct">{progress}%</span>
                        </div>
                        <div className="pipeline-status__bar-track">
                            <motion.div
                                className="pipeline-status__bar-fill"
                                animate={{ width: `${progress}%` }}
                                transition={{ ease: 'easeOut', duration: 0.5 }}
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

            <AnimatePresence>
                {summaryCards.length > 0 && (
                    <motion.div className="dashboard__summary-grid" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                        {summaryCards.map((s, i) => (
                            <motion.div
                                key={s.label}
                                className="summary-card"
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: i * 0.05 }}
                            >
                                <span className="summary-card__icon">{s.icon}</span>
                                <div>
                                    <p className="summary-card__value">{s.value}</p>
                                    <p className="summary-card__label">{s.label}</p>
                                </div>
                            </motion.div>
                        ))}
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Hero card: Annotated Replay is the showcase output, give it the
                full-width spot at the top of the dashboard. */}
            {(() => {
                const hero = FEATURES.find((f) => f.key === 'full_replay');
                if (!hero) return null;
                const state = features[hero.key];
                const Icon = hero.icon;
                return (
                    <motion.div
                        className="feature-card feature-card--hero"
                        initial={{ opacity: 0, scale: 0.96 }}
                        animate={{ opacity: 1, scale: 1 }}
                    >
                        <div
                            className="feature-card__header"
                            style={{ borderColor: hero.color }}
                        >
                            <Icon style={{ color: hero.color }} />
                            <span>{hero.label}</span>
                            {state.status === 'done' && <HiCheckCircle className="feature-card__done-icon" />}
                        </div>
                        <div className="feature-card__body">
                            {state.status === 'locked' && (
                                <p className="feature-card__hint">Replay will generate automatically after analysis.</p>
                            )}
                            {state.status === 'idle' && (
                                <p className="feature-card__hint">Replay is queued automatically and will appear here.</p>
                            )}
                            {state.status === 'generating' && (
                                <div className="feature-card__loading">
                                    <div className="feature-card__spinner" />
                                    <span>Automatically generating replay… {state.progress || 0}%</span>
                                </div>
                            )}
                            {state.status === 'done' && state.url && (
                                <video
                                    src={state.url}
                                    controls
                                    autoPlay
                                    muted
                                    loop
                                    className="feature-card__result-img feature-card__result-img--hero"
                                />
                            )}
                            {state.status === 'error' && (
                                <div className="feature-card__error-block">
                                    <p className="feature-card__error">❌ {state.error}</p>
                                    <button
                                        className="btn btn-primary feature-card__btn"
                                        onClick={() => handleGenerate(hero.key)}
                                    >
                                        Retry {hero.label}
                                    </button>
                                </div>
                            )}
                        </div>
                        {state.status === 'done' && state.url && (
                            <button
                                onClick={() => handleDownload(state.url, `${hero.key}_${sessionId}.mp4`)}
                                className="btn btn-ghost feature-card__download"
                                style={{ width: '100%', borderTop: '1px solid #333' }}
                            >
                                ↓ Download
                            </button>
                        )}
                    </motion.div>
                );
            })()}

            <div className="dashboard__features">
                {FEATURES.filter((f) => f.key !== 'full_replay').map((feat) => {
                    const state = features[feat.key];
                    const Icon = feat.icon;
                    return (
                        <motion.div
                            key={feat.key}
                            className="feature-card"
                            initial={{ opacity: 0, scale: 0.96 }}
                            animate={{ opacity: 1, scale: 1 }}
                        >
                            <div
                                className="feature-card__header"
                                style={{
                                    borderColor: feat.color,
                                    opacity: feat.comingSoon ? 0.55 : 1,
                                }}
                            >
                                <Icon style={{ color: feat.color }} />
                                <span>{feat.label}</span>
                                {state.status === 'done' && <HiCheckCircle className="feature-card__done-icon" />}
                            </div>
                            <div className="feature-card__body">
                                {feat.comingSoon ? (
                                    <div
                                        style={{
                                            padding: '1.5rem 1rem',
                                            textAlign: 'center',
                                            color: '#94a3b8',
                                            fontStyle: 'italic',
                                            background: 'rgba(15,23,42,0.5)',
                                            borderRadius: 8,
                                            border: '1px dashed #334155',
                                        }}
                                    >
                                        Feature in development
                                    </div>
                                ) : (
                                    <>
                                {state.status === 'locked' && (
                                    <p className="feature-card__hint">Waiting for analysis to finish…</p>
                                )}
                                {state.status === 'idle' && (
                                    <button
                                        className="btn btn-primary feature-card__btn"
                                        onClick={() => handleGenerate(feat.key)}
                                    >
                                        Generate {feat.label}
                                    </button>
                                )}
                                {state.status === 'generating' && (
                                    <div className="feature-card__loading">
                                        <div className="feature-card__spinner" />
                                        <span>Generating… {state.progress || 0}%</span>
                                    </div>
                                )}
                                {state.status === 'done' && feat.type === 'image' && state.url && (
                                    <img src={state.url} alt={feat.label} className="feature-card__result-img" />
                                )}
                                {state.status === 'done' && feat.type === 'video' && state.url && (
                                    <video
                                        src={state.url}
                                        controls
                                        autoPlay
                                        muted
                                        loop
                                        className="feature-card__result-img"
                                    />
                                )}
                                {state.status === 'done' && feat.type === 'text' && state.result && (
                                    <div
                                        style={{
                                            padding: '1rem',
                                            background: '#0b1220',
                                            borderRadius: 8,
                                            border: '1px solid #1e293b',
                                            color: '#cbd5e1',
                                            whiteSpace: 'pre-wrap',
                                            fontSize: 14,
                                            lineHeight: 1.6,
                                            maxHeight: 320,
                                            overflow: 'auto',
                                        }}
                                    >
                                        {taskTextResult(state.result) || JSON.stringify(state.result, null, 2)}
                                    </div>
                                )}
                                {state.status === 'error' && (
                                    <p className="feature-card__error">❌ {state.error}</p>
                                )}
                                    </>
                                )}
                            </div>

                            {!feat.comingSoon && state.status === 'done' && state.url && feat.type !== 'text' && (
                                <button
                                    onClick={() =>
                                        handleDownload(
                                            state.url,
                                            `${feat.key}_${sessionId}.${feat.type === 'video' ? 'mp4' : 'png'}`
                                        )
                                    }
                                    className="btn btn-ghost feature-card__download"
                                    style={{ width: '100%', borderTop: '1px solid #333', marginTop: '1rem', paddingTop: '1rem' }}
                                >
                                    ↓ Download
                                </button>
                            )}
                        </motion.div>
                    );
                })}
            </div>

            <motion.div className="dashboard__actions-footer" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
                <button className="btn btn-secondary" onClick={() => navigate('/')}>
                    <HiHome /> Back to Home
                </button>
                <button
                    className="btn btn-secondary"
                    onClick={handleNewPlayer}
                    disabled={isAnalyzing || hasGeneratingFeature}
                    title={
                        isAnalyzing || hasGeneratingFeature
                            ? 'Wait for the current analysis or replay generation to finish'
                            : 'Choose another player from this video'
                    }
                >
                    <HiUserGroup /> New Player
                </button>
                <button className="btn btn-primary" onClick={() => navigate('/upload')}>
                    <HiArrowPath /> New Video
                </button>
            </motion.div>
        </div>
    );
}
