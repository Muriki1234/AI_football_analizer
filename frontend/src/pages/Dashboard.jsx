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
    queueFeature,
    getSummary,
    artifactUrl,
    subscribeSession,
} from '../services/api';
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
    tracking:        'Running SAMURAI tracking…',
    tracking_done:   'Tracking complete — starting analysis…',
    analysis_failed: 'Analysis failed.',
    tracking_failed: 'Tracking failed.',
};

const initialFeatures = Object.fromEntries(
    FEATURES.map((f) => [f.key, { status: 'locked', taskId: null, url: null, result: null, progress: 0 }])
);

export default function Dashboard() {
    const location = useLocation();
    const navigate = useNavigate();

    const sessionId = location.state?.sessionId || location.state?.videoId;

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

    const phaseLabel = PHASE_LABELS[phase] || stage || phase;

    // ── Kick off analysis on mount ──────────────────────────────────────────
    useEffect(() => {
        if (!sessionId) return;
        if (analysisKicked.current) return;
        analysisKicked.current = true;

        (async () => {
            try {
                await startAnalysis(sessionId);
            } catch (e) {
                const msg = e?.response?.data?.detail || e?.message || 'Failed to start analysis';
                setError(msg);
                toast.error(msg);
            }
        })();
    }, [sessionId]);

    // ── SSE stream for live session + task updates ──────────────────────────
    useEffect(() => {
        if (!sessionId) return;

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
                        url: t.url ? artifactUrl(sessionId, (t.url || '').replace(/^\//, '')) : next[key].url,
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
        return unsub;
    }, [sessionId]);

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
                        {stage && <p className="pipeline-status__stage">{stage}</p>}
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

            <div className="dashboard__features">
                {FEATURES.map((feat) => {
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
                                        {typeof state.result === 'string'
                                            ? state.result
                                            : state.result.summary || JSON.stringify(state.result, null, 2)}
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
                <button className="btn btn-primary" onClick={() => navigate('/upload')}>
                    <HiArrowPath /> New Video
                </button>
            </motion.div>
        </div>
    );
}
