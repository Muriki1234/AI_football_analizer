import { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
    HiHome, HiArrowPath, HiChartBar, HiBolt,
    HiUserGroup, HiMapPin, HiCheckCircle, HiExclamationCircle,
    HiArrowTrendingUp
} from 'react-icons/hi2';
import {
    autoStart, pollSessionStatus, startGlobalAnalysis,
    generateFeature, pollTaskStatus, getSummary
} from '../services/api';
import colab from '../services/colabService';
import StepNav from '../components/StepNav';
import './Dashboard.css';

// Feature cards config
const FEATURES = [
    { key: 'heatmap', label: 'Heatmap', icon: HiMapPin, color: '#e74c3c', type: 'image' },
    { key: 'speed_chart', label: 'Speed & Distance', icon: HiBolt, color: '#f39c12', type: 'image' },
    { key: 'possession', label: 'Possession', icon: HiChartBar, color: '#3498db', type: 'image' },
    { key: 'minimap_replay', label: 'Minimap Replay', icon: HiArrowTrendingUp, color: '#00e59b', type: 'video' },
    { key: 'full_replay', label: 'Analyzed Replay', icon: HiCheckCircle, color: '#9b59b6', type: 'video' },
];

const PHASE_LABELS = {
    tracking: '⚙️ SAMURAI tracking player...',
    tracking_done: '✅ Tracking complete — starting analysis...',
    analyzing: '🤖 Running YOLO analysis...',
    analysis_done: '✅ Analysis complete!',
    tracking_failed: '❌ Tracking failed',
    analysis_failed: '❌ Analysis failed',
};

export default function Dashboard() {
    const location = useLocation();
    const navigate = useNavigate();

    const sessionId = location.state?.sessionId;
    const playerName = location.state?.playerName || 'Player';

    // Pipeline state
    const [phase, setPhase] = useState('tracking');
    const [progress, setProgress] = useState(0);
    const [stageLabel, setStageLabel] = useState('Initializing...');
    const [error, setError] = useState(null);
    const [summary, setSummary] = useState(null);
    const [features, setFeatures] = useState(
        Object.fromEntries(FEATURES.map(f => [f.key, { status: 'locked', taskId: null, url: null, result: null }]))
    );

    // Track if we've already kicked off analysis to avoid double start
    const analysisStarted = useRef(false);

    // Custom download handler to work with cross-origin Colab URLs and local Blobs
    const handleDownload = async (url, filename) => {
        try {
            const response = await fetch(url);
            const blob = await response.blob();
            const blobUrl = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = blobUrl;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(blobUrl);
        } catch (e) {
            console.error("Download failed cross-origin, standard link fallback:", e);
            window.open(url, '_blank');
        }
    };


    
    // Auto-download full_replay when done
    useEffect(() => {
        const replay = features['full_replay'];
        if (replay && replay.status === 'done' && replay.url && !replay.downloadedTriggered) {
             setFeatures(prev => ({ ...prev, full_replay: { ...prev.full_replay, downloadedTriggered: true } }));
             handleDownload(replay.url, `Analyzed_Replay_${sessionId}.mp4`);
        }
    }, [features['full_replay']?.status, features['full_replay']?.url]);

    const needsAutoStart = location.state?.autoStart;
    const autoStartFired = useRef(false);

    useEffect(() => {
        if (!sessionId) return;

        (async () => {
            try {
                // Step 0: Auto-start if coming straight from Upload (skip trim/configure)
                // Guard against React StrictMode double-invoke
                if (needsAutoStart && !autoStartFired.current) {
                    autoStartFired.current = true;
                    setPhase('tracking');
                    setStageLabel('Auto-detecting players...');
                    try {
                        await autoStart(sessionId);
                    } catch (e) {
                        // 400 = already started (e.g. StrictMode second invoke) — safe to ignore
                        if (e?.response?.status !== 400) throw e;
                    }
                }

                // Step 1: Wait for tracking_done
                const trackResult = await pollSessionStatus(
                    sessionId, 'tracking_done',
                    (data) => {
                        setPhase(data.status);
                        setProgress(data.progress || 0);
                        setStageLabel(data.stage_label || PHASE_LABELS[data.status] || '');
                    }
                );

                // Step 2: Auto-trigger global analysis
                if (!analysisStarted.current) {
                    analysisStarted.current = true;
                    setPhase('tracking_done');
                    await startGlobalAnalysis(sessionId);
                }

                // Step 3: Wait for analysis_done
                const analysisResult = await pollSessionStatus(
                    sessionId, 'analysis_done',
                    (data) => {
                        setPhase(data.status);
                        setProgress(data.progress || 0);
                        setStageLabel(data.stage_label || PHASE_LABELS[data.status] || '');

                        // Unlock feature buttons once available
                        if (data.available_features?.includes('heatmap')) {
                            setFeatures(prev => Object.fromEntries(
                                Object.entries(prev).map(([k, v]) => [k, { ...v, status: 'idle' }])
                            ));
                        }
                    }
                );

                // Step 4: Fetch summary
                const s = await getSummary(sessionId);
                setSummary(s);

            } catch (e) {
                setError(e.message || 'Pipeline failed');
                setPhase('error');
            }
        })();
    }, [sessionId]);

    const handleGenerateFeature = async (featureKey) => {
        setFeatures(prev => ({ ...prev, [featureKey]: { ...prev[featureKey], status: 'generating' } }));
        try {
            const { task_id } = await generateFeature(sessionId, featureKey);
            const task = await pollTaskStatus(
                sessionId, task_id,
                (t) => setFeatures(prev => ({ ...prev, [featureKey]: { ...prev[featureKey], status: 'generating', progress: t.progress } }))
            );
            setFeatures(prev => ({
                ...prev,
                [featureKey]: { status: 'done', taskId: task_id, url: task.url, result: task.result }
            }));
        } catch (e) {
            setFeatures(prev => ({
                ...prev, [featureKey]: { ...prev[featureKey], status: 'error', error: e.message }
            }));
        }
    };

    const isPipelineDone = phase === 'analysis_done';
    const isFailed = phase === 'tracking_failed' || phase === 'analysis_failed' || phase === 'error';

    return (
        <div className="dashboard">
            <div className="bg-grid" />
            <StepNav />

            {/* Header */}
            <motion.div className="dashboard__header" initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
                <div className="dashboard__header-left">
                    <div>
                        <h1 className="dashboard__title">Analysis Dashboard</h1>
                        <p className="dashboard__subtitle">Deep tracking — {playerName}</p>
                    </div>
                </div>
                <div className="dashboard__header-actions">
                    <button className="btn btn-ghost" onClick={() => navigate('/')}>
                        <HiHome /> Home
                    </button>
                </div>
            </motion.div>

            {/* Pipeline Progress */}
            <AnimatePresence>
                {!isPipelineDone && !isFailed && (
                    <motion.div
                        className="dashboard__pipeline-status"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                    >
                        <div className="pipeline-status__label">
                            <span>{PHASE_LABELS[phase] || stageLabel}</span>
                            <span className="pipeline-status__pct">{progress}%</span>
                        </div>
                        <div className="pipeline-status__bar-track">
                            <motion.div
                                className="pipeline-status__bar-fill"
                                animate={{ width: `${progress}%` }}
                                transition={{ ease: 'easeOut', duration: 0.5 }}
                            />
                        </div>
                        <p className="pipeline-status__stage">{stageLabel}</p>
                    </motion.div>
                )}

                {isFailed && (
                    <motion.div className="dashboard__error-banner" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                        <HiExclamationCircle /> {error || 'Pipeline encountered an error. Check backend logs.'}
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Summary Cards — show once analysis_done */}
            <AnimatePresence>
                {summary && (
                    <motion.div className="dashboard__summary-grid" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                        {[
                            { label: 'Max Speed', value: `${summary.max_speed_kmh} km/h`, icon: '⚡' },
                            { label: 'Avg Speed', value: `${summary.avg_speed_kmh} km/h`, icon: '🏃' },
                            { label: 'Distance', value: `${summary.total_distance_m} m`, icon: '📏' },
                            { label: 'Possession', value: `${summary.possession_seconds}s`, icon: '⚽' },
                            { label: 'Team 1 %', value: `${summary.team1_possession_pct}%`, icon: '🔵' },
                            { label: 'Team 2 %', value: `${summary.team2_possession_pct}%`, icon: '🔴' },
                        ].map((s, i) => (
                            <motion.div key={s.label} className="summary-card" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}>
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

            {/* On-demand Feature Buttons + Results */}
            <div className="dashboard__features">
                {FEATURES.map(feat => {
                    const fState = features[feat.key];
                    const Icon = feat.icon;
                    return (
                        <motion.div key={feat.key} className="feature-card" initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }}>
                            <div className="feature-card__header" style={{ borderColor: feat.color }}>
                                <Icon style={{ color: feat.color }} />
                                <span>{feat.label}</span>
                                {fState.status === 'done' && <HiCheckCircle className="feature-card__done-icon" />}
                            </div>

                            {/* Result area */}
                            <div className="feature-card__body">
                                {fState.status === 'locked' && (
                                    <p className="feature-card__hint">Waiting for analysis to complete...</p>
                                )}
                                {fState.status === 'idle' && (
                                    <button className="btn btn-primary feature-card__btn" onClick={() => handleGenerateFeature(feat.key)}>
                                        Generate {feat.label}
                                    </button>
                                )}
                                {fState.status === 'generating' && (
                                    <div className="feature-card__loading">
                                        <div className="feature-card__spinner" />
                                        <span>Generating... {fState.progress || 0}%</span>
                                    </div>
                                )}
                                {fState.status === 'done' && feat.type === 'image' && fState.url && (
                                    <img src={fState.url} alt={feat.label} className="feature-card__result-img" />
                                )}
                                {fState.status === 'done' && feat.type === 'video' && fState.url && (
                                    <video src={fState.url} controls autoPlay muted loop className="feature-card__result-img" />
                                )}
                                {fState.status === 'error' && feat.key === 'full_replay' && (
                                    <div>
                                        <p className="feature-card__error">❌ {fState.error}</p>
                                        <button
                                            className="btn btn-secondary feature-card__btn"
                                            style={{ marginTop: '0.5rem' }}
                                            onClick={() => {
                                                const base = colab.isConfigured() ? colab.getUrl() : '';
                                                handleDownload(
                                                    `${base}/api/${sessionId}/download_raw`,
                                                    `original_${sessionId}.mp4`
                                                );
                                            }}
                                        >
                                            ↓ 下载原始视频
                                        </button>
                                    </div>
                                )}
                                {fState.status === 'error' && feat.key !== 'full_replay' && (
                                    <p className="feature-card__error">❌ {fState.error}</p>
                                )}
                            </div>

                            {fState.status === 'done' && (
                                <button onClick={() => handleDownload(fState.url, `${feat.key}_${sessionId}.${feat.type === 'video' ? 'mp4' : 'png'}`)} className="btn btn-ghost feature-card__download" style={{width: '100%', borderTop: '1px solid #333', marginTop: '1rem', paddingTop: '1rem'}}>
                                    ↓ Download
                                </button>
                            )}
                        </motion.div>
                    );
                })}
            </div>

            {/* Footer */}
            <motion.div className="dashboard__actions-footer" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
                <button className="btn btn-secondary" onClick={() => navigate('/')}>
                    <HiHome /> Back to Home
                </button>
                <button className="btn btn-secondary" onClick={() => navigate('/configure')}>
                    <HiUserGroup /> Analyze New Player
                </button>
                <button className="btn btn-primary" onClick={() => navigate('/upload')}>
                    <HiArrowPath /> New Video
                </button>
            </motion.div>
        </div>
    );
}
