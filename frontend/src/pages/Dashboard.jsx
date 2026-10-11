import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import toast from 'react-hot-toast';
import {
    HiHome, HiArrowPath, HiBars3, HiXMark, HiExclamationCircle,
    HiUserGroup, HiSparkles, HiChartBar, HiMapPin, HiFire,
    HiPlayCircle, HiArrowDownTray, HiArrowsPointingOut,
    HiMagnifyingGlass, HiPlay, HiBolt,
} from 'react-icons/hi2';
import {
    startAnalysis,
    startTracking,
    startTrackingMulti,
    queueFeature,
    askCoachQA,
    getHighlightsManifest,
    getSession,
    getSummary,
    listSummaries,
    listTasks,
    artifactUrl,
    subscribeSession,
    saveTacticalDrawings
} from '../services/api';

import { absUrl, API_KEY } from '../services/config';
import StepNav from '../components/StepNav';
import VideoTimelineMarkers from '../components/VideoTimelineMarkers';
import CanvasOverlay from '../components/CanvasOverlay';
import MinimapOverlay from '../components/MinimapOverlay';
import HeatmapCanvas from '../components/HeatmapCanvas';
import TelestrationCanvas from '../components/TelestrationCanvas';
import DataAnalysisPanel from '../components/DataAnalysisPanel';
import PitchZoneAnalysisPanel from '../components/PitchZoneAnalysisPanel';
import { useLanguage } from '../i18n/LanguageContext';
import './Dashboard.css';

const TACTICAL_CHARTS_CONFIG = [
    { id: 'pass_network', filename: 'pass_network.png', feature: 'pass_network', labelKey: 'dashboard.charts.passNetwork', descKey: 'dashboard.charts.passNetworkDesc' },
    { id: 'spatial_radar', filename: 'spatial_radar.png', feature: 'spatial_radar', labelKey: 'dashboard.charts.spatialRadar', descKey: 'dashboard.charts.spatialRadarDesc' },
    { id: 'voronoi_pitch_control', filename: 'voronoi_pitch_control.png', feature: 'voronoi', labelKey: 'dashboard.charts.voronoiControl', descKey: 'dashboard.charts.voronoiControlDesc' },
    { id: 'defensive_line', filename: 'defensive_line.png', feature: 'defensive_line', labelKey: 'dashboard.charts.defensiveLine', descKey: 'dashboard.charts.defensiveLineDesc' },
    { id: 'turnover_transitions', filename: 'turnover_transitions.png', feature: 'turnovers', labelKey: 'dashboard.charts.transitions', descKey: 'dashboard.charts.transitionsDesc' },
    { id: 'team_compactness', filename: 'team_compactness.png', feature: 'team_compactness', labelKey: 'dashboard.charts.compactness', descKey: 'dashboard.charts.compactnessDesc' },
    { id: 'possession_chart', filename: 'possession_chart.png', feature: 'possession', labelKey: 'dashboard.charts.possession', descKey: 'dashboard.charts.possessionDesc' },
    { id: 'shot_xg', filename: 'shot_map.png', feature: 'shot_xg', labelKey: 'dashboard.charts.shotsXg', descKey: 'dashboard.charts.shotsXgDesc' },
    { id: 'pressing_intensity', filename: 'pressing_intensity.png', feature: 'pressing_intensity', labelKey: 'dashboard.charts.pressingIntensity', descKey: 'dashboard.charts.pressingIntensityDesc' },
    { id: 'tactical_dossier', filename: 'tactical_dossier.png', feature: 'tactical_dossier', labelKey: 'dashboard.charts.tacticalDossier', descKey: 'dashboard.charts.tacticalDossierDesc' },
];

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
export default function Dashboard() {
    const location = useLocation();
    const navigate = useNavigate();
    const { lang, t } = useLanguage();

    const tacticalCharts = useMemo(() => {
        return TACTICAL_CHARTS_CONFIG.map((c) => ({
            ...c,
            label: t(c.labelKey),
            desc: t(c.descKey),
        }));
    }, [t]);

    const query = new URLSearchParams(location.search);
    const sessionId = location.state?.sessionId || location.state?.videoId || query.get('sessionId');
    const selectedBbox = location.state?.selectedBbox || null;
    const multiSegments = location.state?.multiSegments || null;
    const matchPeriodsFrames = location.state?.matchPeriodsFrames || null;
    const playerName = location.state?.playerName || null;
    const startWithoutSelection = location.state?.startAnalysis === true;
    const isFreshAnalysis = Boolean(
        (selectedBbox && Array.isArray(selectedBbox) && selectedBbox.length === 4) ||
        (multiSegments && multiSegments.length > 0) ||
        startWithoutSelection
    );

    const [session, setSession] = useState(null);
    const [aiSummaryTeam, setAiSummaryTeam] = useState(null);
    const [aiSummaryPlayer, setAiSummaryPlayer] = useState(null);
    const [aiProgress, setAiProgress] = useState(0);   // 0-100, mirrors the ai_summary task row

    const [error, setError] = useState(null);
    const [videoSize, setVideoSize] = useState({ width: 1280, height: 720 });

    const [drawerOpen, setDrawerOpen] = useState(false);
    const [minimapOn, setMinimapOn] = useState(false);
    const [overlayOn, setOverlayOn] = useState(true);
    const [aiGenerating, setAiGenerating] = useState(false);
    const [viewMode, setViewMode] = useState('team'); // 'team' = tactical review, 'player' = player dossier
    const [drawMode, setDrawMode] = useState(false);
    const [tacticalDrawings, setTacticalDrawings] = useState([]);
    const loadedDrawings = useRef(false);
    const [initialStrokes, setInitialStrokes] = useState([]);
    const [minimapExpanded, setMinimapExpanded] = useState(false);
    const [isVideoBuffering, setIsVideoBuffering] = useState(false);
    const telestrationRef = useRef(null);

    const [activeTacticalTab, setActiveTacticalTab] = useState('pass_network');
    const [tacticalModalChart, setTacticalModalChart] = useState(null);
    const [tacticalImgStatus, setTacticalImgStatus] = useState({});
    const [tacticalVersion, setTacticalVersion] = useState({});
    const [isGeneratingTactical, setIsGeneratingTactical] = useState({});

    const handleGenerateTacticalChart = async (chart) => {
        if (!sessionId || isGeneratingTactical[chart.id]) return;
        setIsGeneratingTactical(prev => ({ ...prev, [chart.id]: true }));
        const tId = toast.loading(t('dashboard.chartCalculating', { label: chart.label }));
        try {
            await queueFeature(sessionId, chart.feature || chart.id);
            toast.success(t('dashboard.chartDispatched', { label: chart.label }), { id: tId });
            const imgPath = `/api/sessions/${sessionId}/files/${chart.filename}${API_KEY ? `?key=${encodeURIComponent(API_KEY)}` : ''}`;
            const targetUrl = absUrl(imgPath);
            let attempts = 0;
            const pollId = setInterval(async () => {
                attempts += 1;
                try {
                    const r = await fetch(targetUrl, { method: 'HEAD' });
                    if (r.ok || attempts > 20) {
                        clearInterval(pollId);
                        if (r.ok) {
                            setTacticalVersion(prev => ({ ...prev, [chart.id]: Date.now() }));
                            setTacticalImgStatus(prev => ({ ...prev, [chart.id]: 'loaded' }));
                            toast.success(t('dashboard.chartDone', { label: chart.label }));
                        }
                        setIsGeneratingTactical(prev => ({ ...prev, [chart.id]: false }));
                    }
                } catch {
                    if (attempts > 20) {
                        clearInterval(pollId);
                        setIsGeneratingTactical(prev => ({ ...prev, [chart.id]: false }));
                    }
                }
            }, 800);
        } catch (err) {
            toast.error(t('dashboard.chartFail', { error: err.message }), { id: tId });
            setIsGeneratingTactical(prev => ({ ...prev, [chart.id]: false }));
        }
    };

    const [coachQuery, setCoachQuery] = useState('');
    const [coachLoading, setCoachLoading] = useState(false);
    const [coachAnswer, setCoachAnswer] = useState(null);
    const [coachHistory, setCoachHistory] = useState([]);   // [{role, text}]
    const [coachQuestionsUsed, setCoachQuestionsUsed] = useState(0);
    const MAX_COACH_QUESTIONS = 3;

    const handleAskCoach = async (overrideQuery) => {
        const q = (typeof overrideQuery === 'string' ? overrideQuery : coachQuery).trim();
        if (!q || !sessionId || coachLoading) return;
        if (coachQuestionsUsed >= MAX_COACH_QUESTIONS) return;
        setCoachLoading(true);
        const tId = toast.loading(t('dashboard.copilotAnalyzing'));
        try {
            const data = await askCoachQA(sessionId, q, coachHistory);
            setCoachAnswer(data);
            setCoachQuery('');

            const used = data.questions_used ?? (coachQuestionsUsed + 1);
            const remaining = data.questions_remaining ?? (MAX_COACH_QUESTIONS - used);
            setCoachQuestionsUsed(used);

            // Append to history for multi-turn
            if (data.answer_type !== 'limit_reached') {
                setCoachHistory(prev => [
                    ...prev,
                    { role: 'user', text: q },
                    { role: 'model', text: data.tactical_summary || '' },
                ]);
            }

            if (data.answer_type === 'limit_reached') {
                toast(t('dashboard.copilotLimitReached'), { id: tId });
            } else if (data.answer_type === 'llm') {
                toast.success(t('dashboard.copilotRemaining', { remaining }), { id: tId });
            } else {
                toast(t('dashboard.copilotEventsMatched', { count: data.total_matched_events, remaining }), { id: tId });
            }
        } catch (err) {
            toast.error(err.message || t('dashboard.copilotFailed'), { id: tId });
        } finally {
            setCoachLoading(false);
        }
    };


    const [highlightsManifest, setHighlightsManifest] = useState(null);
    const [highlightsLoading, setHighlightsLoading] = useState(false);
    const [activeHighlightModal, setActiveHighlightModal] = useState(null);

    const loadHighlights = useCallback(async () => {
        if (!sessionId) return;
        try {
            const data = await getHighlightsManifest(sessionId);
            if (data && data.highlights) {
                setHighlightsManifest(data);
            }
        } catch {
            // silent fallback
        }
    }, [sessionId]);

    useEffect(() => {
        if (sessionId && session?.status === 'analysis_done') {
            loadHighlights();
        }
    }, [sessionId, session?.status, loadHighlights]);

    const handleGenerateHighlights = async () => {
        if (!sessionId || highlightsLoading) return;
        setHighlightsLoading(true);
        const tId = toast.loading(t('dashboard.highlightsGenerating'));
        try {
            await queueFeature(sessionId, 'tactical_highlights');
            toast.success(t('dashboard.highlightsDispatched'), { id: tId });
            let attempts = 0;
            const pollId = setInterval(async () => {
                attempts += 1;
                try {
                    const data = await getHighlightsManifest(sessionId);
                    if ((data && data.highlights && data.highlights.length > 0) || attempts > 25) {
                        clearInterval(pollId);
                        if (data && data.highlights) {
                            setHighlightsManifest(data);
                            toast.success(t('dashboard.highlightsSuccess', { count: data.total_highlights }));
                        }
                        setHighlightsLoading(false);
                    }
                } catch {
                    if (attempts > 25) {
                        clearInterval(pollId);
                        setHighlightsLoading(false);
                    }
                }
            }, 1000);
        } catch (err) {
            toast.error(t('dashboard.highlightsFail', { error: err.message }), { id: tId });
            setHighlightsLoading(false);
        }
    };

    const analysisKicked = useRef(false);
    const summaryFetched = useRef(false);
    const heroVideoRef = useRef(null);
    const realtimeEvents = useRef(0);

    const phase = session?.status || 'uploaded';
    const progress = session?.progress ?? 0;
    const stage = session?.stage || '';

    const [isBundling, setIsBundling] = useState(false);

    const handleDownloadBundle = async () => {
        if (!sessionId || isBundling) return;
        setIsBundling(true);
        const toastId = toast.loading(t('dashboard.zipPackaging'));
        try {
            await queueFeature(sessionId, 'match_bundle');
            const bundleUrl = absUrl(`/api/sessions/${sessionId}/files/match_analysis_bundle.zip${API_KEY ? `?key=${encodeURIComponent(API_KEY)}` : ''}`);

            // 轮询等待后端压缩与校验完成，消除 404 竞态
            let ready = false;
            for (let i = 0; i < 20; i++) {
                try {
                    const res = await fetch(bundleUrl, { method: 'HEAD' });
                    if (res.ok) {
                        ready = true;
                        break;
                    }
                } catch (_) {}
                await new Promise((r) => setTimeout(r, 600));
            }

            if (ready) {
                toast.success(t('dashboard.zipSuccess'), { id: toastId });
                window.open(bundleUrl, '_blank');
            } else {
                toast.error(t('dashboard.zipSlow'), { id: toastId });
            }
        } catch (err) {
            console.error('Failed to trigger match bundle:', err);
            toast.error(t('dashboard.zipFail'), { id: toastId });
        } finally {
            setIsBundling(false);
        }
    };

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

    // Load drawings from session DB once
    useEffect(() => {
        if (session && !loadedDrawings.current) {
            loadedDrawings.current = true;
            let extra = session.extra;
            if (typeof extra === 'string') {
                try { extra = JSON.parse(extra); } catch { extra = {}; }
            }
            if (extra?.tactical_drawings) {
                setTacticalDrawings(extra.tactical_drawings);
            }
        }
    }, [session]);

    const phaseLabel = isColdStart
        ? t('dashboard.gpuWarming', { sec: coldStartSec })
        : (lang === 'zh' ? {
            uploaded: t('dashboard.phaseUploaded'),
            uploading: t('dashboard.phaseUploading'),
            queued: t('dashboard.phaseQueued'),
            analyzing: t('dashboard.phaseAnalyzing'),
            analysis_done: t('dashboard.phaseDone'),
            tracking: t('dashboard.phaseTracking'),
            tracking_done: t('dashboard.phaseTrackingDone'),
            analysis_failed: t('dashboard.phaseAnalysisFailed'),
            tracking_failed: t('dashboard.phaseTrackingFailed'),
        }[phase] : PHASE_LABELS[phase]) || STAGE_LABELS[stage] || stage || phase;
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
        setAiSummaryTeam(null);
        setAiSummaryPlayer(null);
        setSpatialRadarData(null);

        setError(null);
        setMinimapOn(false);
        setAiGenerating(false);
        loadedDrawings.current = false;
        analysisKicked.current = false;
        summaryFetched.current = false;
    }, [sessionId]);

    // Kick off pipeline on mount (only if not already started)
    useEffect(() => {
        if (!sessionId) return;
        if (analysisKicked.current) return;
        analysisKicked.current = true;
        (async () => {
            try {
                // Fetch the session FIRST to check its status. 
                // If the user refreshed the page, the session might already be tracking.
                let currentSession = session;
                if (!currentSession) {
                    currentSession = await getSession(sessionId).catch(() => null);
                }
                const status = currentSession?.status;
                if (['queued', 'processing', 'tracking', 'analyzing', 'analysis_done', 'analysis_failed', 'tracking_failed'].includes(status)) {
                    const lastUpdated = new Date(currentSession?.updated_at || currentSession?.created_at).getTime();
                    const minsSinceUpdate = (Date.now() - lastUpdated) / 60000;
                    if (minsSinceUpdate <= 20 || status.includes('failed') || status.includes('done')) {
                        console.log(`Session is in state: ${status}. Skipping auto-start.`);
                        return;
                    }
                }

                // Sanitize history state so subsequent refresh/back-forward won't re-trigger
                if (isFreshAnalysis) {
                    navigate(`${location.pathname}?sessionId=${encodeURIComponent(sessionId)}`, {
                        replace: true,
                        state: { sessionId, videoId: sessionId },
                    });
                }

                if (multiSegments && multiSegments.length > 0) {
                    // Multi-segment path
                    const segments = multiSegments.map((seg) => ({
                        frame: seg.frame,
                        bbox: seg.bbox,
                        period_idx: seg.period_idx ?? 0,
                        img_dims: seg.img_dims,
                    }));
                    await startTrackingMulti(sessionId, segments, matchPeriodsFrames, location.state?.clientFps);
                    toast.success(t('dashboard.trackingParallelSuccess', { count: segments.length }));
                } else if (selectedBbox && Array.isArray(selectedBbox) && selectedBbox.length === 4) {
                    const [x1, y1, x2, y2] = selectedBbox;
                    const imgDims = location.state?.imgDims || null;
                    await startTracking(sessionId, { x1, y1, x2, y2 }, 0, imgDims);
                    if (playerName) toast.success(t('dashboard.trackingPlayerSuccess', { name: playerName }));
                } else if (startWithoutSelection) {
                    await startAnalysis(sessionId);
                }
            } catch (e) {
                const msg = e?.response?.data?.detail || e?.message || t('dashboard.failedToStart');
                setError(msg); toast.error(msg);
            }
        })();
    }, [sessionId, selectedBbox, multiSegments, playerName, startWithoutSelection]);

    // Subscribe to live updates + initial fetch + polling fallback
    useEffect(() => {
        if (!sessionId) return;
        let cancelled = false;

        const handleAiTask = (t) => {
            const isAi = t.task_type?.startsWith('ai_summary');
            if (!isAi) return;
            const isPlayer = t.task_type === 'ai_summary_player' || t.result?.analysis_mode === 'player';
            if (t.result) {
                if (isPlayer) setAiSummaryPlayer(t.result);
                else setAiSummaryTeam(t.result);
            }
            if (t.status === 'completed') {
                setAiGenerating(false);
                setAiProgress(100);
            } else if (t.status === 'running') {
                setAiProgress(Math.max(0, Math.min(100, Number(t.progress) || 0)));
            } else if (t.status === 'failed') {
                setAiGenerating(false);
            }
        };

        const applyTasks = (tasks = []) => {
            for (const t of tasks) handleAiTask(t);
        };

        getSession(sessionId).then((s) => { if (!cancelled) setSession(s); }).catch(() => { });
        if (!isFreshAnalysis) {
            listTasks(sessionId).then(applyTasks).catch(() => { });
        }

        realtimeEvents.current = 0;
        const pollStartedAt = Date.now();
        // Hard cap: even if Realtime never fires and analysis never reaches
        // a terminal status, stop polling after 30 minutes. Otherwise a
        // forgotten tab on the dashboard hammers Supabase every 2s forever.
        const POLL_HARD_CAP_MS = 30 * 60 * 1000;
        const pollInterval = setInterval(() => {
            if (realtimeEvents.current >= 2) { clearInterval(pollInterval); return; }
            if (Date.now() - pollStartedAt > POLL_HARD_CAP_MS) {
                console.warn('[Dashboard] fallback polling hit 30min cap — stopping');
                clearInterval(pollInterval);
                return;
            }
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
                handleAiTask(t);
            },
        });

        return () => { cancelled = true; clearInterval(pollInterval); unsub(); };
    }, [sessionId, isFreshAnalysis]);

    // Fetch summary once analysis_done
    useEffect(() => {
        if (!isDone || summaryFetched.current) return;
        summaryFetched.current = true;
        listSummaries(sessionId).then((summaries) => {
            if (summaries && summaries.length > 0) {
                for (const s of summaries) {
                    if (s.analysis_mode === 'player' || s.task_type === 'ai_summary_player') {
                        setAiSummaryPlayer((prev) => prev || s);
                    } else {
                        setAiSummaryTeam((prev) => prev || s);
                    }
                }
            }
        }).catch(() => { });
    }, [isDone, sessionId]);

    const resolveTelemetryUrl = (remoteUrl, defaultFilename) => {
        const raw = remoteUrl || (sessionId ? `/api/sessions/${sessionId}/files/${defaultFilename}` : null);
        if (!raw) return null;
        if (/^https?:\/\//i.test(raw)) return raw;
        const withKey = API_KEY ? `${raw}${raw.includes('?') ? '&' : '?'}key=${encodeURIComponent(API_KEY)}` : raw;
        return absUrl(withKey);
    };

    const minimapDataUrl = resolveTelemetryUrl(session?.minimap_data_url, 'minimap_positions.json');
    const overlayDataUrl = resolveTelemetryUrl(session?.overlay_data_url, 'overlay_bboxes.json');
    const heatmapDataUrl = resolveTelemetryUrl(session?.heatmap_data_url, 'heatmap_positions.json');
    const spatialRadarUrl = resolveTelemetryUrl(session?.spatial_radar_url, 'spatial_radar.json');
    const [spatialRadarData, setSpatialRadarData] = useState(null);

    useEffect(() => {
        if (!spatialRadarUrl) return;
        let cancelled = false;
        fetch(spatialRadarUrl)
            .then((res) => (res.ok ? res.json() : null))
            .then((data) => {
                if (!cancelled && data) {
                    setSpatialRadarData(data);
                }
            })
            .catch(() => {});
        return () => {
            cancelled = true;
        };
    }, [spatialRadarUrl]);

    const playerSummaryJson = session?.player_summary || null;
    
    const currentSummary = viewMode === 'player' ? aiSummaryPlayer : aiSummaryTeam;

    const aiMarkdown = useMemo(() => {
        const txt = taskTextResult(currentSummary);
        if (!txt) return '';
        try {
            let html = DOMPurify.sanitize(marked.parse(txt));
            // 将 [MM:SS] / [H:MM:SS] / 【MM:SS】 时间戳转为可点击的跳转链接（在sanitize之后操作，安全）
            html = html.replace(
                /[\[【](\d{1,2}:)?(\d{1,3}):(\d{2})[\]】]/g,
                (match, h, m, s) => {
                    const hours = h ? parseInt(h.replace(':', ''), 10) : 0;
                    const sec = hours * 3600 + parseInt(m, 10) * 60 + parseInt(s, 10);
                    return `<button class="ai-timestamp" data-seconds="${sec}">${match}</button>`;
                }
            );
            return html;
        }
        catch { return DOMPurify.sanitize(txt); }
    }, [currentSummary]);

    const aiHighlights = useMemo(() => {
        const txt = taskTextResult(currentSummary);
        if (!txt) return [];
        const highlights = [];
        const regex = /[\[【](\d{1,2}:)?(\d{1,3}):(\d{2})[\]】]/g;
        let match;
        // Keep track of added times to avoid duplicates
        const seen = new Set();
        while ((match = regex.exec(txt)) !== null) {
            const hours = match[1] ? parseInt(match[1].replace(':', ''), 10) : 0;
            const totalSec = hours * 3600 + parseInt(match[2], 10) * 60 + parseInt(match[3], 10);
            if (!seen.has(totalSec)) {
                seen.add(totalSec);
                highlights.push({ time: totalSec, label: match[0] });
            }
        }
        return highlights;
    }, [currentSummary]);

    const handleGenerateAI = async () => {
        if (aiGenerating || !sessionId) return;
        setAiGenerating(true);
        try {
            await queueFeature(sessionId, 'ai_summary', { mode: viewMode });
            toast.success(
                viewMode === 'player'
                    ? t('dashboard.playerGenerating')
                    : t('dashboard.teamGenerating')
            );
        } catch (e) {
            toast.error(e?.message || t('common.error'));
            setAiGenerating(false);
        }
    };

    // Body scroll lock while the drawer is open. Without it, scrolling
    // anywhere over the drawer that *isn't* a deep overflow:auto container
    // (e.g. between cards, on the heatmap canvas, on charts) bubbles back
    // to the page behind and the video moves instead of the drawer.
    //
    // (The earlier "drawer feels frozen" complaint was actually caused by
    // framer-motion's stacking context, not this lock — that's been fixed
    // by switching the drawer to plain CSS, so the lock is safe again.)
    useEffect(() => {
        if (!drawerOpen) return;
        // Lock body only — not <html>. Some Chrome combos treat html as the
        // root scroll container even when something inside is a fixed
        // position drawer with its own overflow:auto; html overflow:hidden
        // then interferes with that inner scroll. Locking body alone stops
        // the page-behind-drawer scroll without touching the drawer's own
        // scroll container.
        if (!drawerOpen) return;
        const prevBody = document.body.style.overflow;
        document.body.style.overflow = 'hidden';
        return () => {
            document.body.style.overflow = prevBody;
        };
    }, [drawerOpen]);

    // Global hotkeys
    useEffect(() => {
        const handleKeyDown = (e) => {
            // Ignore if user is typing in an input field
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.isComposing) return;
            
            const v = heroVideoRef.current;
            switch (e.key.toLowerCase()) {
                case ' ':
                    e.preventDefault();
                    if (v) v.paused ? v.play().catch(()=>{}) : v.pause();
                    break;
                case 'arrowleft':
                    e.preventDefault();
                    if (v) v.currentTime = Math.max(0, v.currentTime - 5);
                    break;
                case 'arrowright':
                    e.preventDefault();
                    if (v) v.currentTime = Math.min(v.duration, v.currentTime + 5);
                    break;
                case 'm':
                    e.preventDefault();
                    setMinimapOn(prev => !prev);
                    break;
                case 'd':
                    e.preventDefault();
                    setDrawMode(prev => !prev);
                    break;
                case 'f':
                    e.preventDefault();
                    if (v) {
                        const wrap = v.parentElement;
                        if (document.fullscreenElement) {
                            document.exitFullscreen().catch(()=>{});
                        } else {
                            wrap.requestFullscreen().catch(()=>{});
                        }
                    }
                    break;
                case ',':
                    e.preventDefault();
                    if (v) {
                        v.pause();
                        v.currentTime = Math.max(0, v.currentTime - 0.04); // Approx 1 frame at 25fps
                    }
                    break;
                case '.':
                    e.preventDefault();
                    if (v) {
                        v.pause();
                        v.currentTime = Math.min(v.duration, v.currentTime + 0.04); // Approx 1 frame at 25fps
                    }
                    break;
                default:
                    break;
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, []);


    const handleNewPlayer = () => {
        if (!sessionId) return;
        if (isAnalyzing) {
            toast(t('dashboard.waitCurrentAnalysis'));
            return;
        }
        navigate(`/configure-multi?sessionId=${encodeURIComponent(sessionId)}`, {
            state: { videoId: sessionId, sessionId },
        });
    };



    const handleToggleDraw = useCallback((mode) => {
        setDrawMode(mode);
        if (!mode) {
            // Save strokes when exiting
            const strokes = telestrationRef.current?.getStrokes?.() || [];
            if (heroVideoRef.current) {
                const time = heroVideoRef.current.currentTime;
                setTacticalDrawings(prev => {
                    const existingIdx = prev.findIndex(d => Math.abs(d.time - time) < 0.5);
                    let nextDrawings = prev;
                    
                    if (strokes.length === 0) {
                        if (existingIdx >= 0) {
                            nextDrawings = [...prev];
                            nextDrawings.splice(existingIdx, 1);
                        }
                    } else {
                        if (existingIdx >= 0) {
                            nextDrawings = [...prev];
                            nextDrawings[existingIdx] = { time, strokes };
                        } else {
                            nextDrawings = [...prev, { time, strokes }].sort((a,b) => a.time - b.time);
                        }
                    }
                    
                    // Save to DB in background
                    saveTacticalDrawings(sessionId, nextDrawings)
                        .then(() => toast.success(t('dashboard.saveBoard')))
                        .catch(err => {
                            console.error('Failed to save tactical drawings:', err);
                            toast.error(t('dashboard.saveBoardFail'));
                        });
                    
                    return nextDrawings;
                });
            }
            telestrationRef.current?.clearCanvas?.();
        } else {
            // Enter drawing mode: load strokes if we are near a saved drawing
            if (heroVideoRef.current) {
                const time = heroVideoRef.current.currentTime;
                const existing = tacticalDrawings.find(d => Math.abs(d.time - time) < 0.5);
                if (existing) {
                    setInitialStrokes([...existing.strokes]);
                } else {
                    setInitialStrokes([]);
                }
            } else {
                setInitialStrokes([]);
            }
        }
    }, [tacticalDrawings, sessionId, t]);

    if (!sessionId) {
        return (
            <div className="dashboard dashboard--v2">
                <div className="bg-grid" />
                <StepNav />
                <div className="dashboard__error-banner">
                    <HiExclamationCircle /> {t('dashboard.noSession') || t('trimmer.noSession')}
                </div>
                <button className="btn btn-primary" onClick={() => navigate('/upload')}>{t('common.back')}</button>
            </div>
        );
    }

    return (
        <div className="dashboard dashboard--v2">
            <div className="bg-grid" />

            {/* Top bar */}
            <div className="dashboard-v2__topbar">
                <button className="btn btn-ghost" onClick={() => navigate('/')}>
                    <HiHome /> {t('common.home')}
                </button>
                <div className="dashboard-v2__title">
                    <span className="dashboard-v2__title-main">{t('common.analysis')}</span>
                    <span className="dashboard-v2__title-sub">{t('dashboard.sessionSubtitle', { id: sessionId.slice(0, 8) })}</span>
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
                        <HiExclamationCircle /> {error || session?.error || t('dashboard.pipelineFailed')}
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
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                            <HiPlayCircle /> <span>{t('dashboard.annotatedReplay')}</span>
                        </div>
                        {session?.status === 'analysis_done' && (
                            <button
                                className="hero-video-card__download"
                                onClick={handleDownloadBundle}
                                disabled={isBundling}
                                title={t('dashboard.exportZip')}
                            >
                                <HiArrowDownTray />
                                <span>{isBundling ? t('dashboard.packaging') : t('dashboard.exportReportZip')}</span>
                            </button>
                        )}
                    </div>

                    <div className="hero-video-card__body">
                        {session?.status === 'analysis_done' ? (
                            <div className="hero-video-card__player-wrap">
                                <video
                                    ref={heroVideoRef}
                                    src={session?.video_url}
                                    autoPlay
                                    muted
                                    loop
                                    playsInline
                                    onClick={(event) => {
                                        const v = event.currentTarget;
                                        if (v.paused) v.play?.().catch(() => { });
                                        else v.pause?.();
                                    }}
                                    onDoubleClick={(e) => { e.preventDefault();
                                        const wrap = e.currentTarget.parentElement;
                                        if (document.fullscreenElement) {
                                            document.exitFullscreen().catch(()=>{});
                                        } else {
                                            wrap.requestFullscreen().catch(()=>{});
                                        }
                                    }}
                                    onLoadedMetadata={(e) => setVideoSize({ width: e.currentTarget.videoWidth || 1280, height: e.currentTarget.videoHeight || 720 })}
                                    onWaiting={() => setIsVideoBuffering(true)}
                                    onPlaying={() => setIsVideoBuffering(false)}
                                    onPause={() => setIsVideoBuffering(false)}
                                    onCanPlay={() => setIsVideoBuffering(false)}
                                    onLoadedData={() => setIsVideoBuffering(false)}
                                    onSeeked={() => setIsVideoBuffering(false)}
                                    className="hero-video-card__player"
                                />
                                {isVideoBuffering && (
                                    <div
                                        className="hero-video-card__buffering-overlay"
                                        onClick={() => {
                                            // Let clicks pass through to the video element
                                            const v = heroVideoRef.current;
                                            if (v && v.paused) v.play().catch(() => {});
                                        }}
                                    >
                                        <div className="feature-card__spinner" style={{ width: 48, height: 48, borderTopColor: '#60a5fa' }} />
                                        <p>{t('dashboard.buffering')}</p>
                                    </div>
                                )}
                                <CanvasOverlay
                                    dataUrl={overlayDataUrl}
                                    videoRef={heroVideoRef}
                                    visible={overlayOn}
                                />
                                <TelestrationCanvas
                                    active={drawMode}
                                    parentRef={telestrationRef}
                                    videoRef={heroVideoRef}
                                    width={videoSize.width}
                                    height={videoSize.height}
                                    initialStrokes={initialStrokes}
                                    onInteractionStart={() => {
                                        if (heroVideoRef.current && !heroVideoRef.current.paused) {
                                            heroVideoRef.current.pause();
                                        }
                                    }}
                                />
                                <MinimapOverlay
                                    dataUrl={minimapDataUrl}
                                    videoRef={heroVideoRef}
                                    visible={minimapOn}
                                    onExpand={() => setMinimapExpanded(true)}
                                />
                            </div>
                        ) : (
                            <div className="hero-video-card__placeholder">
                                <HiPlayCircle style={{ fontSize: 48, opacity: 0.4 }} />
                                <p>{isAnalyzing ? 'Replay will appear here once analysis finishes…' : 'No replay yet.'}</p>
                            </div>
                        )}

                        {session?.status === 'analysis_done' && (
                            <VideoTimelineMarkers
                                segments={session?.segments}
                                matchPeriods={session?.match_periods_frames}
                                fps={session?.video_fps}
                                totalFrames={session?.total_frames}
                                videoRef={heroVideoRef}
                                drawMode={drawMode}
                                highlights={aiHighlights}
                                tacticalDrawings={tacticalDrawings}
                                onToggleDraw={handleToggleDraw}
                            />
                        )}
                    </div>
                </motion.div>
                <StepNav />
              </div>
            </main>

            {/* Side drawer — plain aside w/ CSS transition.
                We tried framer-motion's motion.aside before but its inline
                transform created a stacking context that intermittently
                trapped scroll + click events. Plain CSS transition is
                bulletproof and the animation is identical. */}
            <aside
                className={`dashboard-v2__drawer ${drawerOpen ? 'is-open' : ''}`}
                aria-hidden={!drawerOpen}
            >
                        <div className="drawer__list">
                            {/* Minimap toggle (stays as on/off switch) */}
                            <div className={`drawer__item ${minimapOn ? 'is-active' : ''}`}>
                                <button
                                    className="drawer__item-head"
                                    onClick={() => setMinimapOn((v) => !v)}
                                >
                                    <HiMapPin />
                                    <span>{t('dashboard.minimapOverlay')}</span>
                                    <span className={`drawer__toggle ${minimapOn ? 'on' : ''}`}>
                                        {minimapOn ? 'ON' : 'OFF'}
                                    </span>
                                </button>
                            </div>

                            {/* Draw mode toggle */}
                            <div className={`drawer__item ${drawMode ? 'is-active' : ''}`}>
                                <button
                                    className="drawer__item-head"
                                    onClick={() => handleToggleDraw(!drawMode)}
                                >
                                    <HiFire />
                                    <span>{t('dashboard.telestrationBoard')}</span>
                                    <span className={`drawer__toggle ${drawMode ? 'on' : ''}`}>
                                        {drawMode ? 'ON' : 'OFF'}
                                    </span>
                                </button>
                            </div>

                            {/* Data Analysis — always rendered */}
                            <div className="drawer__item is-static">
                                <div className="drawer__item-head drawer__item-head--static">
                                    <HiChartBar />
                                    <span>{t('dashboard.dataAnalysis')}</span>
                                </div>
                                <DataAnalysisPanel playerSummary={playerSummaryJson} />
                            </div>

                            {/* Tactical Visual Dossier — Gallery */}
                            <div className="drawer__item is-static">
                                <div className="drawer__item-head drawer__item-head--static" style={{ justifyContent: 'space-between' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.7rem' }}>
                                        <HiChartBar />
                                        <span>{t('dashboard.tacticalSuite')}</span>
                                    </div>
                                    <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>{t('dashboard.dimensionsTotal', { count: tacticalCharts.length })}</span>
                                </div>
                                <div className="drawer__section-body" style={{ padding: '0.75rem' }}>
                                    <div style={{ display: 'flex', gap: '6px', overflowX: 'auto', paddingBottom: '8px', marginBottom: '8px' }}>
                                        {tacticalCharts.map((c) => (
                                            <button
                                                key={c.id}
                                                className={`btn btn-xs ${activeTacticalTab === c.id ? 'btn-primary' : 'btn-ghost'}`}
                                                style={{ whiteSpace: 'nowrap', fontSize: '0.74rem', padding: '4px 8px' }}
                                                onClick={() => setActiveTacticalTab(c.id)}
                                            >
                                                {c.label}
                                            </button>
                                        ))}
                                    </div>

                                    {(() => {
                                        const curChart = tacticalCharts.find((c) => c.id === activeTacticalTab) || tacticalCharts[0];
                                        const curVer = tacticalVersion[curChart.id];
                                        const vParam = curVer ? `&v=${curVer}` : '';
                                        const chartUrl = absUrl(`/api/sessions/${sessionId}/files/${curChart.filename}${API_KEY ? `?key=${encodeURIComponent(API_KEY)}${vParam}` : (curVer ? `?v=${curVer}` : '')}`);
                                        const isErr = tacticalImgStatus[curChart.id] === 'error';
                                        return (
                                            <div>
                                                <div style={{ fontSize: '0.76rem', color: '#94a3b8', marginBottom: '6px', lineHeight: 1.4 }}>
                                                    {curChart.desc}
                                                </div>
                                                <div
                                                    style={{
                                                        position: 'relative',
                                                        borderRadius: '8px',
                                                        overflow: 'hidden',
                                                        background: 'rgba(0,0,0,0.3)',
                                                        minHeight: '140px',
                                                        border: '1px solid rgba(255,255,255,0.06)',
                                                        display: 'flex',
                                                        alignItems: 'center',
                                                        justifyContent: 'center',
                                                    }}
                                                >
                                                    {!isErr ? (
                                                        <div style={{ position: 'relative', width: '100%', cursor: 'pointer' }} onClick={() => setTacticalModalChart(curChart)}>
                                                            <img
                                                                src={chartUrl}
                                                                alt={curChart.label}
                                                                style={{ width: '100%', display: 'block', borderRadius: '8px' }}
                                                                onError={() => setTacticalImgStatus((prev) => ({ ...prev, [curChart.id]: 'error' }))}
                                                                onLoad={() => setTacticalImgStatus((prev) => ({ ...prev, [curChart.id]: 'loaded' }))}
                                                            />
                                                            <div
                                                                style={{
                                                                    position: 'absolute',
                                                                    top: '6px',
                                                                    right: '6px',
                                                                    background: 'rgba(0,0,0,0.6)',
                                                                    borderRadius: '4px',
                                                                    padding: '3px 6px',
                                                                    fontSize: '0.7rem',
                                                                    color: '#cbd5e1',
                                                                    display: 'flex',
                                                                    alignItems: 'center',
                                                                    gap: '3px',
                                                                }}
                                                            >
                                                                <HiArrowsPointingOut /> {t('dashboard.clickToEnlarge')}
                                                            </div>
                                                        </div>
                                                    ) : (
                                                        <div style={{ textAlign: 'center', padding: '16px 8px' }}>
                                                            <p style={{ fontSize: '0.78rem', color: '#94a3b8', marginBottom: '8px' }}>
                                                                {t('dashboard.chartNotGenerated')}
                                                            </p>
                                                            <button
                                                                className="btn btn-xs btn-primary"
                                                                disabled={isGeneratingTactical[curChart.id]}
                                                                onClick={() => handleGenerateTacticalChart(curChart)}
                                                            >
                                                                {isGeneratingTactical[curChart.id] ? t('dashboard.generatingChart') : t('dashboard.generateNow')}
                                                            </button>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        );
                                    })()}
                                </div>
                            </div>

                            {/* 20-Zone JDP Spatial Radar */}
                            <div className="drawer__item is-static">
                                <PitchZoneAnalysisPanel
                                    onSeekTimestamp={(sec) => {
                                        if (heroVideoRef.current) {
                                            heroVideoRef.current.currentTime = sec;
                                            heroVideoRef.current.play().catch(() => {});
                                        }
                                    }}
                                    zoneStats={spatialRadarData || session?.tactical_zones || session?.player_summary?.tactical_zones || null}
                                />
                            </div>

                            {/* AI Analysis — generate-on-demand */}
                            <div className="drawer__item is-static">
                                <div className="drawer__item-head drawer__item-head--static">
                                    <HiSparkles />
                                    <span>{t('dashboard.copilotTitle')}</span>
                                </div>
                                <div className="drawer__section-body">
                                    {/* Mode toggle */}
                                    <div className="ai-mode-toggle">
                                        <button
                                            className={`ai-mode-toggle__btn ${viewMode === 'team' ? 'is-active' : ''}`}
                                            onClick={() => setViewMode('team')}
                                        >
                                            {t('dashboard.modeTeam')}
                                        </button>
                                        <button
                                            className={`ai-mode-toggle__btn ${viewMode === 'player' ? 'is-active' : ''}`}
                                            onClick={() => setViewMode('player')}
                                        >
                                            {t('dashboard.modePlayer')}
                                        </button>
                                    </div>

                                    {/* Interactive Tactical Coach Q&A Box */}
                                    <div className="coach-qa-box">
                                        {/* Header with question counter */}
                                        <div className="coach-qa-header">
                                            <span className="coach-qa-title">{t('dashboard.copilotTitle')}</span>
                                            <span className={`coach-qa-counter ${coachQuestionsUsed >= MAX_COACH_QUESTIONS ? 'coach-qa-counter--exhausted' : ''}`}>
                                                {coachQuestionsUsed >= MAX_COACH_QUESTIONS
                                                    ? t('dashboard.limitReached')
                                                    : t('dashboard.questionsRemaining', { remaining: MAX_COACH_QUESTIONS - coachQuestionsUsed, total: MAX_COACH_QUESTIONS })}
                                            </span>
                                        </div>

                                        <form
                                            className="coach-qa-form"
                                            onSubmit={(e) => {
                                                e.preventDefault();
                                                handleAskCoach();
                                            }}
                                        >
                                            <input
                                                type="text"
                                                className="coach-qa-input"
                                                placeholder={coachQuestionsUsed >= MAX_COACH_QUESTIONS
                                                    ? t('dashboard.copilotLimitReached')
                                                    : t('dashboard.copilotPlaceholder')}
                                                value={coachQuery}
                                                onChange={(e) => setCoachQuery(e.target.value)}
                                                disabled={coachLoading || coachQuestionsUsed >= MAX_COACH_QUESTIONS}
                                            />
                                            <button
                                                type="submit"
                                                className="coach-qa-btn"
                                                disabled={coachLoading || !coachQuery.trim() || coachQuestionsUsed >= MAX_COACH_QUESTIONS}
                                            >
                                                {coachLoading ? <HiArrowPath className="spinning" /> : <HiMagnifyingGlass />}
                                                <span>{t('dashboard.copilotAsk')}</span>
                                            </button>
                                        </form>

                                        {/* Suggestion chips — only show before first question */}
                                        {coachHistory.length === 0 && coachQuestionsUsed < MAX_COACH_QUESTIONS && (
                                            <div className="coach-qa-chips">
                                                {[
                                                    { label: t('dashboard.chipRunning'), q: t('dashboard.chipRunningQ') },
                                                    { label: t('dashboard.chipAttacking'), q: t('dashboard.chipAttackingQ') },
                                                    { label: t('dashboard.chipDefending'), q: t('dashboard.chipDefendingQ') },
                                                    { label: t('dashboard.chipTurningPoint'), q: t('dashboard.chipTurningPointQ') },
                                                ].map((chip) => (
                                                    <button
                                                        key={chip.label}
                                                        type="button"
                                                        className="coach-qa-chip"
                                                        onClick={() => {
                                                            setCoachQuery(chip.q);
                                                            handleAskCoach(chip.q);
                                                        }}
                                                        disabled={coachLoading}
                                                    >
                                                        {chip.label}
                                                    </button>
                                                ))}
                                            </div>
                                        )}

                                        {/* Conversation history */}
                                        {coachHistory.length > 0 && (
                                            <div className="coach-qa-history">
                                                {coachHistory.map((turn, idx) => (
                                                    <div key={idx} className={`coach-qa-turn coach-qa-turn--${turn.role}`}>
                                                        <span className="coach-qa-turn__label">
                                                            {turn.role === 'user' ? t('dashboard.you') : t('dashboard.coach')}
                                                        </span>
                                                        <div className="coach-qa-turn__text">
                                                            {turn.text.split('\n').map((line, i) => (
                                                                <p key={i}>{line}</p>
                                                            ))}
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        )}

                                        {/* Latest answer (shown separately if no history yet) */}
                                        {coachAnswer && coachHistory.length === 0 && (
                                            <div className="coach-qa-result">
                                                <div className="coach-qa-result__header">
                                                    <span>
                                                        {coachAnswer.answer_type === 'llm' ? t('dashboard.aiCoachAnswer') :
                                                         coachAnswer.answer_type === 'limit_reached' ? t('dashboard.limitReached') : t('dashboard.ruleBasedSearch')}
                                                    </span>
                                                    {coachAnswer.answer_type !== 'limit_reached' && (
                                                        <span className="coach-qa-result__badge">
                                                            {t('dashboard.latencyMs', { ms: coachAnswer.execution_latency_ms })}
                                                            {coachAnswer.answer_type === 'rule_based' && ` • ${t('dashboard.eventsCount', { count: coachAnswer.total_matched_events })}`}
                                                        </span>
                                                    )}
                                                </div>
                                                <div className="coach-qa-result__text">
                                                    {coachAnswer.tactical_summary?.split('\n').map((line, i) => (
                                                        <p key={i}>{line}</p>
                                                    ))}
                                                </div>
                                                {coachAnswer.playlist?.length > 0 && (
                                                    <div className="coach-qa-playlist">
                                                        {coachAnswer.playlist.map((clip) => {
                                                            const m = Math.floor(clip.clip_start_s / 60);
                                                            const s = Math.floor(clip.clip_start_s % 60).toString().padStart(2, '0');
                                                            return (
                                                                <button
                                                                    key={clip.clip_id}
                                                                    type="button"
                                                                    className="coach-qa-clip"
                                                                    onClick={() => {
                                                                        if (heroVideoRef.current) {
                                                                            heroVideoRef.current.currentTime = clip.clip_start_s;
                                                                            heroVideoRef.current.play().catch(() => {});
                                                                            toast.success(t('dashboard.jumpedToTimestamp', { time: `${m}:${s}`, headline: clip.headline }));
                                                                        }
                                                                    }}
                                                                >
                                                                    <span className="coach-qa-clip__time">
                                                                        <HiPlay /> [{m}:{s}]
                                                                    </span>
                                                                    <span className="coach-qa-clip__title">
                                                                        {clip.headline}
                                                                    </span>
                                                                </button>
                                                            );
                                                        })}
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                    {aiMarkdown ? (
                                        <div
                                            className="markdown-body"
                                            dangerouslySetInnerHTML={{ __html: aiMarkdown }}
                                            onClick={(e) => {
                                                // Timestamp click handler: [MM:SS] links jump the video
                                                const el = e.target.closest('.ai-timestamp');
                                                if (el && heroVideoRef.current) {
                                                    const sec = parseInt(el.dataset.seconds, 10);
                                                    if (!isNaN(sec)) {
                                                        const video = heroVideoRef.current;
                                                        const maxSec = video.duration || 0;
                                                        let targetSec = Math.max(0, sec);
                                                        if (maxSec > 0 && targetSec > maxSec) {
                                                            targetSec = maxSec;
                                                            const m = Math.floor(maxSec / 60);
                                                            const s = String(Math.floor(maxSec % 60)).padStart(2, '0');
                                                            toast(t('dashboard.timestampOutOfRange', { time: `${m}:${s}` }));
                                                        }
                                                        video.currentTime = targetSec;
                                                        video.play().catch(() => {});
                                                    }
                                                }
                                            }}
                                        />
                                    ) : aiGenerating ? (
                                        <div className="drawer__loading">
                                            <div className="feature-card__spinner" />
                                            <div style={{ flex: 1, minWidth: 0 }}>
                                                <div style={{
                                                    display: 'flex',
                                                    justifyContent: 'space-between',
                                                    fontSize: 13,
                                                    marginBottom: 6,
                                                }}>
                                                    <span>{t('dashboard.generatingSummary')}</span>
                                                    <span style={{ color: '#a78bfa', fontVariantNumeric: 'tabular-nums' }}>
                                                        {aiProgress}%
                                                    </span>
                                                </div>
                                                <div className="pipeline-status__bar-track" style={{ marginBottom: 0 }}>
                                                    <div
                                                        className="pipeline-status__bar-fill"
                                                        style={{
                                                            width: `${aiProgress}%`,
                                                            transition: 'width 0.3s ease-out',
                                                        }}
                                                    />
                                                </div>
                                            </div>
                                        </div>
                                    ) : isDone ? (
                                        <div className="drawer__empty-cta">
                                            <p>{viewMode === 'player' ? t('dashboard.generatePersonalReport') : t('dashboard.generateTeamReport')}</p>
                                            <button className="btn btn-primary" onClick={handleGenerateAI}>
                                                <HiSparkles /> {viewMode === 'player' ? t('dashboard.startPersonalAnalysis') : t('dashboard.startTeamAnalysis')}
                                            </button>
                                        </div>
                                    ) : (
                                        <p className="drawer__empty">{t('dashboard.waitingAnalysis')}</p>
                                    )}
                                </div>
                            </div>

                            {/* Broadcast HUD Highlights Reel */}
                            <div className="drawer__item is-static">
                                <div className="drawer__item-head drawer__item-head--static" style={{ justifyContent: 'space-between' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        <HiPlayCircle />
                                        <span>{t('dashboard.keyHighlightsTitle')}</span>
                                    </div>
                                    {highlightsManifest?.total_highlights > 0 && (
                                        <span className="highlights-count-badge">
                                            {t('dashboard.eventsCount', { count: highlightsManifest.total_highlights })}
                                        </span>
                                    )}
                                </div>
                                <div className="drawer__section-body">
                                    {highlightsManifest && highlightsManifest.highlights && highlightsManifest.highlights.length > 0 ? (
                                        <div className="highlights-reel-container">
                                            <div className="highlights-reel-list">
                                                {highlightsManifest.highlights.map((clip) => {
                                                    const thumbUrl = clip.thumbnail_url
                                                        ? (clip.thumbnail_url.startsWith('http')
                                                            ? clip.thumbnail_url
                                                            : absUrl(`${clip.thumbnail_url}${API_KEY ? `?key=${encodeURIComponent(API_KEY)}` : ''}`))
                                                        : null;
                                                    const eventBadge = clip.event_type === 'shot'
                                                        ? { label: t('dashboard.shotsEvent'), color: '#f59e0b' }
                                                        : clip.event_type === 'line_break'
                                                            ? { label: t('dashboard.lineBreakEvent'), color: '#10b981' }
                                                            : clip.event_type === 'counter_attack'
                                                                ? { label: t('dashboard.counterAttackEvent'), color: '#ec4899' }
                                                                : { label: t('dashboard.transitionEvent'), color: '#3b82f6' };

                                                    const defaultClipTitle = language === 'zh'
                                                        ? `战术片段 [${Math.floor((clip.event_time_s || 0) / 60)}:${String(Math.floor((clip.event_time_s || 0) % 60)).padStart(2, '0')}]`
                                                        : `Tactical Clip [${Math.floor((clip.event_time_s || 0) / 60)}:${String(Math.floor((clip.event_time_s || 0) % 60)).padStart(2, '0')}]`;

                                                    return (
                                                        <div
                                                            key={clip.clip_id}
                                                            className="highlight-card"
                                                            onClick={() => setActiveHighlightModal(clip)}
                                                        >
                                                            <div className="highlight-card__thumb-wrapper">
                                                                {thumbUrl ? (
                                                                    <img
                                                                        src={thumbUrl}
                                                                        alt={clip.metadata?.title || clip.clip_id}
                                                                        className="highlight-card__thumb"
                                                                        onError={(e) => { e.target.style.display = 'none'; }}
                                                                    />
                                                                ) : (
                                                                    <div className="highlight-card__thumb-placeholder">
                                                                        <HiPlayCircle />
                                                                    </div>
                                                                )}
                                                                <div className="highlight-card__play-overlay">
                                                                    <HiPlay />
                                                                </div>
                                                                <span
                                                                    className="highlight-card__badge"
                                                                    style={{ backgroundColor: eventBadge.color }}
                                                                >
                                                                    {eventBadge.label}
                                                                </span>
                                                                <span className="highlight-card__duration">
                                                                    {clip.duration_s?.toFixed(1) || '0.0'}s
                                                                </span>
                                                            </div>
                                                            <div className="highlight-card__info">
                                                                <div className="highlight-card__title">
                                                                    {clip.metadata?.title || defaultClipTitle}
                                                                </div>
                                                                <div className="highlight-card__actions">
                                                                    <button
                                                                        type="button"
                                                                        className="highlight-card__seek-btn"
                                                                        title={t('dashboard.jumpToMainView')}
                                                                        onClick={(e) => {
                                                                            e.stopPropagation();
                                                                            if (heroVideoRef.current && typeof clip.event_time_s === 'number') {
                                                                                heroVideoRef.current.currentTime = Math.max(0, clip.event_time_s - 1.5);
                                                                                heroVideoRef.current.play().catch(() => {});
                                                                            }
                                                                        }}
                                                                    >
                                                                        <HiPlay /> {t('dashboard.jumpToMainView')}
                                                                    </button>
                                                                    <button
                                                                        type="button"
                                                                        className="highlight-card__play-btn"
                                                                        onClick={() => setActiveHighlightModal(clip)}
                                                                    >
                                                                        <HiPlay /> {t('dashboard.hudPlay')}
                                                                    </button>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                            <div style={{ marginTop: '10px', textAlign: 'center' }}>
                                                <button
                                                    className="btn btn-xs btn-secondary"
                                                    onClick={handleGenerateHighlights}
                                                    disabled={highlightsLoading}
                                                >
                                                    {highlightsLoading ? <HiArrowPath className="spinning" /> : <HiBolt />} {t('dashboard.rerenderHighlights')}
                                                </button>
                                            </div>
                                        </div>
                                    ) : highlightsLoading ? (
                                        <div className="drawer__loading">
                                            <div className="feature-card__spinner" />
                                            <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                                                {t('dashboard.generatingHighlightsNote')}
                                            </div>
                                        </div>
                                    ) : isDone ? (
                                        <div className="drawer__empty-cta">
                                            <p>{t('dashboard.highlightsCtaNote')}</p>
                                            <button
                                                className="btn btn-primary"
                                                onClick={handleGenerateHighlights}
                                                disabled={highlightsLoading}
                                            >
                                                <HiBolt /> {t('dashboard.generateHighlightsBtn')}
                                            </button>
                                        </div>
                                    ) : (
                                        <p className="drawer__empty">{t('dashboard.waitingForDone')}</p>
                                    )}
                                </div>
                            </div>

                            {/* Heatmap */}
                            <div className="drawer__item is-static">
                                <div className="drawer__item-head drawer__item-head--static">
                                    <HiFire />
                                    <span>{language === 'zh' ? '跑动热力图' : 'Activity Heatmap'}</span>
                                </div>
                                <div className="drawer__section-body">
                                    {heatmapDataUrl ? (
                                        <HeatmapCanvas dataUrl={heatmapDataUrl} />
                                    ) : isDone ? (
                                        <p className="drawer__empty">
                                            {t('dashboard.heatmapNotExported')}
                                        </p>
                                    ) : (
                                        <p className="drawer__empty">{t('dashboard.waitingAnalysis')}</p>
                                    )}
                                </div>
                            </div>
                        </div>

                        <div className="drawer__footer">
                            <button className="btn btn-secondary" onClick={handleNewPlayer} disabled={isAnalyzing}>
                                <HiUserGroup /> {t('dashboard.newPlayer')}
                            </button>
                            <button className="btn btn-primary" onClick={() => navigate('/upload')}>
                                <HiArrowPath /> {t('dashboard.newVideo')}
                            </button>
                        </div>
            </aside>

            {/* Expanded minimap tactical board overlay */}
            {minimapExpanded && (
                <div className="minimap-board-overlay" onClick={() => setMinimapExpanded(false)}>
                    <div className="minimap-board" onClick={(e) => e.stopPropagation()}>
                        <MinimapOverlay
                            dataUrl={minimapDataUrl}
                            videoRef={heroVideoRef}
                            visible={true}
                            expanded={true}
                        />
                        <TelestrationCanvas
                            active={true}
                            parentRef={null}
                            width={900}
                            height={540}
                            onInteractionStart={() => {
                                if (heroVideoRef.current && !heroVideoRef.current.paused) {
                                    heroVideoRef.current.pause();
                                }
                            }}
                        />
                        <div className="minimap-board__toolbar">
                            <button
                                className="minimap-board__close"
                                onClick={() => setMinimapExpanded(false)}
                                title={t('dashboard.close')}
                            >✕</button>
                        </div>
                    </div>
                </div>
            )}

            {/* Tactical chart lightbox modal */}
            {tacticalModalChart && (
                <div className="minimap-board-overlay" onClick={() => setTacticalModalChart(null)}>
                    <div
                        className="tactical-modal-card"
                        onClick={(e) => e.stopPropagation()}
                        style={{
                            background: '#0f172a',
                            border: '1px solid rgba(255, 255, 255, 0.15)',
                            borderRadius: '16px',
                            padding: '20px',
                            maxWidth: '92vw',
                            maxHeight: '90vh',
                            display: 'flex',
                            flexDirection: 'column',
                            boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.75)',
                            position: 'relative',
                        }}
                    >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                            <div>
                                <h3 style={{ margin: 0, fontSize: '1.15rem', color: '#f8fafc', fontWeight: 600 }}>
                                    {tacticalModalChart.label}
                                </h3>
                                <p style={{ margin: '4px 0 0 0', fontSize: '0.8rem', color: '#94a3b8' }}>
                                    {tacticalModalChart.desc}
                                </p>
                            </div>
                            <button
                                className="minimap-board__close"
                                onClick={() => setTacticalModalChart(null)}
                                style={{ position: 'static', marginLeft: '16px' }}
                                title={t('dashboard.close')}
                            >
                                ✕
                            </button>
                        </div>
                        <div style={{ flex: 1, minHeight: 0, display: 'flex', justifyContent: 'center', alignItems: 'center', overflow: 'auto' }}>
                            {(() => {
                                const modalVer = tacticalVersion[tacticalModalChart.id];
                                const modalVParam = modalVer ? `&v=${modalVer}` : '';
                                const modalChartUrl = absUrl(`/api/sessions/${sessionId}/files/${tacticalModalChart.filename}${API_KEY ? `?key=${encodeURIComponent(API_KEY)}${modalVParam}` : (modalVer ? `?v=${modalVer}` : '')}`);
                                return (
                                    <img
                                        src={modalChartUrl}
                                        alt={tacticalModalChart.label}
                                        style={{ maxWidth: '100%', maxHeight: '72vh', objectFit: 'contain', borderRadius: '8px' }}
                                    />
                                );
                            })()}
                        </div>
                    </div>
                </div>
            )}

            {/* Tactical Highlight Video Lightbox Modal */}
            {activeHighlightModal && (
                <div className="minimap-board-overlay" onClick={() => setActiveHighlightModal(null)}>
                    <div
                        className="tactical-modal-card highlight-modal-card"
                        onClick={(e) => e.stopPropagation()}
                        style={{
                            background: '#0f172a',
                            border: '1px solid rgba(255, 255, 255, 0.15)',
                            borderRadius: '16px',
                            padding: '20px',
                            maxWidth: '850px',
                            width: '92vw',
                            display: 'flex',
                            flexDirection: 'column',
                            boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.8)',
                            position: 'relative',
                        }}
                    >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                            <div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <h3 style={{ margin: 0, fontSize: '1.15rem', color: '#f8fafc', fontWeight: 600 }}>
                                        {activeHighlightModal.metadata?.title || (language === 'zh' ? '战术高光镜头 (Broadcast HUD)' : 'Tactical Highlight (Broadcast HUD)')}
                                    </h3>
                                    <span className="coach-qa-result__badge">
                                        {activeHighlightModal.event_type?.toUpperCase() || 'MOMENT'}
                                    </span>
                                </div>
                                <p style={{ margin: '4px 0 0 0', fontSize: '0.8rem', color: '#94a3b8' }}>
                                    {activeHighlightModal.metadata?.description || (language === 'zh' ? `关键事件时刻: ${activeHighlightModal.event_time_s?.toFixed(1)}s (片段时长: ${activeHighlightModal.duration_s?.toFixed(1)}s)` : `Key Event: ${activeHighlightModal.event_time_s?.toFixed(1)}s (Duration: ${activeHighlightModal.duration_s?.toFixed(1)}s)`)}
                                </p>
                            </div>
                            <button
                                className="minimap-board__close"
                                onClick={() => setActiveHighlightModal(null)}
                                style={{ position: 'static', marginLeft: '16px' }}
                                title={t('dashboard.close')}
                            >
                                ✕
                            </button>
                        </div>
                        
                        <div className="highlight-modal-player-container">
                            {(() => {
                                const vidUrl = activeHighlightModal.video_url
                                    ? (activeHighlightModal.video_url.startsWith('http')
                                        ? activeHighlightModal.video_url
                                        : absUrl(`${activeHighlightModal.video_url}${API_KEY ? `?key=${encodeURIComponent(API_KEY)}` : ''}`))
                                    : (sessionId ? absUrl(`/api/sessions/${sessionId}/files/highlights/${activeHighlightModal.video_filename}${API_KEY ? `?key=${encodeURIComponent(API_KEY)}` : ''}`) : '');
                                return (
                                    <video
                                        src={vidUrl}
                                        controls
                                        autoPlay
                                        playsInline
                                        style={{
                                            width: '100%',
                                            maxHeight: '60vh',
                                            borderRadius: '10px',
                                            background: '#000',
                                            outline: 'none'
                                        }}
                                    />
                                );
                            })()}
                        </div>

                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '14px', flexWrap: 'wrap', gap: '10px' }}>
                            <div style={{ fontSize: '0.78rem', color: '#94a3b8' }}>
                                SHA-256: <code style={{ color: '#38bdf8' }}>{activeHighlightModal.sha256 ? activeHighlightModal.sha256.substring(0, 12) + '...' : t('dashboard.verified')}</code>
                            </div>
                            <div style={{ display: 'flex', gap: '8px' }}>
                                <button
                                    className="btn btn-xs btn-secondary"
                                    onClick={() => {
                                        if (heroVideoRef.current && typeof activeHighlightModal.event_time_s === 'number') {
                                            heroVideoRef.current.currentTime = Math.max(0, activeHighlightModal.event_time_s - 1.5);
                                            heroVideoRef.current.play().catch(() => {});
                                            setActiveHighlightModal(null);
                                        }
                                    }}
                                >
                                    <HiPlay /> {t('dashboard.syncWithMainScreen')}
                                </button>
                                {activeHighlightModal.video_filename && (
                                    <a
                                        href={absUrl(`/api/sessions/${sessionId}/files/highlights/${activeHighlightModal.video_filename}${API_KEY ? `?key=${encodeURIComponent(API_KEY)}` : ''}`)}
                                        download={activeHighlightModal.video_filename}
                                        className="btn btn-xs btn-primary"
                                        target="_blank"
                                        rel="noreferrer"
                                    >
                                        <HiArrowDownTray /> {t('dashboard.downloadMp4')}
                                    </a>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
