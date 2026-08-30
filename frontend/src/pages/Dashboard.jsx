import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import toast from 'react-hot-toast';
import {
    HiHome, HiArrowPath, HiBars3, HiXMark, HiExclamationCircle,
    HiUserGroup, HiSparkles, HiChartBar, HiMapPin, HiFire,
    HiPlayCircle, HiArrowDownTray,
} from 'react-icons/hi2';
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
export default function Dashboard() {
    const location = useLocation();
    const navigate = useNavigate();

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
    const [viewMode, setViewMode] = useState('team'); // 'team' = 战术复盘, 'player' = 个人特训
    const [drawMode, setDrawMode] = useState(false);
    const [tacticalDrawings, setTacticalDrawings] = useState([]);
    const loadedDrawings = useRef(false);
    const [initialStrokes, setInitialStrokes] = useState([]);
    const [minimapExpanded, setMinimapExpanded] = useState(false);
    const [isVideoBuffering, setIsVideoBuffering] = useState(false);
    const telestrationRef = useRef(null);

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
        setAiSummaryTeam(null);
        setAiSummaryPlayer(null);

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
                    // Multi-segment path — pass period_idx and the match
                    // periods themselves so the backend can run period-aware
                    // SAMURAI + skip non-match frames in analysis/render.
                    const segments = multiSegments.map((seg) => ({
                        frame: seg.frame,
                        bbox: seg.bbox,
                        period_idx: seg.period_idx ?? 0,
                        img_dims: seg.img_dims,
                    }));
                    await startTrackingMulti(sessionId, segments, matchPeriodsFrames, location.state?.clientFps);
                    toast.success(`Tracking across ${segments.length} segments in parallel…`);
                } else if (selectedBbox && Array.isArray(selectedBbox) && selectedBbox.length === 4) {
                    const [x1, y1, x2, y2] = selectedBbox;
                    const imgDims = location.state?.imgDims || null;
                    await startTracking(sessionId, { x1, y1, x2, y2 }, 0, imgDims);
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
                if (t.task_type === 'ai_summary' || t.task_type === 'ai_summary_team') {
                    setAiSummaryTeam(t.result || null);
                    // Only update progress if we are in team mode, or if progress is generic
                    // We'll update aiProgress in a more comprehensive way below
                    setAiProgress(Math.max(0, Math.min(100, Number(t.progress) || 0)));
                } else if (t.task_type === 'ai_summary_player') {
                    setAiSummaryPlayer(t.result || null);
                    setAiProgress(Math.max(0, Math.min(100, Number(t.progress) || 0)));
                }
            }
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

                if (t.task_type === 'ai_summary' || t.task_type === 'ai_summary_team') {
                    setAiSummaryTeam(t.result || null);
                    setAiProgress(Math.max(0, Math.min(100, Number(t.progress) || 0)));
                } else if (t.task_type === 'ai_summary_player') {
                    setAiSummaryPlayer(t.result || null);
                    setAiProgress(Math.max(0, Math.min(100, Number(t.progress) || 0)));
                }
            },
        });

        return () => { cancelled = true; clearInterval(pollInterval); unsub(); };
    }, [sessionId, isFreshAnalysis]);

    // Fetch summary once analysis_done
    useEffect(() => {
        if (!isDone || summaryFetched.current) return;
        summaryFetched.current = true;
        getSummary(sessionId).then((s) => {
            if (s) {
                // Determine mode by task_type, or just set it to team by default
                if (s.task_type === 'ai_summary_player') {
                    setAiSummaryPlayer((prev) => prev || s);
                } else {
                    setAiSummaryTeam((prev) => prev || s);
                }
            }
        }).catch(() => { });
    }, [isDone, sessionId]);

    const minimapDataUrl = session?.minimap_data_url || null;
    const overlayDataUrl = session?.overlay_data_url || null;
    const heatmapDataUrl = session?.heatmap_data_url || null;

    const playerSummaryJson = session?.player_summary || null;
    
    const currentSummary = viewMode === 'player' ? aiSummaryPlayer : aiSummaryTeam;

    const aiMarkdown = useMemo(() => {
        const txt = taskTextResult(currentSummary);
        if (!txt) return '';
        try {
            let html = DOMPurify.sanitize(marked.parse(txt));
            // 将 [MM:SS] 时间戳转为可点击的跳转链接（在sanitize之后操作，安全）
            html = html.replace(
                /\[(\d{1,3}):(\d{2})\]/g,
                (match, m, s) => {
                    const sec = parseInt(m, 10) * 60 + parseInt(s, 10);
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
        const regex = /\[(\d{1,3}):(\d{2})\]/g;
        let match;
        // Keep track of added times to avoid duplicates
        const seen = new Set();
        while ((match = regex.exec(txt)) !== null) {
            const totalSec = parseInt(match[1], 10) * 60 + parseInt(match[2], 10);
            if (!seen.has(totalSec)) {
                seen.add(totalSec);
                highlights.push({ time: totalSec, label: `[${match[1]}:${match[2]}]` });
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
                    ? '个人特训报告生成中 — 约需 1 分钟'
                    : 'AI 战术分析生成中 — 约需 1 分钟'
            );
        } catch (e) {
            toast.error(e?.message || 'Failed to queue AI summary');
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
            toast('Wait for the current analysis to finish.');
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
                        .then(() => toast.success('战术画板已保存'))
                        .catch(err => {
                            console.error('Failed to save tactical drawings:', err);
                            toast.error('保存战术画板失败');
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
    }, [tacticalDrawings, sessionId]);

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
                    </div>

                    <div className="hero-video-card__body">
                        {session?.status === 'analysis_done' ? (
                            <div className="hero-video-card__player-wrap">
                                <video
                                    ref={heroVideoRef}
                                    crossOrigin="anonymous"
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
                                        <p>Buffering...</p>
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
                                    <span>Minimap Overlay</span>
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
                                    <span>战术画板</span>
                                    <span className={`drawer__toggle ${drawMode ? 'on' : ''}`}>
                                        {drawMode ? 'ON' : 'OFF'}
                                    </span>
                                </button>
                            </div>

                            {/* Data Analysis — always rendered */}
                            <div className="drawer__item is-static">
                                <div className="drawer__item-head drawer__item-head--static">
                                    <HiChartBar />
                                    <span>Data Analysis</span>
                                </div>
                                <DataAnalysisPanel playerSummary={playerSummaryJson} />
                            </div>

                            {/* AI Analysis — generate-on-demand */}
                            <div className="drawer__item is-static">
                                <div className="drawer__item-head drawer__item-head--static">
                                    <HiSparkles />
                                    <span>AI 智能教练</span>
                                </div>
                                <div className="drawer__section-body">
                                    {/* Mode toggle */}
                                    <div className="ai-mode-toggle">
                                        <button
                                            className={`ai-mode-toggle__btn ${viewMode === 'team' ? 'is-active' : ''}`}
                                            onClick={() => setViewMode('team')}
                                        >
                                            🌐 战术复盘
                                        </button>
                                        <button
                                            className={`ai-mode-toggle__btn ${viewMode === 'player' ? 'is-active' : ''}`}
                                            onClick={() => setViewMode('player')}
                                        >
                                            🎯 个人特训
                                        </button>
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
                                                        heroVideoRef.current.currentTime = sec;
                                                        heroVideoRef.current.play().catch(() => {});
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
                                                    <span>Generating AI summary…</span>
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
                                            <p>{viewMode === 'player' ? '生成你的专属私教报告' : '生成 AI 战术分析报告'}</p>
                                            <button className="btn btn-primary" onClick={handleGenerateAI}>
                                                <HiSparkles /> {viewMode === 'player' ? '🎯 开始个人特训分析' : '🌐 开始战术复盘分析'}
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
                                title="关闭"
                            >✕</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
