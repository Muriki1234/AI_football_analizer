import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { IoFootball } from 'react-icons/io5';
import { HiArrowRight, HiArrowRightOnRectangle, HiClock, HiXMark } from 'react-icons/hi2';
import { getRecentSessions, removeRecentSession, addRecentSession } from '../lib/recentSessions';
import { getSession, listMySessions } from '../services/api';
import './Welcome.css';

const formatRelative = (ts) => {
    const diff = Math.max(0, Date.now() - ts);
    const m = Math.floor(diff / 60000);
    if (m < 1) return 'just now';
    if (m < 60) return `${m}m ago`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}h ago`;
    return `${Math.floor(h / 24)}d ago`;
};

// Map raw session status → a friendly badge label + colour class.
// Three tiers matter for routing: 'needs-trim' | 'needs-pick' | 'in-progress' | 'done'
function sessionStage(s) {
    if (!s) return null;
    const st = s.status || 'uploaded';
    if (['analysis_done'].includes(st))
        return { label: 'Done', cls: 'badge--done', tier: 'done' };
    if (['tracking', 'samurai_multi_pending', 'samurai_done', 'analyzing', 'queued'].includes(st))
        return { label: 'Analyzing…', cls: 'badge--progress', tier: 'in-progress' };
    if (['tracking_failed', 'analysis_failed', 'failed'].includes(st))
        return { label: 'Failed', cls: 'badge--failed', tier: 'needs-pick' };
    // status === 'uploaded'
    const hasPeriods = Array.isArray(s.match_periods_sec) && s.match_periods_sec.length > 0;
    if (hasPeriods)
        return { label: 'Pick player', cls: 'badge--pick', tier: 'needs-pick' };
    return { label: 'Continue setup', cls: 'badge--setup', tier: 'needs-trim' };
}

export default function Welcome() {
    const navigate = useNavigate();

    const [recents, setRecents] = useState(() => getRecentSessions());
    // Live status fetched once per recent session (keyed by id)
    const [sessionMeta, setSessionMeta] = useState({});
    const [openingId, setOpeningId] = useState(null);  // which card is loading

    // 兜底：localStorage 里没有 recents 时（新浏览器/清过缓存/匿名 token
    // 翻新），从 DB 直接拉最近 5 条 session 灌进 recents，让用户能看到
    // 自己之前传过的视频，而不是面对一个空主页。
    useEffect(() => {
        if (recents.length > 0) return undefined;
        let cancelled = false;
        listMySessions({ limit: 5 }).then((rows) => {
            if (cancelled || !rows || rows.length === 0) return;
            const seeded = rows.map((row) => ({
                id: row.id,
                fileName: row.fileName,
                videoUrl: row.video_url,
                size: 0,
                addedAt: new Date(row.updated_at || row.created_at).getTime(),
            }));
            // 写回 localStorage，下次进 home 不需要再查 DB
            seeded.forEach((r) => addRecentSession(r));
            setRecents(getRecentSessions());
            // 同时把 row 自己当 meta 填进去，省一次 getSession 往返
            setSessionMeta((prev) => {
                const next = { ...prev };
                rows.forEach((row) => { next[row.id] = row; });
                return next;
            });
        }).catch((e) => {
            console.warn('Failed to seed recents from DB:', e);
        });
        return () => { cancelled = true; };
    }, [recents.length]);

    // Fetch live session data for each recent item on mount so we can show
    // up-to-date status badges without the user having to click anything.
    // Uses Promise.allSettled so one failure doesn't drop the rest, and an
    // AbortController-style flag to skip setState after unmount.
    useEffect(() => {
        if (recents.length === 0) return undefined;
        let cancelled = false;
        const ids = recents.map((r) => r.id);

        Promise.allSettled(ids.map((id) => getSession(id))).then((results) => {
            if (cancelled) return;
            setSessionMeta((prev) => {
                const next = { ...prev };
                results.forEach((res, i) => {
                    if (res.status === 'fulfilled') next[ids[i]] = res.value;
                    // rejected → stale / deleted; leave badge hidden
                });
                return next;
            });
        });

        return () => { cancelled = true; };
    }, [recents]);

    const handleOpenRecent = async (sessionId) => {
        setOpeningId(sessionId);
        try {
            const s = await getSession(sessionId);
            const stage = sessionStage(s);

            if (stage?.tier === 'done' || stage?.tier === 'in-progress') {
                // Analysis started or complete → show dashboard
                navigate(`/dashboard?sessionId=${encodeURIComponent(sessionId)}`, {
                    state: { sessionId },
                });
                return;
            }

            if (stage?.tier === 'needs-pick') {
                // Match periods already saved → skip /trim, go straight to player pick
                navigate(`/configure-multi?sessionId=${encodeURIComponent(sessionId)}`, {
                    state: {
                        sessionId,
                        videoId: sessionId,
                        matchPeriods: s.match_periods_sec,
                    },
                });
                return;
            }

            // 'needs-trim' or unknown → start at the trim/periods page
            navigate(`/trim?sessionId=${encodeURIComponent(sessionId)}`, {
                state: { sessionId, videoId: sessionId },
            });
        } catch (e) {
            // Session deleted or network error — fall back to dashboard empty state
            console.warn('Recent session lookup failed:', e);
            navigate(`/dashboard?sessionId=${encodeURIComponent(sessionId)}`, {
                state: { sessionId },
            });
        } finally {
            setOpeningId(null);
        }
    };

    const handleRemoveRecent = (e, sessionId) => {
        e.stopPropagation();
        removeRecentSession(sessionId);
        setRecents(getRecentSessions());
    };

    return (
        <div className="welcome">
            <div className="bg-grid" />

            {/* Top bar */}
            <motion.div
                className="welcome__topbar"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
            >
                <button className="btn btn-ghost" onClick={() => navigate('/login')}>
                    <HiArrowRightOnRectangle /> Login
                </button>
            </motion.div>

            {/* Floating orbs */}
            <motion.div
                className="welcome__orb welcome__orb--1"
                animate={{ y: [0, -30, 0], x: [0, 15, 0] }}
                transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }}
            />
            <motion.div
                className="welcome__orb welcome__orb--2"
                animate={{ y: [0, 20, 0], x: [0, -20, 0] }}
                transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }}
            />

            <motion.div
                className="welcome__content"
                initial={{ opacity: 0, y: 40 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, ease: 'easeOut' }}
            >
                {/* Logo */}
                <motion.div
                    className="welcome__logo"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
                >
                    <IoFootball />
                </motion.div>

                <motion.h1
                    className="welcome__title"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                >
                    Pitch<span className="text-gradient">Logic</span> AI
                </motion.h1>

                <motion.p
                    className="welcome__subtitle"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.6 }}
                >
                    AI-powered football performance analysis. Upload match footage, get instant tactical insights.
                </motion.p>

                {/* CTA Button */}
                <motion.div
                    className="welcome__cta-wrapper"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.8, type: 'spring' }}
                >
                    <div className="welcome__pulse-ring" />
                    <div className="welcome__pulse-ring welcome__pulse-ring--delay" />
                    <button
                        className="btn btn-primary btn-lg welcome__cta"
                        onClick={() => navigate('/upload')}
                    >
                        Start Football Analysis
                        <HiArrowRight />
                    </button>
                </motion.div>

                {/* Feature pills */}
                <motion.div
                    className="welcome__features"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 1.1 }}
                >
                    {['AI Skeleton Tracking', 'Heatmap Generation', 'Tactical Reports'].map((f, i) => (
                        <motion.span
                            key={f}
                            className="welcome__feature-pill"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 1.2 + i * 0.1 }}
                        >
                            {f}
                        </motion.span>
                    ))}
                </motion.div>

                <AnimatePresence>
                    {recents.length > 0 && (
                        <motion.div
                            className="welcome__recents"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: 10 }}
                            transition={{ delay: 1.5 }}
                        >
                            <div className="welcome__recents-header">
                                <span><HiClock /> Recent uploads</span>
                                <button
                                    className="welcome__recents-view-all"
                                    onClick={() => navigate('/sessions')}
                                >
                                    View all →
                                </button>
                            </div>
                            <ul className="welcome__recents-list">
                                {recents.map((r) => {
                                    const stage = sessionStage(sessionMeta[r.id]);
                                    const isOpening = openingId === r.id;
                                    return (
                                        <li
                                            key={r.id}
                                            className={`welcome__recent-item${isOpening ? ' is-opening' : ''}`}
                                            onClick={() => !isOpening && handleOpenRecent(r.id)}
                                        >
                                            <span className="welcome__recent-name" title={r.fileName}>
                                                {r.fileName}
                                            </span>
                                            <span className="welcome__recent-time">
                                                {formatRelative(r.addedAt)}
                                            </span>
                                            {/* Status badge — shows once live data arrives */}
                                            {stage && !isOpening && (
                                                <span className={`welcome__recent-badge ${stage.cls}`}>
                                                    {stage.label}
                                                </span>
                                            )}
                                            {isOpening && (
                                                <span className="welcome__recent-spinner" />
                                            )}
                                            <button
                                                className="welcome__recent-remove"
                                                onClick={(e) => handleRemoveRecent(e, r.id)}
                                                aria-label="Remove from recent"
                                            >
                                                <HiXMark />
                                            </button>
                                        </li>
                                    );
                                })}
                            </ul>
                        </motion.div>
                    )}
                </AnimatePresence>
            </motion.div>
        </div>
    );
}
