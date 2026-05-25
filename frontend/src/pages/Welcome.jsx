import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import { IoFootball } from 'react-icons/io5';
import { HiArrowRight, HiUserGroup, HiArrowRightOnRectangle, HiClock, HiXMark } from 'react-icons/hi2';
import { getRecentSessions, removeRecentSession } from '../lib/recentSessions';
import { getSession } from '../services/api';
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

export default function Welcome() {
    const navigate = useNavigate();
    const showInDevelopment = () => toast('Feature in development');

    const [recents, setRecents] = useState(() => getRecentSessions());

    const handleOpenRecent = async (sessionId) => {
        // Route based on where the session is in its lifecycle:
        //   - "uploaded" (never analyzed) → /trim to continue from where they left off
        //   - "tracking" / "analyzing" / "analysis_done" / "*_failed" → /dashboard
        // Without this, clicking an "uploaded-only" session would land on a
        // Dashboard with nothing to show but a "No replay yet" placeholder.
        try {
            const s = await getSession(sessionId);
            const status = s?.status || 'uploaded';
            const stillNeedsPicker = status === 'uploaded' || !s?.samurai_cache_path;
            if (stillNeedsPicker && status !== 'analysis_done') {
                navigate(`/trim?sessionId=${encodeURIComponent(sessionId)}`, {
                    state: { sessionId, videoId: sessionId },
                });
                return;
            }
        } catch (e) {
            // If session lookup fails (deleted / network), fall through to
            // /dashboard which will show its own "No session" empty state.
            console.warn('Recent session lookup failed:', e);
        }
        navigate(`/dashboard?sessionId=${encodeURIComponent(sessionId)}`, {
            state: { sessionId },
        });
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
                <button className="btn btn-ghost" onClick={showInDevelopment}>
                    <HiUserGroup /> Player Library
                </button>
                <button className="btn btn-ghost" onClick={showInDevelopment}>
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
                                {recents.map((r) => (
                                    <li
                                        key={r.id}
                                        className="welcome__recent-item"
                                        onClick={() => handleOpenRecent(r.id)}
                                    >
                                        <span className="welcome__recent-name" title={r.fileName}>
                                            {r.fileName}
                                        </span>
                                        <span className="welcome__recent-time">
                                            {formatRelative(r.addedAt)}
                                        </span>
                                        <button
                                            className="welcome__recent-remove"
                                            onClick={(e) => handleRemoveRecent(e, r.id)}
                                            aria-label="Remove from recent"
                                        >
                                            <HiXMark />
                                        </button>
                                    </li>
                                ))}
                            </ul>
                        </motion.div>
                    )}
                </AnimatePresence>
            </motion.div>
        </div>
    );
}
