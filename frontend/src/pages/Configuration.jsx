import { useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { HiUserGroup, HiArrowRight, HiArrowLeft, HiWrenchScrewdriver } from 'react-icons/hi2';
import StepNav from '../components/StepNav';
import './Configuration.css';

export default function Configuration() {
    const navigate = useNavigate();
    const location = useLocation();
    const videoId = location.state?.videoId || location.state?.sessionId;

    const goDashboard = () => {
        if (!videoId) {
            navigate('/upload');
            return;
        }
        navigate('/dashboard', { state: { sessionId: videoId, videoId } });
    };

    return (
        <div className="page-container config-page">
            <div className="bg-grid" />

            <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                style={{ padding: '16px 24px' }}
            >
                <button className="btn btn-ghost" onClick={() => navigate('/trim', { state: { videoId, sessionId: videoId } })}>
                    <HiArrowLeft /> Back
                </button>
            </motion.div>

            <StepNav />

            <motion.div
                className="config-page__header"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
            >
                <HiUserGroup className="config-page__icon" />
                <h1>Player Tracking</h1>
                <p>Per-player deep tracking is coming soon. For now, analyze the full match automatically.</p>
            </motion.div>

            <motion.div
                className="config-split-view"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                style={{
                    gridTemplateColumns: '1fr',
                    maxWidth: 640,
                    margin: '32px auto',
                    padding: 32,
                    borderRadius: 16,
                    background: 'rgba(17, 24, 39, 0.6)',
                    border: '1px solid #334155',
                    textAlign: 'center',
                }}
            >
                <HiWrenchScrewdriver style={{ fontSize: 48, color: '#94a3b8', margin: '0 auto 16px' }} />
                <h3 style={{ color: '#e5e7eb', marginBottom: 12 }}>Single-Player Tracker</h3>
                <p style={{ color: '#9ca3af', marginBottom: 24 }}>
                    The SAMURAI per-player tracker requires an extra GPU model that's not wired up in the current build.
                    Skip this step and use the full-match analysis &mdash; it covers all 22 players, heatmaps, possession,
                    sprints, and the AI summary.
                </p>
                <button
                    className="btn btn-primary btn-lg"
                    onClick={goDashboard}
                    disabled={!videoId}
                    style={{ margin: '0 auto' }}
                >
                    Analyze Full Match
                    <HiArrowRight />
                </button>
                {!videoId && (
                    <p style={{ color: '#f87171', marginTop: 16, fontSize: 13 }}>
                        No video in session &mdash; go back to upload.
                    </p>
                )}
            </motion.div>
        </div>
    );
}
