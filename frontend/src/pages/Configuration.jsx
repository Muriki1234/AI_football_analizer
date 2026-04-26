import { useState, useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import { HiUserGroup, HiArrowRight, HiArrowLeft } from 'react-icons/hi2';
import { analyzeFrame } from '../services/api';
import StepNav from '../components/StepNav';
import './Configuration.css';

export default function Configuration() {
    const [selected, setSelected] = useState(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [players, setPlayers] = useState([]);
    const [frameUrl, setFrameUrl] = useState(null);
    const [imgDims, setImgDims] = useState(null);
    const [detecting, setDetecting] = useState(false);
    const [detectError, setDetectError] = useState(null);

    const navigate = useNavigate();
    const location = useLocation();
    const videoId = location.state?.videoId || location.state?.sessionId;
    const detectCalled = useRef(false);

    useEffect(() => {
        // Guard against React StrictMode double-invoke and re-renders
        if (detectCalled.current) return;
        detectCalled.current = true;

        async function fetchPlayers() {
            if (!videoId) {
                setDetectError('No video. Go back to upload.');
                return;
            }
            try {
                setDetecting(true);
                setDetectError(null);
                const data = await analyzeFrame(videoId, 0);
                const formatted = (data.players_data || []).map((p, i) => ({
                    id: p.id || i + 1,
                    name: p.name || `Player ${i + 1}`,
                    number: p.number || '?',
                    avatar: p.avatar || '👤',
                    bbox: p.bbox,
                }));
                setPlayers(formatted);
                setFrameUrl(data.annotated_frame_url);
                setImgDims(data.image_dimensions);
                if (formatted.length > 0) setSelected(formatted[0].id);
                else setDetectError('No players detected on the first frame.');
            } catch (e) {
                console.error('Detect frame failed', e);
                setDetectError(
                    e?.response?.data?.detail ||
                        e?.message ||
                        'Failed to detect players.'
                );
            } finally {
                setDetecting(false);
            }
        }
        fetchPlayers();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [videoId]);

    const selectPlayer = (id) => setSelected(id);

    const startAnalysis = () => {
        if (!videoId) {
            navigate('/upload');
            return;
        }
        const player = players.find((p) => p.id === selected);
        setAnalyzing(true);
        // Pass selected bbox/player info forward so the dashboard can highlight it
        // once SAMURAI tracking is wired in. Global analysis doesn't need it.
        navigate('/dashboard', {
            state: {
                sessionId: videoId,
                videoId,
                playerId: selected,
                playerName: player?.name || 'Player',
                selectedBbox: player?.bbox || null,
            },
        });
    };

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
                    onClick={() => navigate('/upload')}
                    disabled={analyzing}
                >
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
                <h1>Detected Players</h1>
                <p>Select a player on the frame or from the list to analyze</p>
            </motion.div>

            <div className="config-split-view">
                {/* Left: annotated first frame with clickable bboxes */}
                <motion.div
                    className="config-frame-container"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.1 }}
                >
                    {detecting ? (
                        <div className="config-frame-placeholder">
                            <div className="config-loading-spinner" />
                            <span>Detecting players on first frame…</span>
                        </div>
                    ) : detectError ? (
                        <div className="config-frame-placeholder">
                            <span style={{ color: '#f87171' }}>{detectError}</span>
                        </div>
                    ) : frameUrl ? (
                        <div className="config-frame-wrapper">
                            <img
                                src={frameUrl}
                                alt="First frame with detected players"
                                className="config-frame-img"
                                onError={() => toast.error('Failed to load frame image')}
                            />
                            {imgDims && (
                                <svg
                                    viewBox={`0 0 ${imgDims.width} ${imgDims.height}`}
                                    className="config-frame-svg"
                                    preserveAspectRatio="none"
                                >
                                    {players
                                        .filter((p) => p.bbox)
                                        .map((player) => {
                                            const [x1, y1, x2, y2] = player.bbox;
                                            const isSelected = selected === player.id;
                                            return (
                                                <rect
                                                    key={player.id}
                                                    x={x1}
                                                    y={y1}
                                                    width={x2 - x1}
                                                    height={y2 - y1}
                                                    className={`config-bbox ${isSelected ? 'config-bbox--selected' : ''}`}
                                                    onClick={() => selectPlayer(player.id)}
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

                {/* Right: player list + start CTA */}
                <div className="config-sidebar">
                    <motion.div
                        className="config-grid config-grid--vertical"
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.2 }}
                    >
                        {detecting ? (
                            <div className="config-no-players">🔍 Detecting players…</div>
                        ) : players.length > 0 ? (
                            players.map((player, i) => (
                                <motion.div
                                    key={player.id}
                                    className={`config-player-card ${selected === player.id ? 'config-player-card--selected' : ''}`}
                                    onClick={() => selectPlayer(player.id)}
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.04 * i }}
                                    whileHover={{ y: -2 }}
                                    whileTap={{ scale: 0.98 }}
                                >
                                    <div className="config-player-card__avatar">{player.avatar}</div>
                                    <div className="config-player-card__info">
                                        <span className="config-player-card__name">{player.name}</span>
                                        <span className="config-player-card__number">#{player.number}</span>
                                    </div>
                                    <div className="config-player-card__check">
                                        {selected === player.id && '✓'}
                                    </div>
                                </motion.div>
                            ))
                        ) : (
                            <div className="config-no-players">No players detected</div>
                        )}
                    </motion.div>

                    <motion.div
                        className="config-cta"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.5 }}
                    >
                        <button
                            className={`btn btn-primary btn-lg w-full ${analyzing ? 'btn--loading' : ''}`}
                            onClick={startAnalysis}
                            disabled={analyzing || (!selected && players.length === 0)}
                        >
                            {analyzing ? (
                                <>
                                    <span className="config-spinner" />
                                    Starting…
                                </>
                            ) : (
                                <>
                                    Start Analysis
                                    <HiArrowRight />
                                </>
                            )}
                        </button>
                        <p className="config-cta__hint">
                            {players.length === 0
                                ? 'Waiting for detection…'
                                : selected
                                ? '1 player selected — full match analysis will run'
                                : 'Select a player to highlight'}
                        </p>
                    </motion.div>
                </div>
            </div>
        </div>
    );
}
