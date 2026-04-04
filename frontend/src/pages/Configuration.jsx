import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { analyzeFrame, registerSession, startTracking } from '../services/api';
import { motion, AnimatePresence } from 'framer-motion';
import { HiUserGroup, HiArrowRight, HiLink, HiPlus } from 'react-icons/hi2';
import { useProgress } from '../components/ProgressBar';
import StepNav from '../components/StepNav';
import './Configuration.css';

export default function Configuration() {
    const [selected, setSelected] = useState(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [players, setPlayers] = useState([]);
    const [frameUrl, setFrameUrl] = useState(null);
    const [imgDims, setImgDims] = useState(null);
    const [detecting, setDetecting] = useState(false);

    const navigate = useNavigate();
    const location = useLocation();
    const { start, done } = useProgress();

    const videoId = location.state?.videoId;
    // Support both: coming from Upload (just videoId) or legacy Trimmer (with preLoadedPlayers)
    const preLoadedPlayers = location.state?.detectedPlayers;
    const preLoadedFrameUrl = location.state?.frameUrl;
    const preLoadedImgDims = location.state?.imgDims;

    useEffect(() => {
        async function fetchPlayers() {
            if (preLoadedPlayers) {
                // Legacy: data passed from Trimmer
                const formattedPlayers = preLoadedPlayers.map((p, i) => ({
                    id: p.id || i + 1,
                    name: p.name || `Detected Player ${i + 1}`,
                    number: p.number || '?',
                    avatar: p.avatar || '👤',
                    bbox: p.bbox
                }));
                setPlayers(formattedPlayers);
                setFrameUrl(preLoadedFrameUrl || null);
                setImgDims(preLoadedImgDims || null);
                if (formattedPlayers.length > 0) setSelected(formattedPlayers[0].id);
            } else if (videoId) {
                // New: detect players from first frame directly
                try {
                    setDetecting(true);
                    const data = await analyzeFrame(videoId, 0);
                    const formattedPlayers = (data.players_data || []).map((p, i) => ({
                        id: p.id || i + 1,
                        name: p.name || `Player ${i + 1}`,
                        number: p.number || '?',
                        avatar: p.avatar || '👤',
                        bbox: p.bbox
                    }));
                    setPlayers(formattedPlayers);
                    if (data.annotated_frame_url) setFrameUrl(data.annotated_frame_url);
                    if (data.image_dimensions) setImgDims(data.image_dimensions);
                    if (formattedPlayers.length > 0) setSelected(formattedPlayers[0].id);
                } catch (e) {
                    console.error("Failed to detect players", e);
                } finally {
                    setDetecting(false);
                }
            } else {
                setPlayers([]);
            }
        }
        fetchPlayers();
    }, [videoId]);

    const selectPlayer = (id) => {
        setSelected(id);
    };

    const startAnalysis = async () => {
        if (selected && videoId) {
            try {
                start();
                setAnalyzing(true);

                const selectedPlayer = players.find(p => p.id === selected);
                const bbox = selectedPlayer?.bbox;

                // 1. Register video with the analysis pipeline
                await registerSession(videoId, location.state?.videoPath || '');

                // 2. Start SAMURAI tracking with the selected player bbox
                if (bbox) {
                    await startTracking(videoId, bbox, 0);
                }

                done();
                // Navigate to dashboard — pass sessionId (=videoId) so it can poll
                navigate('/dashboard', {
                    state: {
                        sessionId: videoId,
                        videoId,
                        playerId: selected,
                        playerName: selectedPlayer?.name || 'Player'
                    }
                });
            } catch (e) {
                console.error('Analysis failed to start', e);
                setAnalyzing(false);
                done();
                alert('Failed to start analysis: ' + (e.message || 'Unknown error'));
            }
        } else {
            start();
            setAnalyzing(true);
            setTimeout(() => {
                done();
                navigate('/dashboard');
            }, 800);
        }
    };

    return (
        <div className="page-container config-page">
            <div className="bg-grid" />




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
                {/* Left Side: Frame Overlay */}
                <motion.div
                    className="config-frame-container"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.1 }}
                >
                    {frameUrl ? (
                        <div className="config-frame-wrapper">
                            <img src={frameUrl} alt="Analyzed Frame" className="config-frame-img" />
                            {imgDims && (
                                <svg
                                    viewBox={`0 0 ${imgDims.width} ${imgDims.height}`}
                                    className="config-frame-svg"
                                >
                                    {players.filter(p => p.bbox).map((player) => {
                                        const [x1, y1, x2, y2] = player.bbox;
                                        const width = x2 - x1;
                                        const height = y2 - y1;
                                        const isSelected = selected === player.id;

                                        return (
                                            <rect
                                                key={player.id}
                                                x={x1}
                                                y={y1}
                                                width={width}
                                                height={height}
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

                {/* Right Side: Player Grid and Options */}
                <div className="config-sidebar">
                    <motion.div
                        className="config-grid config-grid--vertical"
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.2 }}
                    >
                        {detecting ? (
                            <div className="config-no-players">🔍 Detecting players...</div>
                        ) : players.length > 0 ? players.map((player, i) => (
                            <motion.div
                                key={player.id}
                                className={`config-player-card ${selected === player.id ? 'config-player-card--selected' : ''}`}
                                onClick={() => selectPlayer(player.id)}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.05 * i }}
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
                        )) : (
                            <div className="config-no-players">Detecting players...</div>
                        )}
                    </motion.div>

                    <motion.div
                        className="config-options"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.4 }}
                    >
                        <button
                            className="btn btn-secondary btn-sm"
                            onClick={() => alert(`Linking ${players.find(p => p.id === selected)?.name} to an existing player...`)}
                            disabled={!selected}
                        >
                            <HiLink /> Link to Library
                        </button>
                        <button
                            className="btn btn-secondary btn-sm"
                            onClick={() => alert(`Adding ${players.find(p => p.id === selected)?.name} to Library...`)}
                            disabled={!selected}
                        >
                            <HiPlus /> Add to Library
                        </button>
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
                            disabled={!selected || analyzing}
                        >
                            {analyzing ? (
                                <>
                                    <span className="config-spinner" />
                                    Starting...
                                </>
                            ) : (
                                <>
                                    Start Deep Tracking
                                    <HiArrowRight />
                                </>
                            )}
                        </button>
                        <p className="config-cta__hint">
                            {selected ? '1 player selected' : 'Select a player to track'}
                        </p>
                    </motion.div>
                </div>
            </div>
        </div>
    );
}
