import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { HiUserGroup, HiArrowRight, HiArrowLeft, HiLink, HiPlus } from 'react-icons/hi2';
import { useProgress } from '../components/ProgressBar';
import StepNav from '../components/StepNav';
import './Configuration.css';

const MOCK_PLAYERS = [
    { id: 1, name: 'Marcus R.', number: 10, avatar: '⚽', detected: true },
    { id: 2, name: 'Player #7', number: 7, avatar: '🏃', detected: true },
    { id: 3, name: 'Player #4', number: 4, avatar: '🧤', detected: true },
    { id: 4, name: 'Player #9', number: 9, avatar: '🦵', detected: true },
    { id: 5, name: 'Player #11', number: 11, avatar: '💨', detected: true },
    { id: 6, name: 'Player #3', number: 3, avatar: '🛡️', detected: true },
];

export default function Configuration() {
    const [selected, setSelected] = useState(1);
    const [analyzing, setAnalyzing] = useState(false);
    const navigate = useNavigate();
    const { start, done } = useProgress();

    const selectPlayer = (id) => {
        setSelected(id);
    };

    const startAnalysis = () => {
        start();
        setAnalyzing(true);
        setTimeout(() => {
            done();
            navigate('/dashboard');
        }, 2200);
    };

    return (
        <div className="page-container config-page">
            <div className="bg-grid" />

            <motion.div
                className="config-page__back"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
            >
                <button className="btn btn-ghost" onClick={() => navigate('/trim')}>
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
                <p>Select a player to analyze</p>
            </motion.div>

            {/* Player Grid */}
            <motion.div
                className="config-grid"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
            >
                {MOCK_PLAYERS.map((player, i) => (
                    <motion.div
                        key={player.id}
                        className={`config-player-card ${selected === player.id ? 'config-player-card--selected' : ''}`}
                        onClick={() => selectPlayer(player.id)}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 * i }}
                        whileHover={{ y: -4 }}
                        whileTap={{ scale: 0.97 }}
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
                ))}
            </motion.div>

            {/* Options */}
            <motion.div
                className="config-options"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
            >
                <button
                    className="btn btn-secondary"
                    onClick={() => alert(`Linking ${MOCK_PLAYERS.find(p => p.id === selected)?.name} to an existing player...`)}
                    disabled={!selected}
                >
                    <HiLink /> Link to Player Library
                </button>
                <button
                    className="btn btn-secondary"
                    onClick={() => alert(`Adding ${MOCK_PLAYERS.find(p => p.id === selected)?.name} to Player Library...`)}
                    disabled={!selected}
                >
                    <HiPlus /> Add to Player Library
                </button>
            </motion.div>


            {/* CTA */}
            <motion.div
                className="config-cta"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.6 }}
            >
                <button
                    className={`btn btn-primary btn-lg ${analyzing ? 'btn--loading' : ''}`}
                    onClick={startAnalysis}
                    disabled={!selected || analyzing}
                >
                    {analyzing ? (
                        <>
                            <span className="config-spinner" />
                            Analyzing...
                        </>
                    ) : (
                        <>
                            Start Analysis
                            <HiArrowRight />
                        </>
                    )}
                </button>
                <p className="config-cta__hint">
                    {selected ? '1 player selected' : 'No player selected'}
                </p>
            </motion.div>
        </div>
    );
}
