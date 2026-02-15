import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { HiMagnifyingGlass, HiPlus, HiChevronRight, HiArrowLeft } from 'react-icons/hi2';
import { IoFootball } from 'react-icons/io5';
import './PlayerLibrary.css';

const PLAYERS = [
    { id: 1, name: 'Marcus R.', number: 10, position: 'Forward', sessions: 12, avgSpeed: 28.2, emoji: '⚽' },
    { id: 2, name: 'James K.', number: 7, position: 'Midfielder', sessions: 8, avgSpeed: 25.1, emoji: '🏃' },
    { id: 3, name: 'Oscar T.', number: 4, position: 'Defender', sessions: 15, avgSpeed: 22.8, emoji: '🛡️' },
    { id: 4, name: 'Daniel P.', number: 9, position: 'Striker', sessions: 6, avgSpeed: 30.4, emoji: '🦵' },
    { id: 5, name: 'Ryan L.', number: 11, position: 'Winger', sessions: 10, avgSpeed: 31.0, emoji: '💨' },
    { id: 6, name: 'Leo V.', number: 3, position: 'Left Back', sessions: 9, avgSpeed: 24.6, emoji: '🧤' },
];

export default function PlayerLibrary() {
    const navigate = useNavigate();

    return (
        <div className="page-container player-lib">
            <div className="bg-grid" />

            <motion.div
                className="player-lib__header"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
            >
                <div className="player-lib__header-left">
                    <button className="btn btn-ghost" onClick={() => navigate('/')}>
                        <HiArrowLeft /> Back
                    </button>
                    <div>
                        <h1>Player Library</h1>
                        <p>{PLAYERS.length} players tracked</p>
                    </div>
                </div>
                <div className="player-lib__header-actions">
                    <div className="player-lib__search">
                        <HiMagnifyingGlass />
                        <input type="text" placeholder="Search players..." />
                    </div>
                    <button className="btn btn-primary">
                        <HiPlus /> Add Player
                    </button>
                </div>
            </motion.div>

            <motion.div
                className="player-lib__grid"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
            >
                {PLAYERS.map((player, i) => (
                    <motion.div
                        key={player.id}
                        className="player-card card"
                        onClick={() => navigate(`/players/${player.id}`)}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.05 * i }}
                        whileHover={{ y: -4 }}
                    >
                        <div className="player-card__top">
                            <span className="player-card__emoji">{player.emoji}</span>
                            <HiChevronRight className="player-card__arrow" />
                        </div>
                        <h3 className="player-card__name">{player.name}</h3>
                        <span className="player-card__pos">#{player.number} • {player.position}</span>
                        <div className="player-card__stats">
                            <div>
                                <span className="player-card__stat-value">{player.sessions}</span>
                                <span className="player-card__stat-label">Sessions</span>
                            </div>
                            <div>
                                <span className="player-card__stat-value">{player.avgSpeed}</span>
                                <span className="player-card__stat-label">Avg km/h</span>
                            </div>
                        </div>
                    </motion.div>
                ))}
            </motion.div>
        </div>
    );
}
