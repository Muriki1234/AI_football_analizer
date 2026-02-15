import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { HiArrowLeft, HiChartBar } from 'react-icons/hi2';
import './PlayerProfile.css';

const PLAYER_DATA = {
    1: { name: 'Marcus R.', number: 10, position: 'Forward', emoji: '⚽' },
    2: { name: 'James K.', number: 7, position: 'Midfielder', emoji: '🏃' },
    3: { name: 'Oscar T.', number: 4, position: 'Defender', emoji: '🛡️' },
    4: { name: 'Daniel P.', number: 9, position: 'Striker', emoji: '🦵' },
    5: { name: 'Ryan L.', number: 11, position: 'Winger', emoji: '💨' },
    6: { name: 'Leo V.', number: 3, position: 'Left Back', emoji: '🧤' },
};

const TREND_DATA = [
    { session: 'Jan 5', speed: 24.1, positioning: 6.2, stamina: 72 },
    { session: 'Jan 12', speed: 25.3, positioning: 6.5, stamina: 74 },
    { session: 'Jan 20', speed: 26.0, positioning: 6.8, stamina: 76 },
    { session: 'Jan 28', speed: 25.8, positioning: 7.0, stamina: 78 },
    { session: 'Feb 3', speed: 27.5, positioning: 7.3, stamina: 80 },
    { session: 'Feb 7', speed: 28.2, positioning: 7.6, stamina: 82 },
    { session: 'Feb 10', speed: 29.0, positioning: 7.8, stamina: 85 },
    { session: 'Feb 13', speed: 28.5, positioning: 8.1, stamina: 87 },
];

const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload?.length) {
        return (
            <div className="profile-tooltip">
                <p className="profile-tooltip__label">{label}</p>
                {payload.map((p) => (
                    <p key={p.name} style={{ color: p.color, fontSize: '0.82rem' }}>
                        {p.name}: <strong>{p.value}</strong>
                    </p>
                ))}
            </div>
        );
    }
    return null;
};

export default function PlayerProfile() {
    const { id } = useParams();
    const navigate = useNavigate();
    const player = PLAYER_DATA[id] || PLAYER_DATA[1];

    return (
        <div className="page-container profile-page">
            <div className="bg-grid" />

            <motion.div
                className="profile-page__header"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
            >
                <button className="btn btn-ghost" onClick={() => navigate('/players')}>
                    <HiArrowLeft /> Back to Library
                </button>
            </motion.div>

            {/* Player info */}
            <motion.div
                className="profile-hero card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
            >
                <div className="profile-hero__left">
                    <span className="profile-hero__avatar">{player.emoji}</span>
                    <div>
                        <h1 className="profile-hero__name">{player.name}</h1>
                        <p className="profile-hero__meta">#{player.number} • {player.position}</p>
                    </div>
                </div>
                <div className="profile-hero__stats">
                    <div className="stat-card">
                        <div className="stat-card__value">8</div>
                        <div className="stat-card__label">Sessions</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-card__value">28.5</div>
                        <div className="stat-card__label">Peak km/h</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-card__value">87%</div>
                        <div className="stat-card__label">Stamina</div>
                    </div>
                </div>
            </motion.div>

            {/* Trend Charts */}
            <motion.div
                className="profile-charts"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
            >
                <div className="profile-chart card">
                    <div className="profile-chart__header">
                        <HiChartBar className="profile-chart__icon" />
                        <h3>Performance Trends</h3>
                    </div>
                    <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={TREND_DATA} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
                            <XAxis dataKey="session" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                            <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                            <Tooltip content={<CustomTooltip />} />
                            <Legend
                                wrapperStyle={{ fontSize: '0.8rem', paddingTop: '12px' }}
                                iconType="circle"
                            />
                            <Line type="monotone" dataKey="speed" stroke="#00e59b" strokeWidth={2} dot={{ r: 4, fill: '#00e59b' }} name="Speed (km/h)" />
                            <Line type="monotone" dataKey="positioning" stroke="#7c3aed" strokeWidth={2} dot={{ r: 4, fill: '#7c3aed' }} name="Positioning" />
                            <Line type="monotone" dataKey="stamina" stroke="#f59e0b" strokeWidth={2} dot={{ r: 4, fill: '#f59e0b' }} name="Stamina (%)" />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </motion.div>
        </div>
    );
}
