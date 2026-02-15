import { motion } from 'framer-motion';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import './VelocityChart.css';

const data = [
    { time: '0\'', speed: 5.2 },
    { time: '5\'', speed: 8.1 },
    { time: '10\'', speed: 12.4 },
    { time: '15\'', speed: 28.3 },
    { time: '20\'', speed: 15.2 },
    { time: '25\'', speed: 22.7 },
    { time: '30\'', speed: 31.5 },
    { time: '35\'', speed: 18.6 },
    { time: '40\'', speed: 9.3 },
    { time: '45\'', speed: 14.8 },
    { time: '50\'', speed: 26.1 },
    { time: '55\'', speed: 33.2 },
    { time: '60\'', speed: 19.4 },
    { time: '65\'', speed: 11.7 },
    { time: '70\'', speed: 24.9 },
    { time: '75\'', speed: 30.1 },
    { time: '80\'', speed: 16.3 },
    { time: '85\'', speed: 8.9 },
    { time: '90\'', speed: 6.1 },
];

const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload?.[0]) {
        return (
            <div className="velocity-tooltip">
                <p className="velocity-tooltip__time">{label}</p>
                <p className="velocity-tooltip__speed">{payload[0].value} km/h</p>
            </div>
        );
    }
    return null;
};

export default function VelocityChart() {
    return (
        <motion.div
            className="velocity-panel card"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 16 }}
        >
            <div className="velocity-panel__header">
                <h3>Velocity Analysis</h3>
                <span className="velocity-panel__peak">Peak: <strong>33.2 km/h</strong></span>
            </div>
            <div className="velocity-panel__chart">
                <ResponsiveContainer width="100%" height={220}>
                    <AreaChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                        <defs>
                            <linearGradient id="velocityGrad" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#00e59b" stopOpacity={0.4} />
                                <stop offset="100%" stopColor="#00e59b" stopOpacity={0.02} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
                        <XAxis dataKey="time" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                        <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} unit=" km/h" />
                        <Tooltip content={<CustomTooltip />} />
                        <Area
                            type="monotone"
                            dataKey="speed"
                            stroke="#00e59b"
                            strokeWidth={2}
                            fill="url(#velocityGrad)"
                            dot={false}
                            activeDot={{ r: 5, fill: '#00e59b', strokeWidth: 0 }}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </motion.div>
    );
}
