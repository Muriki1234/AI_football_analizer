import { motion } from 'framer-motion';
import './Heatmap.css';

export default function Heatmap() {
    return (
        <motion.div
            className="heatmap-panel card"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 16 }}
        >
            <h3 className="heatmap-panel__title">Position Heatmap</h3>
            <div className="heatmap-pitch">
                {/* SVG Football Pitch */}
                <svg viewBox="0 0 680 440" className="heatmap-pitch__svg">
                    {/* Pitch outline */}
                    <rect x="10" y="10" width="660" height="420" rx="4" fill="none" stroke="rgba(148,163,184,0.2)" strokeWidth="2" />
                    {/* Center line */}
                    <line x1="340" y1="10" x2="340" y2="430" stroke="rgba(148,163,184,0.2)" strokeWidth="1.5" />
                    {/* Center circle */}
                    <circle cx="340" cy="220" r="60" fill="none" stroke="rgba(148,163,184,0.2)" strokeWidth="1.5" />
                    <circle cx="340" cy="220" r="3" fill="rgba(148,163,184,0.3)" />
                    {/* Penalty areas */}
                    <rect x="10" y="120" width="120" height="200" rx="2" fill="none" stroke="rgba(148,163,184,0.2)" strokeWidth="1.5" />
                    <rect x="550" y="120" width="120" height="200" rx="2" fill="none" stroke="rgba(148,163,184,0.2)" strokeWidth="1.5" />
                    {/* Goal areas */}
                    <rect x="10" y="170" width="45" height="100" rx="2" fill="none" stroke="rgba(148,163,184,0.15)" strokeWidth="1" />
                    <rect x="625" y="170" width="45" height="100" rx="2" fill="none" stroke="rgba(148,163,184,0.15)" strokeWidth="1" />

                    {/* Heatmap blobs */}
                    <defs>
                        <radialGradient id="heatGrad1">
                            <stop offset="0%" stopColor="#00e59b" stopOpacity="0.7" />
                            <stop offset="50%" stopColor="#00e59b" stopOpacity="0.25" />
                            <stop offset="100%" stopColor="#00e59b" stopOpacity="0" />
                        </radialGradient>
                        <radialGradient id="heatGrad2">
                            <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.65" />
                            <stop offset="50%" stopColor="#f59e0b" stopOpacity="0.2" />
                            <stop offset="100%" stopColor="#f59e0b" stopOpacity="0" />
                        </radialGradient>
                        <radialGradient id="heatGrad3">
                            <stop offset="0%" stopColor="#ef4444" stopOpacity="0.6" />
                            <stop offset="50%" stopColor="#ef4444" stopOpacity="0.2" />
                            <stop offset="100%" stopColor="#ef4444" stopOpacity="0" />
                        </radialGradient>
                    </defs>

                    <motion.ellipse cx="420" cy="200" rx="100" ry="80" fill="url(#heatGrad1)"
                        initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ duration: 0.8, delay: 0.2 }} />
                    <motion.ellipse cx="350" cy="280" rx="70" ry="60" fill="url(#heatGrad2)"
                        initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ duration: 0.8, delay: 0.4 }} />
                    <motion.ellipse cx="500" cy="160" rx="55" ry="50" fill="url(#heatGrad3)"
                        initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ duration: 0.8, delay: 0.6 }} />
                    <motion.ellipse cx="280" cy="220" rx="60" ry="45" fill="url(#heatGrad1)"
                        initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ duration: 0.8, delay: 0.8 }} />
                </svg>

                {/* Legend */}
                <div className="heatmap-legend">
                    <span><i style={{ background: '#00e59b' }} /> High Activity</span>
                    <span><i style={{ background: '#f59e0b' }} /> Medium</span>
                    <span><i style={{ background: '#ef4444' }} /> Sprint Zone</span>
                </div>
            </div>
        </motion.div>
    );
}
