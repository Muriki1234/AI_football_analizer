import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { HiArrowTrendingUp, HiBolt, HiHeart } from 'react-icons/hi2';
import './RunningStats.css';

const STATS = [
    { icon: HiArrowTrendingUp, label: 'Total Distance', value: 10.8, unit: 'km', color: '#00e59b' },
    { icon: HiBolt, label: 'Sprint Count', value: 23, unit: 'sprints', color: '#f59e0b' },
    { icon: HiHeart, label: 'Endurance Index', value: 87, unit: '%', color: '#7c3aed' },
];

function AnimatedNumber({ target, decimals = 0 }) {
    const [current, setCurrent] = useState(0);

    useEffect(() => {
        let frame;
        const start = performance.now();
        const duration = 1200;

        const animate = (now) => {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            setCurrent(eased * target);
            if (progress < 1) frame = requestAnimationFrame(animate);
        };

        frame = requestAnimationFrame(animate);
        return () => cancelAnimationFrame(frame);
    }, [target]);

    return <span>{current.toFixed(decimals)}</span>;
}

export default function RunningStats() {
    return (
        <motion.div
            className="running-stats"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 16 }}
        >
            {STATS.map((stat, i) => (
                <motion.div
                    key={stat.label}
                    className="stat-card"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.15 }}
                >
                    <stat.icon className="stat-card__icon" style={{ color: stat.color }} />
                    <div className="stat-card__value">
                        <AnimatedNumber target={stat.value} decimals={stat.label === 'Total Distance' ? 1 : 0} />
                    </div>
                    <div className="stat-card__unit">{stat.unit}</div>
                    <div className="stat-card__label">{stat.label}</div>
                </motion.div>
            ))}
        </motion.div>
    );
}
