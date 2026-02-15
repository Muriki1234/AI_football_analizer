import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { IoFootball } from 'react-icons/io5';
import { HiArrowRight, HiUserGroup, HiArrowRightOnRectangle } from 'react-icons/hi2';
import './Welcome.css';

export default function Welcome() {
    const navigate = useNavigate();

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
                <button className="btn btn-ghost" onClick={() => navigate('/players')}>
                    <HiUserGroup /> Player Library
                </button>
                <button className="btn btn-ghost" onClick={() => navigate('/login')}>
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
            </motion.div>
        </div>
    );
}
