import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { HiPlay, HiPause, HiEye, HiArrowLeft } from 'react-icons/hi2';
import Heatmap from '../components/Heatmap';
import VelocityChart from '../components/VelocityChart';
import RunningStats from '../components/RunningStats';
import AIInsights from '../components/AIInsights';
import StepNav from '../components/StepNav';
import './Dashboard.css';

const PANELS = [
    { key: 'heatmap', label: 'Heatmap' },
    { key: 'velocity', label: 'Velocity' },
    { key: 'running', label: 'Running Stats' },
];

export default function Dashboard() {
    const [activePanels, setActivePanels] = useState(new Set(['heatmap']));
    const [playing, setPlaying] = useState(false);

    const togglePanel = (key) => {
        setActivePanels((prev) => {
            const next = new Set(prev);
            next.has(key) ? next.delete(key) : next.add(key);
            return next;
        });
    };

    return (
        <div className="dashboard">
            <div className="bg-grid" />

            <StepNav />

            {/* Header */}
            <motion.div
                className="dashboard__header"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
            >
                <div className="dashboard__header-left">
                    <button className="btn btn-ghost" onClick={() => window.location.href = '/configure'}>
                        <HiArrowLeft /> Back
                    </button>
                    <div>
                        <h1 className="dashboard__title">Analysis Dashboard</h1>
                        <p className="dashboard__subtitle">Match analysis for Marcus R. — Feb 13, 2026</p>
                    </div>
                </div>
                <div className="dashboard__header-actions">
                    <button className="btn btn-ghost" onClick={() => window.location.href = '/players'}>
                        <HiEye /> Player Library
                    </button>
                </div>
            </motion.div>

            <div className="dashboard__layout">
                {/* Main Viewport */}
                <motion.div
                    className="dashboard__main"
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.15 }}
                >
                    <div className="dashboard__video-viewport">
                        {/* Simulated analysed video with overlays */}
                        <div className="dashboard__video-placeholder">
                            {/* Skeleton tracking overlay */}
                            <svg className="dashboard__skeleton-overlay" viewBox="0 0 640 360">
                                {/* Player silhouette with skeleton lines */}
                                <g className="dashboard__skeleton-figure" transform="translate(300, 160)">
                                    {/* Head */}
                                    <circle cx="0" cy="-50" r="12" fill="none" stroke="#00e59b" strokeWidth="2" opacity="0.9" />
                                    {/* Torso */}
                                    <line x1="0" y1="-38" x2="0" y2="10" stroke="#00e59b" strokeWidth="2" opacity="0.8" />
                                    {/* Arms */}
                                    <line x1="0" y1="-30" x2="-25" y2="-10" stroke="#00e59b" strokeWidth="2" opacity="0.7" />
                                    <line x1="0" y1="-30" x2="25" y2="-15" stroke="#00e59b" strokeWidth="2" opacity="0.7" />
                                    {/* Legs */}
                                    <line x1="0" y1="10" x2="-18" y2="50" stroke="#00e59b" strokeWidth="2" opacity="0.7" />
                                    <line x1="0" y1="10" x2="15" y2="48" stroke="#00e59b" strokeWidth="2" opacity="0.7" />
                                    {/* Joints */}
                                    <circle cx="0" cy="-30" r="3" fill="#00e59b" opacity="0.9" />
                                    <circle cx="-25" cy="-10" r="3" fill="#00e59b" opacity="0.9" />
                                    <circle cx="25" cy="-15" r="3" fill="#00e59b" opacity="0.9" />
                                    <circle cx="0" cy="10" r="3" fill="#00e59b" opacity="0.9" />
                                    <circle cx="-18" cy="50" r="3" fill="#00e59b" opacity="0.9" />
                                    <circle cx="15" cy="48" r="3" fill="#00e59b" opacity="0.9" />
                                </g>

                                {/* Highlight ring around player */}
                                <circle cx="300" cy="180" r="40" fill="none" stroke="#00e59b" strokeWidth="1.5" opacity="0.4">
                                    <animate attributeName="r" values="40;50;40" dur="2s" repeatCount="indefinite" />
                                    <animate attributeName="opacity" values="0.4;0.1;0.4" dur="2s" repeatCount="indefinite" />
                                </circle>
                                <circle cx="300" cy="180" r="50" fill="none" stroke="#00e59b" strokeWidth="1" opacity="0.2">
                                    <animate attributeName="r" values="50;65;50" dur="2s" repeatCount="indefinite" />
                                    <animate attributeName="opacity" values="0.2;0;0.2" dur="2s" repeatCount="indefinite" />
                                </circle>

                                {/* Second player */}
                                <g className="dashboard__skeleton-figure" transform="translate(460, 200)">
                                    <circle cx="0" cy="-40" r="10" fill="none" stroke="#7c3aed" strokeWidth="1.5" opacity="0.7" />
                                    <line x1="0" y1="-30" x2="0" y2="5" stroke="#7c3aed" strokeWidth="1.5" opacity="0.6" />
                                    <line x1="0" y1="-22" x2="-20" y2="-5" stroke="#7c3aed" strokeWidth="1.5" opacity="0.5" />
                                    <line x1="0" y1="-22" x2="18" y2="-8" stroke="#7c3aed" strokeWidth="1.5" opacity="0.5" />
                                    <line x1="0" y1="5" x2="-12" y2="38" stroke="#7c3aed" strokeWidth="1.5" opacity="0.5" />
                                    <line x1="0" y1="5" x2="14" y2="35" stroke="#7c3aed" strokeWidth="1.5" opacity="0.5" />
                                </g>

                                {/* Label badge */}
                                <rect x="253" y="90" width="94" height="22" rx="11" fill="rgba(0,229,155,0.15)" stroke="#00e59b" strokeWidth="0.5" />
                                <text x="300" y="105" textAnchor="middle" fill="#00e59b" fontSize="11" fontFamily="Inter" fontWeight="600">Marcus R.</text>
                            </svg>

                            <div className="dashboard__video-label">AI-Tracked Playback</div>
                        </div>

                        {/* Playback controls */}
                        <div className="dashboard__controls">
                            <button
                                className="dashboard__play-btn"
                                onClick={() => setPlaying(!playing)}
                            >
                                {playing ? <HiPause /> : <HiPlay />}
                            </button>
                            <div className="dashboard__progress-track">
                                <motion.div
                                    className="dashboard__progress-fill"
                                    animate={{ width: playing ? '100%' : '35%' }}
                                    transition={{ duration: playing ? 90 : 0 }}
                                />
                            </div>
                            <span className="dashboard__time">32:15 / 90:00</span>
                        </div>
                    </div>
                </motion.div>

                {/* Sidebar */}
                <motion.div
                    className="dashboard__sidebar"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 }}
                >
                    <h3 className="dashboard__sidebar-title">Analysis Panels</h3>

                    {PANELS.map((panel) => (
                        <div
                            key={panel.key}
                            className="dashboard__toggle-row"
                            onClick={() => togglePanel(panel.key)}
                        >
                            <span>{panel.label}</span>
                            <div className={`toggle ${activePanels.has(panel.key) ? 'active' : ''}`} />
                        </div>
                    ))}
                </motion.div>
            </div>

            {/* Panels area */}
            <div className="dashboard__panels">
                <AnimatePresence>
                    {activePanels.has('running') && <RunningStats key="running" />}
                    {activePanels.has('heatmap') && <Heatmap key="heatmap" />}
                    {activePanels.has('velocity') && <VelocityChart key="velocity" />}
                </AnimatePresence>
            </div>

            {/* AI Insights — always visible */}
            <div className="dashboard__ai-section">
                <AIInsights />
            </div>
        </div>
    );
}
