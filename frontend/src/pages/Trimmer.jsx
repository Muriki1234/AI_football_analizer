import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { HiScissors, HiArrowRight, HiArrowLeft } from 'react-icons/hi2';
import RangeSlider from '../components/RangeSlider';
import StepNav from '../components/StepNav';
import './Trimmer.css';

const MOCK_DURATION = 5400; // 90 minutes in seconds

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

export default function Trimmer() {
    const [range, setRange] = useState([0, MOCK_DURATION]);
    const navigate = useNavigate();

    const clipDuration = range[1] - range[0];

    return (
        <div className="page-container trimmer-page">
            <div className="bg-grid" />

            <motion.div
                className="trimmer-page__back"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
            >
                <button className="btn btn-ghost" onClick={() => navigate('/upload')}>
                    <HiArrowLeft /> Back
                </button>
            </motion.div>

            <StepNav />

            <motion.div
                className="trimmer-page__header"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
            >
                <HiScissors className="trimmer-page__icon" />
                <h1>Trim Video Clip</h1>
                <p>Select the time range you want to analyze</p>
            </motion.div>

            <motion.div
                className="trimmer__viewport"
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2 }}
            >
                {/* Simulated video viewport */}
                <div className="trimmer__video-area">
                    <div className="trimmer__video-placeholder">
                        <div className="trimmer__timestamp">{formatTime(range[0])}</div>
                        <div className="trimmer__video-label">Video Preview</div>
                        <div className="trimmer__timestamp">{formatTime(range[1])}</div>
                    </div>

                    {/* Visual timeline bar */}
                    <div className="trimmer__timeline">
                        {Array.from({ length: 20 }).map((_, i) => (
                            <div key={i} className="trimmer__timeline-bar" style={{
                                height: `${20 + Math.random() * 30}px`,
                                opacity: (i / 20 >= range[0] / MOCK_DURATION && i / 20 <= range[1] / MOCK_DURATION) ? 1 : 0.25
                            }} />
                        ))}
                    </div>
                </div>

                {/* Range slider */}
                <div className="trimmer__slider-area">
                    <RangeSlider
                        min={0}
                        max={MOCK_DURATION}
                        onChange={setRange}
                        formatLabel={formatTime}
                    />
                </div>

                {/* Info + CTA */}
                <div className="trimmer__footer">
                    <div className="trimmer__clip-info">
                        <span className="trimmer__clip-badge">
                            Clip Duration: <strong>{formatTime(clipDuration)}</strong>
                        </span>
                    </div>
                    <button
                        className="btn btn-primary"
                        onClick={() => navigate('/configure')}
                    >
                        Confirm Selection
                        <HiArrowRight />
                    </button>
                </div>
            </motion.div>
        </div>
    );
}
