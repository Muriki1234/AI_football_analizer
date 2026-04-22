import { useState, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import { HiScissors, HiArrowRight, HiArrowLeft, HiForward } from 'react-icons/hi2';
import { trimVideo, artifactUrl } from '../services/api';
import RangeSlider from '../components/RangeSlider';
import StepNav from '../components/StepNav';
import './Trimmer.css';

function formatTime(seconds) {
    if (!isFinite(seconds) || seconds < 0) return '00:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

export default function Trimmer() {
    const [range, setRange] = useState([0, 0]);
    const [duration, setDuration] = useState(0);
    const [isTrimming, setIsTrimming] = useState(false);
    const [isVideoLoading, setIsVideoLoading] = useState(true);
    const videoRef = useRef(null);
    const navigate = useNavigate();
    const location = useLocation();
    const videoId = location.state?.videoId || location.state?.sessionId;

    const clipDuration = Math.max(0, range[1] - range[0]);

    const goToDashboard = () => {
        navigate('/dashboard', { state: { sessionId: videoId, videoId } });
    };

    const handleConfirm = async () => {
        if (!videoId) {
            navigate('/dashboard');
            return;
        }
        // If user didn't move the handles, skip the trim and jump straight to analysis.
        if (duration > 0 && range[0] <= 0.1 && Math.abs(range[1] - duration) < 0.1) {
            goToDashboard();
            return;
        }
        setIsTrimming(true);
        const toastId = toast.loading('Trimming clip…');
        try {
            await trimVideo(videoId, range[0], range[1]);
            toast.success('Clip ready', { id: toastId });
            goToDashboard();
        } catch (e) {
            console.error(e);
            toast.error(
                e?.response?.data?.detail || e?.message || 'Trim failed',
                { id: toastId }
            );
            setIsTrimming(false);
        }
    };

    return (
        <div className="page-container trimmer-page">
            <div className="bg-grid" />

            <motion.div
                className="trimmer-page__back"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
            >
                <button className="btn btn-ghost" onClick={() => navigate('/upload')} disabled={isTrimming}>
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
                <p>Select a time range, or skip to analyze the whole video.</p>
            </motion.div>

            <motion.div
                className="trimmer__viewport"
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2 }}
            >
                <div className="trimmer__video-area">
                    {videoId ? (
                        <video
                            ref={videoRef}
                            src={artifactUrl(videoId, 'video.mp4')}
                            className="trimmer__video-player"
                            controls
                            onLoadedMetadata={(e) => {
                                const d = e.target.duration;
                                setDuration(d);
                                setRange([0, d]);
                                setIsVideoLoading(false);
                            }}
                        />
                    ) : (
                        <div className="trimmer__video-placeholder">
                            <div className="trimmer__timestamp">{formatTime(range[0])}</div>
                            <div className="trimmer__video-label">Video Preview (No Video Loaded)</div>
                            <div className="trimmer__timestamp">{formatTime(range[1])}</div>
                        </div>
                    )}

                    <div className="trimmer__timeline">
                        {Array.from({ length: 20 }).map((_, i) => (
                            <div key={i} className="trimmer__timeline-bar" style={{
                                height: `${20 + Math.random() * 30}px`,
                                opacity: duration > 0 && (i / 20 >= range[0] / duration && i / 20 <= range[1] / duration) ? 1 : 0.25,
                            }} />
                        ))}
                    </div>
                </div>

                <div className="trimmer__slider-area">
                    {isVideoLoading && videoId && (
                        <div className="trimmer__loading">
                            <div className="trimmer__loading-spinner" />
                            <span>Loading video…</span>
                        </div>
                    )}
                    {duration > 0 && (
                        <RangeSlider
                            key={duration}
                            min={0}
                            max={duration}
                            onChange={(newRange) => {
                                if (newRange[0] !== range[0] && videoRef.current) {
                                    videoRef.current.currentTime = newRange[0];
                                } else if (newRange[1] !== range[1] && videoRef.current) {
                                    videoRef.current.currentTime = newRange[1];
                                }
                                setRange(newRange);
                            }}
                            formatLabel={formatTime}
                        />
                    )}
                </div>

                <div className="trimmer__footer">
                    <div className="trimmer__clip-info">
                        <span className="trimmer__clip-badge">
                            Clip Duration: <strong>{formatTime(clipDuration)}</strong>
                        </span>
                    </div>
                    <div style={{ display: 'flex', gap: 8 }}>
                        <button
                            className="btn btn-ghost"
                            onClick={goToDashboard}
                            disabled={isTrimming}
                            title="Skip trimming and analyze the whole video"
                        >
                            <HiForward /> Use whole video
                        </button>
                        <button
                            className={`btn btn-primary ${isTrimming ? 'btn--loading' : ''}`}
                            onClick={handleConfirm}
                            disabled={isTrimming || duration === 0}
                        >
                            {isTrimming ? 'Trimming…' : 'Confirm & Analyze'}
                            {!isTrimming && <HiArrowRight />}
                        </button>
                    </div>
                </div>
            </motion.div>
        </div>
    );
}
