import { useState, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { trimVideo, analyzeFrame } from '../services/api';
import { motion } from 'framer-motion';
import { HiScissors, HiArrowRight, HiArrowLeft } from 'react-icons/hi2';
import RangeSlider from '../components/RangeSlider';
import StepNav from '../components/StepNav';
import './Trimmer.css';

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

export default function Trimmer() {
    const [range, setRange] = useState([0, 0]);
    const [duration, setDuration] = useState(0);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const videoRef = useRef(null);
    const navigate = useNavigate();
    const location = useLocation();
    const videoId = location.state?.videoId;

    const clipDuration = range[1] - range[0];

    const handleConfirm = async () => {
        if (videoId) {
            try {
                setIsAnalyzing(true);
                // 1. Save trim settings
                await trimVideo(videoId, range[0], range[1]);

                // 2. Extract the first frame of the trimmed selection (offset 0 in the trimmed file)
                const analysisResult = await analyzeFrame(videoId, 0);

                // 3. Navigate to config with the pre-loaded data
                navigate('/configure', {
                    state: {
                        videoId,
                        detectedPlayers: analysisResult.players_data,
                        frameUrl: analysisResult.annotated_frame_url,
                        imgDims: analysisResult.image_dimensions,
                        videoPath: location.state?.videoPath
                    }
                });
            } catch (e) {
                console.error(e);
                alert('Failed to process video selection');
                setIsAnalyzing(false);
            }
        } else {
            // Fallback for dev/demo if no video uploaded
            navigate('/configure');
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
                <button className="btn btn-ghost" onClick={() => navigate('/upload')} disabled={isAnalyzing}>
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
                {/* Video Player */}
                <div className="trimmer__video-area">
                    {videoId ? (
                        <video
                            ref={videoRef}
                            src={`/api/videos/${videoId}`}
                            className="trimmer__video-player"
                            controls
                            onLoadedMetadata={(e) => {
                                const d = e.target.duration;
                                setDuration(d);
                                setRange([0, d]);
                            }}
                        />
                    ) : (
                        <div className="trimmer__video-placeholder">
                            <div className="trimmer__timestamp">{formatTime(range[0])}</div>
                            <div className="trimmer__video-label">Video Preview (No Video Loaded)</div>
                            <div className="trimmer__timestamp">{formatTime(range[1])}</div>
                        </div>
                    )}

                    {/* Visual timeline bar */}
                    <div className="trimmer__timeline">
                        {Array.from({ length: 20 }).map((_, i) => (
                            <div key={i} className="trimmer__timeline-bar" style={{
                                height: `${20 + Math.random() * 30}px`,
                                opacity: duration > 0 && (i / 20 >= range[0] / duration && i / 20 <= range[1] / duration) ? 1 : 0.25
                            }} />
                        ))}
                    </div>
                </div>

                {/* Range slider */}
                <div className="trimmer__slider-area">
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

                {/* Info + CTA */}
                <div className="trimmer__footer">
                    <div className="trimmer__clip-info">
                        <span className="trimmer__clip-badge">
                            Clip Duration: <strong>{formatTime(clipDuration)}</strong>
                        </span>
                    </div>
                    <button
                        className={`btn btn-primary ${isAnalyzing ? 'btn--loading' : ''}`}
                        onClick={handleConfirm}
                        disabled={isAnalyzing}
                    >
                        {isAnalyzing ? 'Extracting Frame...' : 'Confirm Selection'}
                        {!isAnalyzing && <HiArrowRight />}
                    </button>
                </div>
            </motion.div>
        </div>
    );
}
