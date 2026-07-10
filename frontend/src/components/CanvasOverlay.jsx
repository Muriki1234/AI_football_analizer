import React, { useEffect, useRef, useState } from 'react';

const CanvasOverlay = ({ dataUrl, videoRef, visible }) => {
    const canvasRef = useRef(null);
    const [overlayData, setOverlayData] = useState(null);

    // Fetch the overlay tracking JSON
    useEffect(() => {
        if (!dataUrl) return;
        fetch(dataUrl)
            .then(r => r.json())
            .then(data => setOverlayData(data))
            .catch(err => console.error("Failed to load overlay data", err));
    }, [dataUrl]);

    useEffect(() => {
        if (!visible || !overlayData || !videoRef.current || !canvasRef.current) {
            if (canvasRef.current) {
                const ctx = canvasRef.current.getContext('2d');
                ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
            }
            return;
        }

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        const fps = overlayData.fps || 25.0;

        let animationFrameId;

        const renderLoop = () => {
            if (video.paused || video.ended) {
                // We still want to render even if paused, to show boxes when seeking
                drawFrame();
            } else {
                drawFrame();
            }
            animationFrameId = requestAnimationFrame(renderLoop);
        };

        const drawFrame = () => {
            const { videoWidth, videoHeight, currentTime } = video;
            if (!videoWidth || !videoHeight) return;

            let trackW = videoWidth;
            let trackH = videoHeight;
            if (overlayData.resolution) {
                trackW = overlayData.resolution[0];
                trackH = overlayData.resolution[1];
            } else if (videoHeight > 720) {
                // Heuristic for old sessions where backend downscaled to 720p max
                const scale = 720 / videoHeight;
                trackW = Math.round(videoWidth * scale);
                trackH = 720;
            }

            // Match canvas internal resolution to tracking resolution (or video natural resolution)
            if (canvas.width !== trackW || canvas.height !== trackH) {
                canvas.width = trackW;
                canvas.height = trackH;
            }

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Calculate current frame index based on time and fps
            const currentFrameIdx = Math.floor(currentTime * fps);
            
            if (currentFrameIdx >= 0 && currentFrameIdx < overlayData.frames.length) {
                const frameData = overlayData.frames[currentFrameIdx];
                if (!frameData) return;
                
                const [targetBbox, players, ballBbox] = frameData;

                // Draw standard players
                if (players) {
                    for (const p of players) {
                        const [id, x1, y1, x2, y2, team, has_ball] = p;
                        const w = x2 - x1;
                        const h = y2 - y1;
                        
                        // Set colors based on team
                        let strokeStyle = "rgba(0, 255, 0, 0.7)"; // fallback green
                        if (team === 1) strokeStyle = "rgba(52, 152, 219, 0.8)"; // blue
                        else if (team === 2) strokeStyle = "rgba(231, 76, 60, 0.8)"; // red
                        
                        // Draw ellipse at bottom of player
                        ctx.save();
                        ctx.beginPath();
                        ctx.ellipse(x1 + w/2, y2, w/2, w/4, 0, 0, 2 * Math.PI);
                        ctx.fillStyle = strokeStyle;
                        ctx.fill();
                        ctx.restore();

                        // Draw ID text
                        ctx.font = "bold 14px Arial";
                        ctx.fillStyle = "white";
                        ctx.textAlign = "center";
                        ctx.shadowColor = "black";
                        ctx.shadowBlur = 4;
                        ctx.fillText(id, x1 + w/2, y1 - 5);
                        ctx.shadowBlur = 0; // reset
                        
                        // Draw possession marker
                        if (has_ball) {
                            ctx.beginPath();
                            ctx.moveTo(x1 + w/2, y1 - 25);
                            ctx.lineTo(x1 + w/2 - 10, y1 - 40);
                            ctx.lineTo(x1 + w/2 + 10, y1 - 40);
                            ctx.closePath();
                            ctx.fillStyle = "rgba(0, 80, 255, 0.9)"; // vivid blue (was BGR 255,80,0 -> RGB 0,80,255)
                            ctx.fill();
                        }
                    }
                }

                // Draw ball
                if (ballBbox) {
                    const [bx1, by1, bx2, by2] = ballBbox;
                    const cx = (bx1 + bx2) / 2;
                    const cy = by1; // top of ball
                    ctx.beginPath();
                    ctx.moveTo(cx, cy - 5);
                    ctx.lineTo(cx - 5, cy - 15);
                    ctx.lineTo(cx + 5, cy - 15);
                    ctx.closePath();
                    ctx.fillStyle = "rgba(0, 255, 80, 0.9)"; // GREEN
                    ctx.fill();
                }

                // Draw target (SAMURAI)
                if (targetBbox) {
                    const [sx, sy, sw, sh] = targetBbox;
                    ctx.strokeStyle = "rgba(255, 255, 0, 0.9)"; // Yellow box
                    ctx.lineWidth = 3;
                    ctx.strokeRect(sx, sy, sw, sh);
                    
                    // Draw name tag
                    ctx.fillStyle = "rgba(255, 255, 0, 0.9)";
                    ctx.fillRect(sx, sy - 20, Math.max(100, sw), 20);
                    ctx.fillStyle = "black";
                    ctx.font = "bold 12px Arial";
                    ctx.textAlign = "center";
                    ctx.fillText("Tracked Target", sx + Math.max(100, sw)/2, sy - 5);
                }
            }
        };

        // Also bind to seek/play/timeupdate events to ensure it draws when paused
        video.addEventListener('seeked', drawFrame);
        video.addEventListener('timeupdate', drawFrame);
        
        renderLoop();

        return () => {
            cancelAnimationFrame(animationFrameId);
            video.removeEventListener('seeked', drawFrame);
            video.removeEventListener('timeupdate', drawFrame);
        };
    }, [visible, overlayData, videoRef]);

    return (
        <canvas
            ref={canvasRef}
            style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                objectFit: 'contain',
                pointerEvents: 'none',
                zIndex: 5,
                opacity: visible ? 1 : 0,
                transition: 'opacity 0.3s'
            }}
        />
    );
};

export default CanvasOverlay;
