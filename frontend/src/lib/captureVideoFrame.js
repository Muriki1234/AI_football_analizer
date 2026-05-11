/**
 * Capture frame N from a video URL using a hidden <video> element + canvas.
 * Runs entirely in the browser — no server GPU needed.
 *
 * Returns:
 *   - dataUrl: image/jpeg data URL (for <img> src)
 *   - base64: just the base64 payload (no "data:image/jpeg;base64," prefix)
 *   - width, height: pixel dimensions of the captured frame
 *
 * Requires the video URL to be CORS-friendly. Supabase Storage public URLs
 * are by default.
 */
export const captureVideoFrame = (videoUrl, frameIdx = 0, fps = 25, quality = 0.85) => {
    return new Promise((resolve, reject) => {
        const video = document.createElement('video');
        video.crossOrigin = 'anonymous';
        video.muted = true;
        video.playsInline = true;
        video.preload = 'auto';
        video.src = videoUrl;

        const cleanup = () => {
            video.removeAttribute('src');
            video.load();
        };

        const onError = (msg) => () => {
            cleanup();
            reject(new Error(msg));
        };

        video.addEventListener('error', onError('Video failed to load'), { once: true });

        video.addEventListener('loadedmetadata', () => {
            if (!video.videoWidth || !video.videoHeight) {
                return onError('Video has no dimensions')();
            }
            const targetSec = Math.max(0, frameIdx / Math.max(1, fps));
            // For frame 0, some browsers (Safari) need a tiny offset before seeked fires
            video.currentTime = frameIdx === 0 ? 0.001 : targetSec;
        }, { once: true });

        video.addEventListener('seeked', () => {
            try {
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const dataUrl = canvas.toDataURL('image/jpeg', quality);
                cleanup();
                resolve({
                    dataUrl,
                    base64: dataUrl.split(',')[1],
                    width: canvas.width,
                    height: canvas.height,
                });
            } catch (e) {
                cleanup();
                reject(e);
            }
        }, { once: true });
    });
};
