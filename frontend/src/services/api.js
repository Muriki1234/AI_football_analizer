import axios from 'axios';

const api = axios.create({
    baseURL: '/api',
    headers: {
        'Content-Type': 'application/json',
    },
});

export const uploadVideo = async (file, onProgress) => {
    const formData = new FormData();
    formData.append('file', file);

    // Simulate slower upload for UX if file is small, 
    // but here we just pass the real progress event
    const response = await api.post('/upload', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
            if (onProgress) {
                const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                onProgress(percentCompleted);
            }
        },
    });
    return response.data;
};

export const trimVideo = async (videoId, start, end) => {
    const response = await api.post('/trim', {
        video_id: videoId,
        start,
        end
    });
    return response.data;
};

export const analyzeFrame = async (videoId, timeInSeconds) => {
    const response = await api.post('/analyze_frame', {
        video_id: videoId,
        time_in_seconds: timeInSeconds
    });
    return response.data;
};

export const getPlayers = async (videoId) => {
    const response = await api.get(`/players/${videoId}`);
    return response.data;
};

export const analyzePlayer = async (videoId, playerId, coordinates = null) => {
    const payload = {
        video_id: videoId,
        player_id: playerId
    };
    if (coordinates) {
        payload.coordinates = coordinates;
    }
    const response = await api.post('/analyze', payload);
    return response.data;
};

// ── Analysis Pipeline ─────────────────────────────────────────────────────

/**
 * Register an uploaded video with the analysis pipeline.
 * Must be called before startTracking.
 */
export const registerSession = async (sessionId, videoPath) => {
    const response = await api.post(`/${sessionId}/register`, { video_path: videoPath });
    return response.data;
};

/**
 * Start SAMURAI tracking for the selected player.
 * bbox: [x1, y1, x2, y2]
 */
export const startTracking = async (sessionId, bbox, frame = 0) => {
    const [x1, y1, x2, y2] = bbox;
    const response = await api.post(`/${sessionId}/track`, { x1, y1, x2, y2, frame });
    return response.data;
};

/**
 * Start global YOLO analysis (call after tracking_done).
 */
export const startGlobalAnalysis = async (sessionId) => {
    const response = await api.post(`/${sessionId}/analyze`);
    return response.data;
};

/**
 * Poll the session status until target status is reached or failure.
 * onProgress(data) called on every poll.
 */
export const pollSessionStatus = (sessionId, targetStatus, onProgress, interval = 1500) => {
    return new Promise((resolve, reject) => {
        const poll = async () => {
            try {
                const { data } = await api.get(`/${sessionId}/status`);
                if (onProgress) onProgress(data);
                if (data.status === targetStatus) return resolve(data);
                if (data.status && data.status.includes('failed')) return reject(new Error(data.error || 'Pipeline failed'));
                setTimeout(poll, interval);
            } catch (e) {
                reject(e);
            }
        };
        poll();
    });
};

/**
 * Trigger on-demand feature generation.
 * feature: 'heatmap' | 'speed_chart' | 'possession' | 'minimap_replay'
 */
export const generateFeature = async (sessionId, feature) => {
    const response = await api.post(`/${sessionId}/generate/${feature}`);
    return response.data; // { task_id, status: 'queued' }
};

/**
 * Poll a task until done or failed.
 * onProgress(task) called on every poll.
 */
export const pollTaskStatus = (sessionId, taskId, onProgress, interval = 1000) => {
    return new Promise((resolve, reject) => {
        const poll = async () => {
            try {
                const { data } = await api.get(`/${sessionId}/task/${taskId}`);
                if (onProgress) onProgress(data);
                if (data.status === 'done') return resolve(data);
                if (data.status === 'failed') return reject(new Error(data.error || 'Task failed'));
                setTimeout(poll, interval);
            } catch (e) {
                reject(e);
            }
        };
        poll();
    });
};

/**
 * Get player summary stats after analysis_done.
 */
export const getSummary = async (sessionId) => {
    const response = await api.get(`/${sessionId}/summary`);
    return response.data;
};

export default api;
