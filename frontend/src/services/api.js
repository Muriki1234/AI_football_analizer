import axios from 'axios';
import colab from './colabService';

const api = axios.create({
    baseURL: '/api',
    headers: {
        'Content-Type': 'application/json',
    },
});

export const uploadVideo = async (file, onProgress) => {
    const formData = new FormData();
    formData.append('file', file);

    // 1. 上传到本地 Flask，拿到 video_id
    const response = await api.post('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
            if (onProgress) {
                const pct = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                // 本地上传占进度条前50%
                onProgress(Math.round(pct * 0.5));
            }
        },
    });

    const { video_id } = response.data;

    // 2. 如果配置了 Colab，把视频同步传给 Colab
    if (colab.isConfigured()) {
        try {
            console.log('📤 正在把视频传给 Colab GPU...');
            await colab.sendVideo(video_id, file, (pct) => {
                // Colab上传占进度条后50%
                if (onProgress) onProgress(50 + Math.round(pct * 0.5));
            });
            console.log('✅ 视频已到达 Colab');
        } catch (e) {
            console.error('⚠️ 视频传给 Colab 失败:', e.message);
            // 不阻断本地流程
        }
    }

    return response.data;
};

export const trimVideo = async (videoId, start, end) => {
    // Colab 上裁剪（视频已在Colab）
    if (colab.isConfigured()) {
        try {
            await colab.trimVideo(videoId, start, end);
        } catch (e) {
            console.error('Colab trim failed:', e.message);
        }
    }

    // 本地也记录裁剪区间（维持本地状态一致）
    const response = await api.post('/trim', {
        video_id: videoId,
        start,
        end
    });
    return response.data;
};

export const analyzeFrame = async (videoId, timeInSeconds) => {
    // 始终使用本地后端进行单帧分析 (Roboflow 或 Mock)，保证选框精准度
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
    const payload = { video_id: videoId, player_id: playerId };
    if (coordinates) payload.coordinates = coordinates;
    const response = await api.post('/analyze', payload);
    return response.data;
};

// ── Auto-Start Pipeline（跳过 Trim & 球员选择）────────────────────────

export const autoStart = async (sessionId) => {
    if (colab.isConfigured()) return colab.autoStart(sessionId);
    const response = await api.post(`/${sessionId}/auto_start`);
    return response.data;
};

// ── Analysis Pipeline（全部走 Colab）────────────────────────────────────

export const registerSession = async (sessionId, videoPath) => {
    // 强制在本地注册 Session，把 videoId 转化为 sessionId
    // Colab 那边不需要 register_session，因为 sendVideo 已经包含了所需上下文
    const response = await api.post(`/${sessionId}/register`, { video_path: videoPath });
    return response.data;
};

export const startTracking = async (sessionId, bbox, frame = 0) => {
    const [x1, y1, x2, y2] = bbox;
    if (colab.isConfigured()) return colab.startTracking(sessionId, { x1, y1, x2, y2, frame });
    const response = await api.post(`/${sessionId}/track`, { x1, y1, x2, y2, frame });
    return response.data;
};

export const startGlobalAnalysis = async (sessionId) => {
    if (colab.isConfigured()) return colab.startAnalysis(sessionId);
    const response = await api.post(`/${sessionId}/analyze`);
    return response.data;
};

export const pollSessionStatus = (sessionId, targetStatus, onProgress, interval = 1500) => {
    if (colab.isConfigured()) {
        return colab.pollStatus(sessionId, [targetStatus], onProgress);
    }
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

export const generateFeature = async (sessionId, feature) => {
    if (colab.isConfigured()) {
                const methods = {
            heatmap: () => colab.generateHeatmap(sessionId),
            speed_chart: () => colab.generateSpeedChart(sessionId),
            possession: () => colab.generatePossession(sessionId),
            minimap_replay: () => colab.generateMinimapReplay(sessionId),
            full_replay: () => colab.generateFullReplay(sessionId),
        };
        const taskId = await methods[feature]();
        return { task_id: taskId, status: 'queued' };
    }
    const response = await api.post(`/${sessionId}/generate/${feature}`);
    return response.data;
};

export const pollTaskStatus = (sessionId, taskId, onProgress, interval = 1000) => {
    if (colab.isConfigured()) {
        return colab.pollTask(sessionId, taskId, (progress) => {
            if (onProgress) onProgress({ status: 'running', progress });
        }).then(t => {
            if (onProgress) onProgress(t);
            return t;
        });
    }
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

export const getSummary = async (sessionId) => {
    if (colab.isConfigured()) return colab.getSummary(sessionId);
    const response = await api.get(`/${sessionId}/summary`);
    return response.data;
};

export default api;