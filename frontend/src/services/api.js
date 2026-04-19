import axios from 'axios';
import colab from './colabService';

const api = axios.create({
    baseURL: '/api',
    headers: {
        'Content-Type': 'application/json',
    },
});

export const uploadVideo = async (file, onProgress) => {
    // ── Colab 模式：视频只传到 GPU 服务器，本地零磁盘占用 ──────────────────
    if (colab.isConfigured()) {
        // 用 crypto.randomUUID 生成 12 位 hex session ID（无需本地 Flask）
        const video_id = crypto.randomUUID().replace(/-/g, '').substring(0, 12);
        console.log('📤 Colab 模式：直接上传到 GPU，跳过本地存储...');
        await colab.sendVideo(video_id, file, (pct) => {
            if (onProgress) onProgress(pct);
        });
        console.log('✅ 视频已到达 Colab，video_id:', video_id);
        return { video_id };
    }

    // ── 本地模式：上传到本地 Flask ──────────────────────────────────────────
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
            if (onProgress) {
                const pct = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                onProgress(pct);
            }
        },
    });

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
    // Colab 配置时走 GPU（Soccana YOLO 模型），结果保存在 Colab；否则走本地（Roboflow / Mock）
    if (colab.isConfigured()) return colab.analyzeFrame(videoId, timeInSeconds);
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


// ── Analysis Pipeline（全部走 Colab）────────────────────────────────────

export const registerSession = async (sessionId, videoPath) => {
    // Colab 模式下 sendVideo 已经创建 session，register 是 no-op；
    // 直接打本地 Flask 会 400（本地无视频文件）
    if (colab.isConfigured()) return colab.registerSession(sessionId, videoPath);
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
            heatmap:          () => colab.generateHeatmap(sessionId),
            speed_chart:      () => colab.generateSpeedChart(sessionId),
            possession:       () => colab.generatePossession(sessionId),
            minimap_replay:   () => colab.generateMinimapReplay(sessionId),
            full_replay:      () => colab.generateFullReplay(sessionId),
            sprint_analysis:  () => colab.generateSprintAnalysis(sessionId),
            defensive_line:   () => colab.generateDefensiveLine(sessionId),
            ai_summary:       () => colab.generateAiSummary(sessionId),
        };
        const fn = methods[feature];
        if (!fn) throw new Error(`Unknown feature: ${feature}`);
        const taskId = await fn();
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