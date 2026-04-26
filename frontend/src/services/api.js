/**
 * API client for the v2 FastAPI server. No more Colab branching — the server
 * is always the same URL (configured via VITE_API_BASE_URL).
 *
 * Highlights:
 *   - uploadVideo: chunked upload (5 MB) with retries for flaky tunnels.
 *   - subscribeSession: SSE stream so the dashboard doesn't poll.
 *   - All requests carry X-API-Key when configured (see services/config.js).
 */

import axios from 'axios';
import { API_BASE_URL, API_KEY, absUrl, authHeaders } from './config';

const CHUNK_SIZE = 5 * 1024 * 1024; // 5 MB
const MAX_CHUNK_RETRIES = 3;

export const api = axios.create({
    baseURL: API_BASE_URL || '',
    headers: { 'Content-Type': 'application/json' },
});

api.interceptors.request.use((cfg) => {
    if (API_KEY) cfg.headers['X-API-Key'] = API_KEY;
    return cfg;
});

const unwrap = (res) => res.data;

// ── Sessions (upload + trim) ─────────────────────────────────────────────────

const createSessionShell = async () => {
    const res = await api.post('/api/sessions');
    return res.data.session_id;
};

const putChunk = async (sessionId, index, chunk) => {
    let attempt = 0;
    while (true) {
        try {
            await axios.put(
                absUrl(`/api/sessions/${sessionId}/chunks/${index}`),
                chunk,
                {
                    headers: {
                        'Content-Type': 'application/octet-stream',
                        ...authHeaders(),
                    },
                    timeout: 120_000,
                }
            );
            return;
        } catch (err) {
            attempt += 1;
            if (attempt >= MAX_CHUNK_RETRIES) throw err;
            await new Promise((r) => setTimeout(r, 500 * attempt));
        }
    }
};

/**
 * Upload a file in 5MB chunks. Returns { session_id, video_path }.
 * onProgress is called with integer percent 0..100.
 */
export const uploadVideo = async (file, onProgress) => {
    const sessionId = await createSessionShell();

    const totalChunks = Math.max(1, Math.ceil(file.size / CHUNK_SIZE));
    for (let i = 0; i < totalChunks; i++) {
        const start = i * CHUNK_SIZE;
        const end = Math.min(start + CHUNK_SIZE, file.size);
        // eslint-disable-next-line no-await-in-loop
        await putChunk(sessionId, i, file.slice(start, end));
        if (onProgress) {
            onProgress(Math.round(((i + 1) / totalChunks) * 95));
        }
    }

    const complete = await api.post(`/api/sessions/${sessionId}/complete`, {
        total_chunks: totalChunks,
        filename: file.name,
    });
    if (onProgress) onProgress(100);

    return {
        video_id: sessionId,
        session_id: sessionId,
        ...complete.data,
    };
};

export const trimVideo = async (sessionId, start, end) => {
    const res = await api.post(`/api/sessions/${sessionId}/trim`, { start, end });
    return res.data;
};

export const getSession = (sessionId) =>
    api.get(`/api/sessions/${sessionId}`).then(unwrap);

// ── Analysis ─────────────────────────────────────────────────────────────────

export const startAnalysis = (sessionId) =>
    api.post(`/api/sessions/${sessionId}/analyze`).then(unwrap);

export const startTracking = (sessionId, bbox, frame = 0) =>
    api
        .post(`/api/sessions/${sessionId}/track`, {
            bbox: [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
            frame,
        })
        .then(unwrap);

export const queueFeature = (sessionId, feature) =>
    api.post(`/api/sessions/${sessionId}/features/${feature}`).then(unwrap);

export const listTasks = (sessionId) =>
    api.get(`/api/sessions/${sessionId}/tasks`).then(unwrap);

export const getTask = (sessionId, taskId) =>
    api.get(`/api/sessions/${sessionId}/tasks/${taskId}`).then(unwrap);

export const getSummary = (sessionId) =>
    api.get(`/api/sessions/${sessionId}/summary`).then(unwrap);

/**
 * Build a URL to a session artifact (hits /files/{path}). When an API key is
 * configured, append it as ?key= so plain <img>/<video> tags can authenticate
 * (these tags can't send custom headers).
 */
export const artifactUrl = (sessionId, relPath) => {
    const path = encodeURIComponent(relPath).replace(/%2F/g, '/');
    const base = absUrl(`/api/sessions/${sessionId}/files/${path}`);
    return API_KEY ? `${base}?key=${encodeURIComponent(API_KEY)}` : base;
};

// ── Legacy shims (keep existing pages compiling) ─────────────────────────────
// The old UI calls these; keep them around so we don't break imports while
// the individual pages get rewritten. They quietly map to the v2 API where
// possible, and return a clearly-flagged "not implemented" marker otherwise.

export const registerSession = async () => ({ ok: true });

export const analyzeFrame = async (sessionId, frame = 0) => {
    const res = await api.post(`/api/sessions/${sessionId}/detect-frame`, { frame });
    const data = res.data || {};
    // The server returned a relative URL like /api/sessions/{id}/files/first_frame.jpg.
    // Convert to artifactUrl() which handles ?key= for img-tag auth.
    const rel = data.annotated_frame_path || 'first_frame.jpg';
    return {
        players_data: data.players || [],
        annotated_frame_url: artifactUrl(sessionId, rel),
        image_dimensions: data.image_dimensions || null,
    };
};

export const startGlobalAnalysis = startAnalysis;

export const generateFeature = queueFeature;

// ── SSE (Server-Sent Events) subscription ────────────────────────────────────

/**
 * Subscribe to session events. Returns an unsubscribe() function.
 * Pass handlers for each event kind:
 *   onSession({status, progress, stage, ...})
 *   onTask({task_id, task_type, status, progress, url, result})
 *   onError(err)  — fired on network / protocol errors
 */
export const subscribeSession = (sessionId, handlers = {}) => {
    const baseUrl = absUrl(`/api/sessions/${sessionId}/events`);
    // EventSource can't send custom headers; pass the API key via ?key= query.
    const url = API_KEY ? `${baseUrl}?key=${encodeURIComponent(API_KEY)}` : baseUrl;
    const source = new EventSource(url, { withCredentials: false });

    const parse = (raw) => {
        try {
            return typeof raw === 'string' ? JSON.parse(raw) : raw;
        } catch {
            return null;
        }
    };

    const unwrapEventPayload = (payload) => {
        if (!payload || typeof payload !== 'object') return payload;
        if ('kind' in payload && 'data' in payload) return payload.data;
        return payload;
    };

    source.addEventListener('session', (e) => {
        const data = unwrapEventPayload(parse(e.data));
        if (data && handlers.onSession) handlers.onSession(data);
    });
    source.addEventListener('task', (e) => {
        const data = unwrapEventPayload(parse(e.data));
        if (data && handlers.onTask) handlers.onTask(data);
    });
    source.addEventListener('heartbeat', () => {
        if (handlers.onHeartbeat) handlers.onHeartbeat();
    });
    source.onerror = (err) => {
        if (handlers.onError) handlers.onError(err);
    };

    return () => source.close();
};

// ── Backwards-compat polling fallback (SSE is strongly preferred) ────────────

export const pollSessionStatus = (sessionId, targetStatus, onProgress, interval = 1500) =>
    new Promise((resolve, reject) => {
        const tick = async () => {
            try {
                const data = await getSession(sessionId);
                if (onProgress) onProgress(data);
                if (data.status === targetStatus) return resolve(data);
                if (data.status && data.status.endsWith('_failed'))
                    return reject(new Error(data.error || 'Pipeline failed'));
                setTimeout(tick, interval);
            } catch (e) {
                reject(e);
            }
        };
        tick();
    });

export const pollTaskStatus = (sessionId, taskId, onProgress, interval = 1000) =>
    new Promise((resolve, reject) => {
        const tick = async () => {
            try {
                const data = await getTask(sessionId, taskId);
                if (onProgress) onProgress(data);
                if (data.status === 'done') return resolve(data);
                if (data.status === 'failed')
                    return reject(new Error(data.error || 'Task failed'));
                setTimeout(tick, interval);
            } catch (e) {
                reject(e);
            }
        };
        tick();
    });

export default api;
