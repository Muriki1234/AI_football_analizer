import { supabase } from '../lib/supabase';
import { addRecentSession } from '../lib/recentSessions';
import { captureVideoFrame } from '../lib/captureVideoFrame';

// Poll /api/status with exponential backoff: tight while warming up,
// loose afterwards. Cuts Vercel function invocations ~60% versus a flat
// 2.5s interval without sacrificing perceived responsiveness.
//
// Cadence: 2s × 5  →  5s × 6  →  10s for the rest. Total budget ≈ 2 min.
const pollJobResult = async (jobId, maxWaitMs = 120000) => {
    const deadline = Date.now() + maxWaitMs;
    const intervalFor = (n) => (n < 5 ? 2000 : n < 11 ? 5000 : 10000);
    let n = 0;
    while (Date.now() < deadline) {
        await new Promise(r => setTimeout(r, intervalFor(n++)));
        const res = await fetch(`/api/status?id=${jobId}`);
        const data = await res.json();
        if (data.status === 'COMPLETED') return data.output;
        if (data.status === 'FAILED') throw new Error(data.error || 'RunPod job failed');
        // IN_QUEUE / IN_PROGRESS → keep polling
    }
    throw new Error('Detection timed out after 2 minutes');
};

// Helper to get public URL for a file in Supabase Storage
const getPublicUrl = (fileName) => {
    const { data } = supabase.storage.from('videos').getPublicUrl(fileName);
    return data.publicUrl;
};

// ── Sessions (Supabase Auth & DB) ──────────────────────────────────────────

/**
 * Upload a file directly to Supabase Storage.
 * Then creates a record in the 'sessions' table.
 */
export const uploadVideo = async (file, onProgress) => {
    // Try to get current user; if not logged in, auto sign-in anonymously
    let { data: { user } } = await supabase.auth.getUser();
    if (!user) {
        const { data, error } = await supabase.auth.signInAnonymously();
        if (error) throw new Error('Failed to create guest session: ' + error.message);
        user = data.user;
    }

    const sessionId = crypto.randomUUID();
    const fileName = `${sessionId}/${file.name}`;

    // 1. Upload to Supabase Storage
    const { error: uploadError } = await supabase.storage
        .from('videos')
        .upload(fileName, file, {
            cacheControl: '3600',
            upsert: false
        });

    if (uploadError) throw uploadError;
    if (onProgress) onProgress(100);

    const videoUrl = getPublicUrl(fileName);

    // 2. Create session record in Database
    const { error: dbError } = await supabase
        .from('sessions')
        .insert([{
            id: sessionId,
            user_id: user.id,
            video_url: videoUrl,
            status: 'uploaded'
        }]);

    if (dbError) throw dbError;

    addRecentSession({
        id: sessionId,
        fileName: file.name,
        videoUrl,
        size: file.size,
    });

    return { session_id: sessionId, video_url: videoUrl };
};

// Flatten the `extra` JSONB column into the top-level session object so
// downstream code can do `session.tracks_cache_path`, `session.minimap_data_url`,
// etc. without poking into `.extra`. Mirrors the backend's get_session().
const _flattenSession = (row) => {
    if (!row) return row;
    let extra = row.extra;
    if (typeof extra === 'string') {
        try { extra = JSON.parse(extra); } catch { extra = null; }
    }
    if (extra && typeof extra === 'object') {
        for (const [k, v] of Object.entries(extra)) {
            if (!(k in row)) row[k] = v;
        }
    }
    return row;
};

export const getSession = async (sessionId) => {
    const { data, error } = await supabase
        .from('sessions')
        .select('*')
        .eq('id', sessionId)
        .single();

    if (error) throw error;
    return _flattenSession(data);
};

// ── Analysis (Vercel Proxy to RunPod) ───────────────────────────────────────

export const startAnalysis = async (sessionId) => {
    const session = await getSession(sessionId);
    
    // Call our Vercel Proxy instead of RunPod directly
    const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            input: {
                action: 'analyze',
                session_id: sessionId,
                video_url: session.video_url
            }
        })
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to start analysis');

    // Update session status to queued
    await supabase.from('sessions').update({ status: 'queued' }).eq('id', sessionId);
    
    return data; // This is the RunPod Job ID
};

/**
 * Start SAMURAI player tracking via Vercel → RunPod.
 * Called from Dashboard when user selects a player bbox on the Configuration page.
 */
export const startTracking = async (sessionId, bbox, frame = 0) => {
    const session = await getSession(sessionId);

    const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            input: {
                action: 'track',
                session_id: sessionId,
                video_url: session.video_url,
                bbox: bbox,
                frame: frame
            }
        })
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to start tracking');

    // Update session status
    await supabase.from('sessions').update({ status: 'tracking' }).eq('id', sessionId);

    return data;
};

/**
 * Multi-segment tracking: user picks the same player at N keyframes spaced
 * across the video. Backend spawns N parallel SAMURAI subprocesses (sharing
 * the GPU) and runs the merged streaming analysis concurrently — for an
 * hour-long match this turns a ~40 min sequential pipeline into ~10 min.
 *
 * segments: [{ frame: number, bbox: {x1,y1,x2,y2} }, ...]
 */
/**
 * Persist the user's match-period selection to session.extra so the
 * MultiSegmentConfig + Dashboard can read it later (including after
 * a "Back to Trim" round-trip — point D in the design).
 *
 * periods: [{start: number, end: number}, ...] in seconds.
 */
export const saveMatchPeriods = async (sessionId, periods) => {
    // Stored as match_periods_sec inside the JSONB extra column; the
    // backend writer uses the same field on its side.
    const { data: row, error: getErr } = await supabase
        .from('sessions')
        .select('extra')
        .eq('id', sessionId)
        .single();
    if (getErr) throw getErr;

    let extra = row?.extra;
    if (typeof extra === 'string') {
        try { extra = JSON.parse(extra); } catch { extra = {}; }
    }
    extra = { ...(extra || {}), match_periods_sec: periods };

    const { error } = await supabase
        .from('sessions')
        .update({ extra })
        .eq('id', sessionId);
    if (error) throw error;
};

export const startTrackingMulti = async (sessionId, segments, matchPeriodsFrames = null) => {
    const session = await getSession(sessionId);

    // matchPeriodsFrames: [{startFrame, endFrame}, ...] — sent to backend so
    // it can skip non-match frames in analysis + render
    const input = {
        action: 'track',
        session_id: sessionId,
        video_url: session.video_url,
        segments,
    };
    if (matchPeriodsFrames && matchPeriodsFrames.length > 0) {
        input.match_periods = matchPeriodsFrames.map((p) => [
            p.startFrame, p.endFrame,
        ]);
    }

    const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to start multi-segment tracking');

    await supabase.from('sessions').update({ status: 'tracking' }).eq('id', sessionId);
    return data;
};

/**
 * Detect players on a specific frame entirely client-side:
 *   1. Pull the frame out of the video in the browser (canvas)
 *   2. POST it to /api/detect_frame which calls the Roboflow hosted API
 *
 * Skips RunPod GPU entirely — no cold-start, no idle time while the user
 * picks a player. Falls back to the old RunPod path on failure so the UI
 * still works even if Roboflow is down or out of quota.
 */
export const analyzeFrame = async (sessionId, frameIndex = 0) => {
    const session = await getSession(sessionId);

    try {
        const fps = session.video_fps || 25;
        const { dataUrl, base64, width, height } = await captureVideoFrame(
            session.video_url, frameIndex, fps
        );

        // 15-second timeout — Roboflow is normally < 2s but on a cold
        // edge it can hang. Without a timeout the user sees an infinite
        // spinner with no way out.
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 15000);
        let res;
        try {
            res = await fetch('/api/detect_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_base64: base64 }),
                signal: controller.signal,
            });
        } catch (e) {
            if (e.name === 'AbortError') {
                throw new Error('Roboflow timed out after 15s. Try again.');
            }
            throw e;
        } finally {
            clearTimeout(timeoutId);
        }

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.error || `Detect frame failed (${res.status})`);
        }

        const data = await res.json();
        return {
            players_data: data.players || [],
            annotated_frame_url: dataUrl,   // raw frame; Configuration draws SVG boxes over it
            image_dimensions: data.image_dimensions || { width, height },
        };
    } catch (e) {
        console.warn('Roboflow detect_frame failed, falling back to RunPod:', e);
        // Fallback: the original RunPod path
        const res = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                input: {
                    action: 'detect_frame',
                    session_id: sessionId,
                    video_url: session.video_url,
                    frame: frameIndex,
                },
            }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to analyze frame');

        let output;
        if (data.status === 'COMPLETED') output = data.output;
        else if (data.id) output = await pollJobResult(data.id);
        else output = data.output || data;

        return {
            players_data: output.players || output.players_data || [],
            annotated_frame_url: output.annotated_frame_url || null,
            image_dimensions: output.image_dimensions || null,
        };
    }
};

export const queueFeature = async (sessionId, feature) => {
    const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            input: {
                action: 'feature',
                feature: feature,
                session_id: sessionId
            }
        })
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to queue feature');
    return data;
};

// ── Tasks & Real-time ────────────────────────────────────────────────────────

/**
 * Fetch the current user's sessions, newest first. Used by the Sessions
 * library page so users can find old analyses without relying on the
 * 5-item localStorage "Recent" list.
 *
 * Returns: [{id, status, created_at, fileName?, ...}, ...]
 * fileName is parsed out of video_url's path when present.
 */
export const listMySessions = async ({ limit = 50, query: q = '' } = {}) => {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return [];
    let req = supabase
        .from('sessions')
        .select('id, status, created_at, updated_at, video_url, progress, stage, error')
        .eq('user_id', user.id)
        // updated_at, not created_at — so a session that's currently being
        // re-analysed bubbles to the top, and the user sees activity at a
        // glance. created_at stays in the result for the "uploaded X ago" label.
        .order('updated_at', { ascending: false })
        .limit(limit);
    if (q) {
        // ilike on video_url catches filenames the user typed into search
        req = req.ilike('video_url', `%${q}%`);
    }
    const { data, error } = await req;
    if (error) throw error;
    return (data || []).map((row) => ({
        ...row,
        fileName: decodeURIComponent(
            (row.video_url || '').split('/').pop()?.split('?')[0] || row.id.slice(0, 8)
        ),
    }));
};

/**
 * Delete a session row. Tasks are wiped via the ON DELETE CASCADE FK
 * (added in the cleanup migration). Storage objects (the uploaded video
 * + R2 artifacts) are *not* removed here — the nightly pg_cron job picks
 * them up. Deleting them inline would need 2 extra round-trips per session
 * and isn't worth the latency for a UI-driven delete.
 */
export const deleteSession = async (sessionId) => {
    const { error } = await supabase
        .from('sessions')
        .delete()
        .eq('id', sessionId);
    if (error) throw error;
};

export const listTasks = async (sessionId) => {
    const { data, error } = await supabase
        .from('tasks')
        .select('*')
        .eq('session_id', sessionId)
        .order('created_at', { ascending: true });
    
    if (error) throw error;
    return data;
};

/**
 * The magic of Supabase: Subscribe to changes in real-time.
 * No more polling!
 */
export const subscribeSession = (sessionId, handlers = {}) => {
    const sessionChannel = supabase
        .channel(`session:${sessionId}`)
        .on('postgres_changes', {
            event: '*',
            schema: 'public',
            table: 'sessions',
            filter: `id=eq.${sessionId}`
        }, (payload) => {
            if (handlers.onSession) handlers.onSession(_flattenSession(payload.new));
        })
        .on('postgres_changes', {
            event: '*',
            schema: 'public',
            table: 'tasks',
            filter: `session_id=eq.${sessionId}`
        }, (payload) => {
            if (handlers.onTask) handlers.onTask(payload.new);
        })
        .subscribe();

    return () => supabase.removeChannel(sessionChannel);
};

// ── Compatibility Shims (keeping UI from breaking) ──────────────────────────
export const getSummary = async (sessionId) => {
    const { data } = await supabase.from('tasks').select('result').eq('session_id', sessionId).eq('task_type', 'ai_summary').maybeSingle();
    return data?.result || {};
};

export const artifactUrl = (sessionId, relPath) => {
    // If it's already a full URL (like R2), return it.
    if (relPath?.startsWith('http')) return relPath;
    return relPath;
};

export default {
    uploadVideo,
    getSession,
    startAnalysis,
    startTracking,
    startTrackingMulti,
    analyzeFrame,
    queueFeature,
    listTasks,
    subscribeSession
};
