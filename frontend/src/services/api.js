import { supabase } from '../lib/supabase';

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
    const sessionId = `sess_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
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
            video_url: videoUrl,
            status: 'uploaded'
        }]);

    if (dbError) throw dbError;

    return { session_id: sessionId, video_url: videoUrl };
};

export const getSession = async (sessionId) => {
    const { data, error } = await supabase
        .from('sessions')
        .select('*')
        .eq('id', sessionId)
        .single();
    
    if (error) throw error;
    return data;
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
            if (handlers.onSession) handlers.onSession(payload.new);
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
    const { data } = await supabase.from('tasks').select('result').eq('session_id', sessionId).eq('task_type', 'ai_summary').single();
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
    queueFeature,
    listTasks,
    subscribeSession
};
