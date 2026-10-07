import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.SUPABASE_URL || process.env.VITE_SUPABASE_URL;
const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY || process.env.VITE_SUPABASE_ANON_KEY;
let cachedClient;
const options = { auth: { persistSession: false, autoRefreshToken: false, detectSessionInUrl: false } };

export function getPublicAuthClient() {
    if (!SUPABASE_URL || !SUPABASE_ANON_KEY) throw new Error('Supabase configuration missing');
    cachedClient ||= createClient(SUPABASE_URL, SUPABASE_ANON_KEY, options);
    return cachedClient;
}

export function getUserClient(token) {
    getPublicAuthClient();
    return createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
        ...options, global: { headers: { Authorization: `Bearer ${token}` } },
    });
}

export function extractJwt(req) {
    const header = req.headers.authorization || req.headers.Authorization;
    if (typeof header !== 'string') return '';
    const match = header.match(/^Bearer\s+(\S+)$/i);
    return match?.[1] || '';
}

// Supabase validates the token; the database also verifies that its auth session
// still exists. This rejects anonymous users and access tokens retained after logout.
export async function requireSupabaseUser(req, res) {
    res.setHeader('Cache-Control', 'no-store');
    const token = extractJwt(req);
    if (!token) { res.status(401).json({ error: 'Please sign in to continue.' }); return null; }
    let client;
    try { client = getPublicAuthClient(); }
    catch { res.status(503).json({ error: 'Sign-in is temporarily unavailable.' }); return null; }
    try {
        const { data, error } = await client.auth.getUser(token);
        if (error || !data?.user) {
            res.status(error?.status >= 500 ? 503 : 401).json({ error: 'Your sign-in is invalid or expired.' }); return null;
        }
        const user = data.user;
        if (user.is_anonymous || !(user.email_confirmed_at || user.phone_confirmed_at)) {
            res.status(403).json({ error: 'Please sign in with a verified email or phone number.' }); return null;
        }
        const active = await getUserClient(token).rpc('customer_session_active');
        if (active.error) { console.error('[auth] session validation unavailable'); res.status(503).json({ error: 'Unable to verify your sign-in. Please try again.' }); return null; }
        if (active.data !== true) { res.status(401).json({ error: 'Your sign-in has ended. Please sign in again.' }); return null; }
        return user;
    } catch {
        res.status(503).json({ error: 'Unable to verify your sign-in. Please try again.' }); return null;
    }
}

// Always verify ownership explicitly as well as relying on RLS.
export async function requireSessionOwner(req, res, sessionId, userJwt, expectedUserId) {
    if (typeof sessionId !== 'string' || !/^[A-Za-z0-9_-]{1,128}$/.test(sessionId)) {
        res.status(400).json({ error: 'A valid session_id is required.' }); return null;
    }
    try {
        const { data, error } = await getUserClient(userJwt)
            .from('sessions').select('id, user_id, video_url, status, extra, updated_at')
            .eq('id', sessionId).maybeSingle();
        if (error) { res.status(503).json({ error: 'Unable to load this analysis.' }); return null; }
        if (!data || !expectedUserId || data.user_id !== expectedUserId) {
            res.status(403).json({ error: 'This analysis is unavailable or belongs to another account.' }); return null;
        }
        return data;
    } catch {
        res.status(503).json({ error: 'Unable to load this analysis.' }); return null;
    }
}
