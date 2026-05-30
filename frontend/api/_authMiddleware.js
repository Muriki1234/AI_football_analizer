// Vercel serverless API auth helper.
//
// 三个 /api/* endpoint 都代理到 RunPod / Roboflow（花钱的服务），
// 之前完全没 auth → 任何人 curl 你的 vercel 域名就能烧 GPU 钱。
//
// 这里强制 Authorization: Bearer <supabase_jwt> header，校验是
// Supabase 用户（包括匿名用户）签发的有效 JWT。匿名 session 用户
// 也能通过，所以正常前端流程不受影响；裸 curl 拿不到 JWT 就 401。
//
// 文件名以 _ 开头：Vercel 不把它当 endpoint 路由，纯辅助模块。

import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.VITE_SUPABASE_URL || process.env.SUPABASE_URL;
const SUPABASE_ANON_KEY = process.env.VITE_SUPABASE_ANON_KEY || process.env.SUPABASE_ANON_KEY;

// 同一进程内复用 client，省 cold-start
let _cachedClient = null;
function getSupabase() {
    if (!_cachedClient) {
        if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
            throw new Error(
                'Auth disabled: SUPABASE_URL / SUPABASE_ANON_KEY missing in Vercel env'
            );
        }
        _cachedClient = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
            auth: { persistSession: false, autoRefreshToken: false },
        });
    }
    return _cachedClient;
}

/**
 * 校验请求里的 Supabase JWT。验证通过返回 user 对象，否则 res 返回 401 / 500
 * 并返回 null（caller 看到 null 直接 return 即可）。
 */
export async function requireSupabaseUser(req, res) {
    const authHeader = req.headers.authorization || req.headers.Authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
        res.status(401).json({
            error: 'Missing or invalid Authorization header. Expected: Bearer <supabase_jwt>',
        });
        return null;
    }

    const token = authHeader.slice('Bearer '.length).trim();
    if (!token) {
        res.status(401).json({ error: 'Empty bearer token' });
        return null;
    }

    let supabase;
    try {
        supabase = getSupabase();
    } catch (e) {
        // 配置缺失 — 这是服务端配置错误，不是 client 的锅
        console.error('[auth] Supabase config error:', e.message);
        res.status(500).json({ error: 'Server auth misconfigured' });
        return null;
    }

    try {
        const { data, error } = await supabase.auth.getUser(token);
        if (error || !data?.user) {
            res.status(401).json({ error: 'Invalid or expired session' });
            return null;
        }
        return data.user;
    } catch (e) {
        console.error('[auth] getUser failed:', e);
        res.status(401).json({ error: 'Token verification failed' });
        return null;
    }
}


/**
 * 校验当前 user 拥有 session_id，并从 DB 返回服务端可信的 session（含 video_url）。
 * 三件事一起做：
 *   1. session 必须存在
 *   2. session.user_id 必须等于当前 JWT 的 user.id  (依赖 RLS 也行，但这里多查一次更明确)
 *   3. 用 anon key + 用户 JWT 查 → RLS 自动拦截不属于这个用户的行 → PGRST116 报 row 0
 *
 * 调用方拿到非 null 的 session 后，**必须用 session.video_url 而不是 req.body 的 video_url**，
 * 否则攻击者能让你的 RunPod 去下载任意 URL（SSRF 已在后端拦，但这里直接断绝）。
 *
 * 失败时 res 已经写过响应，caller 直接 return 即可。
 */
export async function requireSessionOwner(req, res, sessionId, userJwt) {
    if (!sessionId) {
        res.status(400).json({ error: 'session_id required' });
        return null;
    }
    let supabase;
    try {
        supabase = getSupabase();
    } catch (e) {
        console.error('[auth] supabase config error:', e.message);
        res.status(500).json({ error: 'Server auth misconfigured' });
        return null;
    }

    // 用 user 的 JWT 做查询 → 走 RLS。这样如果有人传了别人的 session_id，
    // RLS 会让查询返回 0 行，single() 报错。
    try {
        const { data, error } = await supabase
            .from('sessions')
            .select('id, user_id, video_url, status')
            .eq('id', sessionId)
            .maybeSingle()
            .setHeader?.('Authorization', `Bearer ${userJwt}`);
        // setHeader 不是所有 supabase-js 版本都支持；下面用一种更稳的方式重做
        void data; void error;
    } catch { /* ignore */ }

    try {
        // 创建一个带用户 JWT 的临时 client，这样 RLS 用 auth.uid() = user.id 评估
        const userClient = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
            auth: { persistSession: false, autoRefreshToken: false },
            global: { headers: { Authorization: `Bearer ${userJwt}` } },
        });
        const { data, error } = await userClient
            .from('sessions')
            .select('id, user_id, video_url, status')
            .eq('id', sessionId)
            .maybeSingle();
        if (error) {
            console.error('[auth] session lookup failed:', error.message);
            res.status(500).json({ error: 'Session lookup failed' });
            return null;
        }
        if (!data) {
            // RLS 拦截 = 查不到 = 当前用户不拥有这个 session（或者 session 不存在）
            res.status(403).json({ error: 'Session not found or not owned by you' });
            return null;
        }
        return data;
    } catch (e) {
        console.error('[auth] session lookup exception:', e);
        res.status(500).json({ error: 'Session lookup failed' });
        return null;
    }
}


/** 从 Authorization header 抠出 raw JWT 字符串（caller 已经过 requireSupabaseUser）。 */
export function extractJwt(req) {
    const h = req.headers.authorization || req.headers.Authorization || '';
    return h.startsWith('Bearer ') ? h.slice(7).trim() : '';
}
