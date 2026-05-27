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
