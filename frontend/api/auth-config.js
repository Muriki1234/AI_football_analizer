import { getPublicAuthClient } from './_authMiddleware.js';

// Readiness is explicitly enabled after a real provider delivery test, rather
// than implying the built-in Supabase testing mailer can serve customers.
export default async function handler(req, res) {
    res.setHeader('Cache-Control', 'no-store');
    if (req.method !== 'GET') {
        res.setHeader('Allow', 'GET');
        return res.status(405).json({ error: 'Method not allowed' });
    }
    try {
        getPublicAuthClient();
        const url = process.env.SUPABASE_URL || process.env.VITE_SUPABASE_URL;
        const key = process.env.SUPABASE_ANON_KEY || process.env.VITE_SUPABASE_ANON_KEY;
        const response = await fetch(`${url.replace(/\/$/, '')}/auth/v1/settings`, {
            headers: { apikey: key }, signal: AbortSignal.timeout(8000),
        });
        if (!response.ok) throw new Error('Auth settings unavailable');
        const settings = await response.json();
        return res.status(200).json({
            emailOtp: settings.external?.email === true && process.env.AUTH_EMAIL_OTP_ENABLED === 'true',
            phoneOtp: settings.external?.phone === true && process.env.AUTH_PHONE_OTP_ENABLED === 'true',
            password: settings.external?.email === true,
        });
    } catch {
        return res.status(503).json({ error: 'Sign-in configuration temporarily unavailable.' });
    }
}
