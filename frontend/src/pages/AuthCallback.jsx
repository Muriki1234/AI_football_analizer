import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabase';
import { isCustomer } from '../lib/auth';
import { useAuth } from '../auth/AuthContext';
import './Login.css';

// One exchange per URL, including React StrictMode's repeated effect setup.
let exchange;
let exchangedUrl;
export default function AuthCallback() {
    const { loading, recovery } = useAuth();
    const navigate = useNavigate();
    const [error, setError] = useState(() => {
        const url = new URL(window.location.href);
        return url.searchParams.has('error') || new URLSearchParams(url.hash.slice(1)).has('error')
            ? 'This link is invalid or expired. Please request a new verification code.' : '';
    });
    useEffect(() => {
        let cancelled = false;
        const url = new URL(window.location.href);
        const hash = new URLSearchParams(url.hash.slice(1));
        if (url.searchParams.has('error') || hash.has('error')) {
            window.history.replaceState(null, '', '/auth/callback');
            return () => { cancelled = true; };
        }
        if (loading) return undefined;
        if (exchangedUrl !== url.href) {
            exchangedUrl = url.href;
            exchange = (async () => {
                const code = url.searchParams.get('code');
                if (code) {
                    const result = await supabase.auth.exchangeCodeForSession(code);
                    if (result.error) throw result.error;
                }
                const { data, error: userError } = await supabase.auth.getUser();
                if (userError || !isCustomer(data.user)) throw new Error('Invalid confirmation');
                return data.user;
            })();
        }
        exchange.then(() => { if (!cancelled) navigate(recovery ? '/reset-password' : '/', { replace: true }); })
            .catch(() => { if (!cancelled) setError('This link is invalid or expired. Please request a new verification code.'); });
        return () => { cancelled = true; };
    }, [loading, recovery, navigate]);
    const params = new URLSearchParams(window.location.search + '&' + window.location.hash.slice(1));
    const hasError = params.has('error');
    return <main className="login-page"><section className="login-card"><h1 className="login-card__title">Confirming your sign-in</h1>
        {error || hasError ? <><p className="auth-message auth-message--error" role="alert">{error || 'This link is invalid or expired. Please request a new code.'}</p><Link to="/login" className="btn btn-primary">Return to sign in</Link></> : <p role="status">Please wait…</p>}
    </section></main>;
}
