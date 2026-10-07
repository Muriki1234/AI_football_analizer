import { useEffect, useState } from 'react';
import { Link, Navigate, useLocation } from 'react-router-dom';
import { HiEnvelope, HiLockClosed, HiEye, HiEyeSlash } from 'react-icons/hi2';
import { IoFootball } from 'react-icons/io5';
import { supabase } from '../lib/supabase';
import { authErrorMessage, safeReturnPath } from '../lib/auth';
import { useAuth } from '../auth/AuthContext';
import './Login.css';

export default function Login() {
    const { user, loading: restoring, error: restoreError, recovery } = useAuth();
    const location = useLocation();
    const returnTo = safeReturnPath(location.state?.from);
    const [method, setMethod] = useState('password');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmation, setConfirmation] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [challenge, setChallenge] = useState(null);
    const [code, setCode] = useState('');
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState('');
    const [notice, setNotice] = useState('');
    const [retryAt, setRetryAt] = useState(0);
    const [now, setNow] = useState(Date.now);
    const [config, setConfig] = useState(null);
    const [configError, setConfigError] = useState(false);
    const [configAttempt, setConfigAttempt] = useState(0);

    useEffect(() => {
        const abort = new AbortController();
        fetch('/api/auth-config', { signal: abort.signal })
            .then(async (response) => {
                if (!response.ok) throw new Error('Configuration unavailable');
                return response.json();
            })
            .then((data) => { setConfig(data); setConfigError(false); })
            .catch((e) => { if (e.name !== 'AbortError') setConfigError(true); });
        return () => abort.abort();
    }, [configAttempt]);

    useEffect(() => {
        if (!retryAt) return undefined;
        const timer = setInterval(() => setNow(Date.now()), 1000);
        return () => clearInterval(timer);
    }, [retryAt]);

    const waitSeconds = Math.max(0, Math.ceil((retryAt - now) / 1000));
    const otpAvailable = config?.emailOtp === true;
    const needsEmail = method !== 'password';
    const switchMethod = (next) => {
        setMethod(next); setChallenge(null); setCode(''); setPassword(''); setConfirmation(''); setShowPassword(false); setError(''); setNotice('');
    };

    const startCooldown = () => {
        const time = Date.now(); setNow(time); setRetryAt(time + 60000);
    };
    const sendCode = async () => {
        if (waitSeconds || busy || !otpAvailable) return;
        setBusy(true); setError(''); setNotice('');
        try {
            const target = challenge?.target || email.trim().toLowerCase();
            if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(target)) {
                setError('Enter a valid email address.'); return;
            }
            const options = { emailRedirectTo: `${window.location.origin}/auth/callback` };
            let result;
            if (challenge?.signup) {
                result = await supabase.auth.resend({ type: 'signup', email: target, options });
            } else if (method === 'signup') {
                if (password.length < 10) { setError('Use at least 10 characters.'); return; }
                if (password !== confirmation) { setError('Passwords do not match.'); return; }
                result = await supabase.auth.signUp({ email: target, password, options });
            } else {
                result = await supabase.auth.signInWithOtp({ email: target, options: { ...options, shouldCreateUser: true } });
            }
            if (result.error) throw result.error;
            setChallenge({ target, signup: method === 'signup' });
            setPassword(''); setConfirmation(''); setCode('');
            // Supabase deliberately hides whether an address already has an account.
            setNotice(method === 'signup'
                ? 'Check your inbox and spam folder for a confirmation code. Already registered? Return to password sign-in.'
                : 'Code sent. Check your inbox and spam folder.');
            startCooldown();
        } catch (e) {
            setError(authErrorMessage(e));
            if (e.status === 429) startCooldown();
        } finally { setBusy(false); }
    };

    const submit = async (event) => {
        event.preventDefault();
        if (busy || (!challenge && method === 'forgot' && waitSeconds)) return;
        if (!challenge && ['email', 'signup'].includes(method)) { await sendCode(); return; }
        setBusy(true); setError(''); setNotice('');
        try {
            if (challenge) {
                if (!/^\d{6,10}$/.test(code)) { setError('Enter the code from your email.'); return; }
                const { error: verifyError } = await supabase.auth.verifyOtp({ email: challenge.target, token: code, type: 'email' });
                if (verifyError) throw verifyError;
            } else if (method === 'password') {
                const result = await supabase.auth.signInWithPassword({ email: email.trim().toLowerCase(), password });
                if (result.error) throw result.error;
            } else if (method === 'forgot') {
                if (!config?.emailOtp) { setError('Password recovery is temporarily unavailable. Please contact support.'); return; }
                const result = await supabase.auth.resetPasswordForEmail(email.trim().toLowerCase(), { redirectTo: `${window.location.origin}/reset-password` });
                if (result.error) throw result.error;
                setNotice('If an account exists for this email, we have sent password reset instructions.');
                startCooldown();
            }
        } catch (e) { setError(authErrorMessage(e)); if (e.status === 429) startCooldown(); }
        finally { setBusy(false); }
    };

    if (restoring) return <div className="auth-loading" role="status">Restoring your sign-in…</div>;
    if (recovery) return <Navigate to="/reset-password" replace />;
    if (user) return <Navigate to={returnTo} replace />;

    return <main className="login-page">
        <div className="bg-grid" />
        <div className="login-orb login-orb--1" /><div className="login-orb login-orb--2" />
        <section className="login-card" aria-labelledby="login-title">
            <Link className="login-card__logo" to="/"><IoFootball className="login-card__logo-icon" /><span>FootNova AI</span></Link>
            <h1 className="login-card__title" id="login-title">{method === 'forgot' ? 'Reset your password' : challenge ? 'Verify your email' : method === 'signup' ? 'Create your account' : 'Welcome to FootNova'}</h1>
            <p className="login-card__subtitle">{challenge ? `Enter the code sent to ${challenge.target}` : method === 'password' ? 'Sign in with your email and password' : method === 'signup' ? 'Verify your email once, then sign in with your password' : method === 'forgot' ? 'We will email you reset instructions' : 'Sign in or create an account with an email code'}</p>
            {!challenge && ['password', 'email'].includes(method) && <div className="login-methods" role="group" aria-label="Sign-in method">
                <button type="button" aria-pressed={method === 'password'} disabled={busy} onClick={() => switchMethod('password')}><HiLockClosed /> Password</button>
                <button type="button" aria-pressed={method === 'email'} disabled={busy} onClick={() => switchMethod('email')}><HiEnvelope /> Email code</button>
            </div>}
            {(error || restoreError) && <p className="auth-message auth-message--error" role="alert">{error || restoreError}</p>}
            {notice && <p className="auth-message" role="status">{notice}</p>}
            {configError && <p className="auth-message auth-message--error" role="alert">Unable to load sign-in methods. <button type="button" className="auth-link" onClick={() => { setConfigError(false); setConfigAttempt((n) => n + 1); }}>Try again</button></p>}
            {!challenge && needsEmail && config && !otpAvailable && <p className="auth-message" role="status">Email verification is not available yet. Existing customers can sign in with their password. Please contact support for a new account.</p>}
            <form className="login-form" onSubmit={submit}>
                {challenge ? <div className="login-field">
                    <label htmlFor="login-code">Verification code</label>
                    <div className="login-input-wrap"><HiLockClosed className="login-input-icon" /><input id="login-code" inputMode="numeric" autoComplete="one-time-code" autoFocus value={code} onChange={(e) => setCode(e.target.value.replace(/\D/g, '').slice(0, 10))} minLength={6} maxLength={10} required disabled={busy} /></div>
                </div> : <div className="login-field"><label htmlFor="login-email">Email</label><div className="login-input-wrap"><HiEnvelope className="login-input-icon" /><input id="login-email" type="email" autoComplete="email" placeholder="you@example.com" value={email} onChange={(e) => setEmail(e.target.value)} maxLength={254} required disabled={busy} /></div></div>}
                {!challenge && ['password', 'signup'].includes(method) && <div className="login-field"><label htmlFor="login-password">Password</label><div className="login-input-wrap"><HiLockClosed className="login-input-icon" /><input id="login-password" type={showPassword ? 'text' : 'password'} autoComplete={method === 'signup' ? 'new-password' : 'current-password'} value={password} onChange={(e) => setPassword(e.target.value)} minLength={method === 'signup' ? 10 : undefined} required disabled={busy} /><button type="button" className="login-pwd-toggle" aria-label={showPassword ? 'Hide password' : 'Show password'} onClick={() => setShowPassword((v) => !v)}>{showPassword ? <HiEyeSlash /> : <HiEye />}</button></div>{method === 'password' ? <div className="login-forgot"><button type="button" className="auth-link" onClick={() => switchMethod('forgot')} disabled={busy}>Forgot password?</button></div> : <p className="login-hint">Use at least 10 characters.</p>}</div>}
                {!challenge && method === 'signup' && <div className="login-field"><label htmlFor="signup-confirmation">Confirm password</label><div className="login-input-wrap"><HiLockClosed className="login-input-icon" /><input id="signup-confirmation" type={showPassword ? 'text' : 'password'} autoComplete="new-password" value={confirmation} onChange={(e) => setConfirmation(e.target.value)} minLength={10} required disabled={busy} /></div></div>}
                <button type="submit" className="btn btn-primary btn-lg login-submit" disabled={busy || (!challenge && needsEmail && (!otpAvailable || waitSeconds > 0))}>{busy ? 'Please wait…' : challenge ? 'Verify and sign in' : method === 'password' ? 'Sign in' : waitSeconds ? `Try again in ${waitSeconds}s` : method === 'forgot' ? 'Send reset email' : method === 'signup' ? 'Create account and send code' : 'Send verification code'}</button>
            </form>
            {challenge ? <div className="login-actions"><button type="button" className="auth-link" disabled={busy || !otpAvailable || waitSeconds > 0} onClick={sendCode}>{waitSeconds ? `Resend code in ${waitSeconds}s` : 'Resend code'}</button><button type="button" className="auth-link" disabled={busy} onClick={() => { setChallenge(null); setCode(''); setError(''); setNotice(''); }}>Use a different email</button><button type="button" className="auth-link" disabled={busy} onClick={() => switchMethod('password')}>Return to password sign-in</button></div> : <p className="login-toggle"><button type="button" disabled={busy} onClick={() => switchMethod(method === 'signup' || method === 'forgot' ? 'password' : 'signup')}>{method === 'signup' || method === 'forgot' ? 'Already registered? Sign in' : 'New to FootNova? Create an account'}</button></p>}
            {!challenge && method === 'email' && <p className="login-hint login-hint--center">Use the same email address to access your analyses. You can set a password from your account after signing in.</p>}
            <p className="login-toggle"><Link to="/">Back to home</Link></p>
        </section>
    </main>;
}
