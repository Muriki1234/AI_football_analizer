import { useEffect, useState } from 'react';
import { Link, Navigate, useLocation } from 'react-router-dom';
import { HiEnvelope, HiLockClosed, HiEye, HiEyeSlash, HiDevicePhoneMobile } from 'react-icons/hi2';
import { IoFootball } from 'react-icons/io5';
import { supabase } from '../lib/supabase';
import { authErrorMessage, normalizePhone, safeReturnPath } from '../lib/auth';
import { useAuth } from '../auth/AuthContext';
import './Login.css';

const COUNTRY_CODES = [['+86', 'China +86'], ['+64', 'New Zealand +64'], ['+61', 'Australia +61'], ['+1', 'US / Canada +1'], ['+44', 'UK +44'], ['+852', 'Hong Kong +852'], ['+853', 'Macao +853'], ['+886', 'Taiwan +886'], ['+65', 'Singapore +65'], ['+81', 'Japan +81'], ['+82', 'South Korea +82'], ['+49', 'Germany +49'], ['+33', 'France +33'], ['+91', 'India +91']];

export default function Login() {
    const { user, loading: restoring, error: restoreError, recovery } = useAuth();
    const location = useLocation();
    const returnTo = safeReturnPath(location.state?.from);
    const [method, setMethod] = useState('email');
    const [email, setEmail] = useState('');
    const [phone, setPhone] = useState('');
    const [prefix, setPrefix] = useState('+86');
    const [password, setPassword] = useState('');
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
    const otpAvailable = method === 'phone' ? config?.phoneOtp : config?.emailOtp;
    const switchMethod = (next) => {
        setMethod(next); setChallenge(null); setCode(''); setPassword(''); setError(''); setNotice('');
    };

    const sendCode = async () => {
        if (waitSeconds || busy || !otpAvailable) return;
        setBusy(true); setError(''); setNotice('');
        try {
            const target = challenge?.target || (method === 'phone' ? normalizePhone(phone, prefix) : email.trim().toLowerCase());
            if (method !== 'phone' && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(target)) {
                setError('Enter a valid email address.'); return;
            }
            const payload = method === 'phone'
                ? { phone: target, options: { shouldCreateUser: true, channel: 'sms' } }
                : { email: target, options: { shouldCreateUser: true, emailRedirectTo: `${window.location.origin}/auth/callback` } };
            const result = await supabase.auth.signInWithOtp(payload);
            if (result.error) throw result.error;
            setChallenge({ target, type: method === 'phone' ? 'sms' : 'email' });
            setCode('');
            setNotice(method === 'phone' ? 'Code sent. Check your text messages.' : 'Code sent. Check your inbox and spam folder.');
            const time = Date.now(); setNow(time); setRetryAt(time + 60000);
        } catch (e) {
            setError(e.message?.startsWith('Enter a valid') ? e.message : authErrorMessage(e));
            if (e.status === 429) { const time = Date.now(); setNow(time); setRetryAt(time + 60000); }
        } finally { setBusy(false); }
    };

    const submit = async (event) => {
        event.preventDefault();
        if (busy) return;
        if (!challenge && ['email', 'phone'].includes(method)) { await sendCode(); return; }
        setBusy(true); setError(''); setNotice('');
        try {
            if (challenge) {
                if (!/^\d{6,10}$/.test(code)) { setError('Enter the code from your message.'); return; }
                const field = challenge.type === 'sms' ? 'phone' : 'email';
                const { error: verifyError } = await supabase.auth.verifyOtp({ [field]: challenge.target, token: code, type: challenge.type });
                if (verifyError) throw verifyError;
            } else if (method === 'password') {
                const result = await supabase.auth.signInWithPassword({ email: email.trim().toLowerCase(), password });
                if (result.error) throw result.error;
            } else if (method === 'forgot') {
                if (!config?.emailOtp) { setError('Password recovery is temporarily unavailable. Please contact support.'); return; }
                const result = await supabase.auth.resetPasswordForEmail(email.trim().toLowerCase(), { redirectTo: `${window.location.origin}/reset-password` });
                if (result.error) throw result.error;
                setNotice('If an account exists for this email, we have sent password reset instructions.');
                const time = Date.now(); setNow(time); setRetryAt(time + 60000);
            }
        } catch (e) { setError(authErrorMessage(e)); }
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
            <h1 className="login-card__title" id="login-title">{method === 'forgot' ? 'Reset your password' : challenge ? 'Enter your code' : 'Welcome to FootNova'}</h1>
            <p className="login-card__subtitle">{challenge ? `We sent a code to ${challenge.target}` : method === 'password' ? 'Sign in with your existing account' : method === 'forgot' ? 'We will email you reset instructions' : 'Sign in or create an account with a verification code'}</p>
            {!challenge && method !== 'forgot' && <div className="login-methods" role="group" aria-label="Sign-in method">
                <button type="button" aria-pressed={method === 'email'} disabled={busy} onClick={() => switchMethod('email')}><HiEnvelope /> Email code</button>
                <button type="button" aria-pressed={method === 'phone'} disabled={busy} onClick={() => switchMethod('phone')}><HiDevicePhoneMobile /> Phone code</button>
            </div>}
            {(error || restoreError) && <p className="auth-message auth-message--error" role="alert">{error || restoreError}</p>}
            {notice && <p className="auth-message" role="status">{notice}</p>}
            {configError && <p className="auth-message auth-message--error" role="alert">Unable to load sign-in methods. <button type="button" className="auth-link" onClick={() => { setConfigError(false); setConfigAttempt((n) => n + 1); }}>Try again</button></p>}
            {!challenge && ['email', 'phone', 'forgot'].includes(method) && config && !otpAvailable && <p className="auth-message" role="status">{method === 'phone' ? 'Phone sign-in is not available yet. Please use email or contact support.' : 'Email codes are not available yet. Existing customers can use their password. Please contact support for a new account.'}</p>}
            <form className="login-form" onSubmit={submit}>
                {challenge ? <div className="login-field">
                    <label htmlFor="login-code">Verification code</label>
                    <div className="login-input-wrap"><HiLockClosed className="login-input-icon" /><input id="login-code" inputMode="numeric" autoComplete="one-time-code" autoFocus value={code} onChange={(e) => setCode(e.target.value.replace(/\D/g, '').slice(0, 10))} minLength={6} maxLength={10} required disabled={busy} /></div>
                </div> : method === 'phone' ? <>
                    <div className="login-field"><label htmlFor="country-code">Country / region</label><select id="country-code" className="login-country" value={prefix} onChange={(e) => setPrefix(e.target.value)} disabled={busy}>{COUNTRY_CODES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></div>
                    <div className="login-field"><label htmlFor="login-phone">Mobile number</label><div className="login-input-wrap"><HiDevicePhoneMobile className="login-input-icon" /><input id="login-phone" type="tel" autoComplete="tel-national" placeholder="Mobile number" value={phone} onChange={(e) => setPhone(e.target.value)} maxLength={30} required disabled={busy} /></div><p className="login-hint">Other country? Enter the full number starting with +.</p></div>
                </> : <div className="login-field"><label htmlFor="login-email">Email</label><div className="login-input-wrap"><HiEnvelope className="login-input-icon" /><input id="login-email" type="email" autoComplete="email" placeholder="you@example.com" value={email} onChange={(e) => setEmail(e.target.value)} maxLength={254} required disabled={busy} /></div></div>}
                {method === 'password' && <div className="login-field"><label htmlFor="login-password">Password</label><div className="login-input-wrap"><HiLockClosed className="login-input-icon" /><input id="login-password" type={showPassword ? 'text' : 'password'} autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} required disabled={busy} /><button type="button" className="login-pwd-toggle" aria-label={showPassword ? 'Hide password' : 'Show password'} onClick={() => setShowPassword((v) => !v)}>{showPassword ? <HiEyeSlash /> : <HiEye />}</button></div><div className="login-forgot"><button type="button" className="auth-link" onClick={() => switchMethod('forgot')} disabled={busy}>Forgot password?</button></div></div>}
                <button type="submit" className="btn btn-primary btn-lg login-submit" disabled={busy || (!challenge && method !== 'password' && (!otpAvailable || waitSeconds > 0))}>{busy ? 'Please wait…' : challenge ? 'Verify and sign in' : method === 'password' ? 'Sign in' : waitSeconds ? `Try again in ${waitSeconds}s` : method === 'forgot' ? 'Send reset email' : 'Send verification code'}</button>
            </form>
            {challenge ? <div className="login-actions"><button type="button" className="auth-link" disabled={busy || waitSeconds > 0} onClick={sendCode}>{waitSeconds ? `Resend code in ${waitSeconds}s` : 'Resend code'}</button><button type="button" className="auth-link" disabled={busy} onClick={() => { setChallenge(null); setCode(''); setError(''); setNotice(''); }}>Use a different {method === 'phone' ? 'number' : 'email'}</button></div> : <p className="login-toggle"><button type="button" disabled={busy} onClick={() => switchMethod(method === 'password' || method === 'forgot' ? 'email' : 'password')}>{method === 'password' || method === 'forgot' ? 'Sign in with a verification code' : 'Already have a password? Sign in'}</button></p>}
            <p className="login-hint login-hint--center">Email and phone sign-ins create separate accounts. Use the same method each time to find your analyses.</p>
            <p className="login-toggle"><Link to="/">Back to home</Link></p>
        </section>
    </main>;
}
