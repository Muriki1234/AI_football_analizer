import { useState } from 'react';
import { Link, Navigate } from 'react-router-dom';
import { supabase } from '../lib/supabase';
import { authErrorMessage } from '../lib/auth';
import { useAuth } from '../auth/AuthContext';
import './Login.css';

export default function ResetPassword() {
    const { user, loading, finishRecovery } = useAuth();
    const [password, setPassword] = useState('');
    const [confirmation, setConfirmation] = useState('');
    const [error, setError] = useState('');
    const [busy, setBusy] = useState(false);
    const [done, setDone] = useState(false);
    const submit = async (event) => {
        event.preventDefault();
        if (busy) return;
        if (password !== confirmation) { setError('Passwords do not match.'); return; }
        if (password.length < 10) { setError('Use at least 10 characters.'); return; }
        setBusy(true); setError('');
        try {
            const result = await supabase.auth.updateUser({ password });
            if (result.error) throw result.error;
            finishRecovery(); setPassword(''); setConfirmation(''); setDone(true);
        } catch (e) { setError(authErrorMessage(e)); }
        finally { setBusy(false); }
    };
    if (loading) return <div className="auth-loading" role="status">Checking your reset link…</div>;
    if (done) return <Navigate to="/" replace />;
    return <main className="login-page"><section className="login-card">
        <h1 className="login-card__title">Set a new password</h1>
        {user ? <><p className="login-card__subtitle">Use at least 10 characters.</p>
            {error && <p className="auth-message auth-message--error" role="alert">{error}</p>}
            <form className="login-form" onSubmit={submit}>
                <div className="login-field"><label htmlFor="new-password">New password</label><div className="login-input-wrap"><input id="new-password" type="password" autoComplete="new-password" value={password} onChange={(e) => setPassword(e.target.value)} minLength={10} required disabled={busy} /></div></div>
                <div className="login-field"><label htmlFor="confirm-password">Confirm password</label><div className="login-input-wrap"><input id="confirm-password" type="password" autoComplete="new-password" value={confirmation} onChange={(e) => setConfirmation(e.target.value)} minLength={10} required disabled={busy} /></div></div>
                <button className="btn btn-primary login-submit" disabled={busy}>{busy ? 'Saving…' : 'Save new password'}</button>
            </form></> : <><p className="auth-message" role="alert">This reset link is invalid or expired. Request a new link from the sign-in page.</p><Link to="/login" className="btn btn-primary">Return to sign in</Link></>}
    </section></main>;
}
