import { useState } from 'react';
import { Link, Navigate } from 'react-router-dom';
import { supabase } from '../lib/supabase';
import { authErrorMessage } from '../lib/auth';
import { useAuth } from '../auth/AuthContext';
import { useLanguage } from '../i18n/LanguageContext';
import './Login.css';

export default function ResetPassword() {
    const { t, language } = useLanguage();
    const { user, loading, finishRecovery } = useAuth();
    const [password, setPassword] = useState('');
    const [confirmation, setConfirmation] = useState('');
    const [error, setError] = useState('');
    const [busy, setBusy] = useState(false);
    const [done, setDone] = useState(false);

    const submit = async (event) => {
        event.preventDefault();
        if (busy) return;
        if (password !== confirmation) { setError(t('auth.passwordsNotMatch')); return; }
        if (password.length < 10) { setError(t('auth.passwordMinLength')); return; }
        setBusy(true); setError('');
        try {
            const result = await supabase.auth.updateUser({ password });
            if (result.error) throw result.error;
            finishRecovery(); setPassword(''); setConfirmation(''); setDone(true);
        } catch (e) { setError(authErrorMessage(e)); }
        finally { setBusy(false); }
    };

    if (loading) return <div className="auth-loading" role="status">{t('auth.resetCheckingLink')}</div>;
    if (done) return <Navigate to="/" replace />;

    return (
        <main className="login-page">
            <section className="login-card">
                <h1 className="login-card__title">{t('auth.resetPasswordTitle')}</h1>
                {user ? (
                    <>
                        <p className="login-card__subtitle">{t('auth.passwordMinLength')}</p>
                        {error && <p className="auth-message auth-message--error" role="alert">{error}</p>}
                        <form className="login-form" onSubmit={submit}>
                            <div className="login-field">
                                <label htmlFor="new-password">{t('auth.passwordLabel')}</label>
                                <div className="login-input-wrap">
                                    <input
                                        id="new-password"
                                        type="password"
                                        autoComplete="new-password"
                                        placeholder={t('auth.passwordPlaceholder')}
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        minLength={10}
                                        required
                                        disabled={busy}
                                    />
                                </div>
                            </div>
                            <div className="login-field">
                                <label htmlFor="confirm-password">{t('auth.confirmPasswordLabel')}</label>
                                <div className="login-input-wrap">
                                    <input
                                        id="confirm-password"
                                        type="password"
                                        autoComplete="new-password"
                                        placeholder={t('auth.confirmPasswordPlaceholder')}
                                        value={confirmation}
                                        onChange={(e) => setConfirmation(e.target.value)}
                                        minLength={10}
                                        required
                                        disabled={busy}
                                    />
                                </div>
                            </div>
                            <button className="btn btn-primary login-submit" disabled={busy}>
                                {busy ? (language === 'zh' ? '保存中…' : 'Saving…') : t('auth.resetPasswordBtn')}
                            </button>
                        </form>
                        <p className="login-toggle"><Link to="/">{t('auth.backToHome')}</Link></p>
                    </>
                ) : (
                    <>
                        <p className="auth-message" role="alert">{t('auth.resetInvalidLink')}</p>
                        <Link to="/login" className="btn btn-primary">{t('auth.returnToPassword')}</Link>
                    </>
                )}
            </section>
        </main>
    );
}
