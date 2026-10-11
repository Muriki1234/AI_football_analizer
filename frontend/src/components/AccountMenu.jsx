import { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { useAuth } from '../auth/AuthContext';
import { useLanguage } from '../i18n/LanguageContext';

export default function AccountMenu() {
    const { user, loading, signOut } = useAuth();
    const { lang, setLang, t } = useLanguage();
    const location = useLocation();
    const navigate = useNavigate();
    const [busy, setBusy] = useState(false);

    const isAuthRoute = /^\/(login|auth|reset-password)(\/|$)/.test(location.pathname);

    const logout = async () => {
        setBusy(true);
        try { await signOut(); navigate('/login', { replace: true }); }
        catch { toast.error(t('common.signOutError')); }
        finally { setBusy(false); }
    };

    const LangSwitch = (
        <div className="lang-switcher" style={{ display: 'inline-flex', alignItems: 'center', background: 'rgba(255,255,255,0.06)', borderRadius: '16px', padding: '2px', border: '1px solid rgba(255,255,255,0.1)' }}>
            <button
                type="button"
                onClick={() => setLang('zh')}
                className={`lang-btn ${lang === 'zh' ? 'active' : ''}`}
                style={{
                    padding: '2px 8px',
                    fontSize: '0.72rem',
                    fontWeight: 600,
                    borderRadius: '12px',
                    border: 'none',
                    cursor: 'pointer',
                    background: lang === 'zh' ? '#2563eb' : 'transparent',
                    color: lang === 'zh' ? '#ffffff' : '#94a3b8',
                    transition: 'all 0.15s ease'
                }}
            >中文</button>
            <button
                type="button"
                onClick={() => setLang('en')}
                className={`lang-btn ${lang === 'en' ? 'active' : ''}`}
                style={{
                    padding: '2px 8px',
                    fontSize: '0.72rem',
                    fontWeight: 600,
                    borderRadius: '12px',
                    border: 'none',
                    cursor: 'pointer',
                    background: lang === 'en' ? '#2563eb' : 'transparent',
                    color: lang === 'en' ? '#ffffff' : '#94a3b8',
                    transition: 'all 0.15s ease'
                }}
            >EN</button>
        </div>
    );

    if (loading) return null;

    if (isAuthRoute) {
        return (
            <nav className="account-menu" aria-label="Language">
                {LangSwitch}
            </nav>
        );
    }

    return (
        <nav className="account-menu" aria-label="Account">
            {user ? (
                <>
                    <span className="account-menu__identity" title={user.email || user.phone}>
                        {user.email || `+${user.phone?.replace(/^\+/, '')}`}
                    </span>
                    <Link to="/sessions">{t('common.myAnalyses')}</Link>
                    {user.email && <Link to="/reset-password">{t('common.password')}</Link>}
                    <button type="button" disabled={busy} onClick={logout}>
                        {busy ? t('common.signingOut') : t('common.signOut')}
                    </button>
                </>
            ) : (
                <Link className="btn btn-ghost" to="/login">{t('common.signIn')}</Link>
            )}
            {LangSwitch}
        </nav>
    );
}
