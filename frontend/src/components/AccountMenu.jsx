import { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { useAuth } from '../auth/AuthContext';

export default function AccountMenu() {
    const { user, loading, signOut } = useAuth();
    const location = useLocation();
    const navigate = useNavigate();
    const [busy, setBusy] = useState(false);
    if (loading || /^\/(login|auth|reset-password)(\/|$)/.test(location.pathname)) return null;
    const logout = async () => {
        setBusy(true);
        try { await signOut(); navigate('/login', { replace: true }); }
        catch { toast.error('Unable to sign out. Please check your connection and try again.'); }
        finally { setBusy(false); }
    };
    return <nav className="account-menu" aria-label="Account">
        {user ? <>
            <span className="account-menu__identity" title={user.email || user.phone}>{user.email || `+${user.phone?.replace(/^\+/, '')}`}</span>
            <Link to="/sessions">My analyses</Link>
            <button type="button" disabled={busy} onClick={logout}>{busy ? 'Signing out…' : 'Sign out'}</button>
        </> : <Link className="btn btn-ghost" to="/login">Sign in</Link>}
    </nav>;
}
