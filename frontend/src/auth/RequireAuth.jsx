import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from './AuthContext';

export default function RequireAuth({ children }) {
    const { user, loading, recovery } = useAuth();
    const location = useLocation();
    if (loading) return <div className="auth-loading" role="status">Restoring your sign-in…</div>;
    if (recovery) return <Navigate to="/reset-password" replace />;
    if (!user) return <Navigate to="/login" replace state={{ from: location.pathname + location.search }} />;
    return <div key={user.id}>{children}</div>;
}
