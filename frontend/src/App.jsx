import { lazy, Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation, Navigate } from 'react-router-dom';
import { AnimatePresence, motion as Motion } from 'framer-motion';
import { ProgressProvider } from './components/ProgressBar';
import Welcome from './pages/Welcome';
const Upload = lazy(() => import('./pages/Upload'));
const Trimmer = lazy(() => import('./pages/Trimmer'));
const MultiSegmentConfig = lazy(() => import('./pages/MultiSegmentConfig'));
const Sessions = lazy(() => import('./pages/Sessions'));
const Dashboard = lazy(() => import('./pages/Dashboard'));
// PlayerLibrary / PlayerProfile 是 100% mock data，没接后端，不暴露路由
// 直到真的有 player metadata pipeline 再上线。
import Login from './pages/Login';
import AuthProvider from './auth/AuthProvider';
import RequireAuth from './auth/RequireAuth';
import AuthCallback from './pages/AuthCallback';
import ResetPassword from './pages/ResetPassword';
import AccountMenu from './components/AccountMenu';
import { useAuth } from './auth/AuthContext';
import './index.css';

const pageVariants = {
    initial: { opacity: 0, y: 12 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.35, ease: 'easeOut' } },
    exit: { opacity: 0, y: -12, transition: { duration: 0.2 } },
};

function AnimatedRoutes() {
    const location = useLocation();
    const { user } = useAuth();
    return (
        <AnimatePresence mode="wait">
            <Routes location={location} key={location.key}>
                <Route path="/login" element={<PageWrap><Login /></PageWrap>} />
                <Route path="/auth/callback" element={<AuthCallback />} />
                <Route path="/reset-password" element={<ResetPassword />} />
                <Route path="/" element={<PageWrap><Welcome key={user?.id || 'visitor'} /></PageWrap>} />
                <Route path="/upload" element={<RequireAuth><PageWrap><Upload /></PageWrap></RequireAuth>} />
                <Route path="/trim" element={<RequireAuth><PageWrap><Trimmer /></PageWrap></RequireAuth>} />
                {/* /configure (legacy single-pick) redirects to multi-segment */}
                <Route path="/configure" element={<RequireAuth><PageWrap><MultiSegmentConfig /></PageWrap></RequireAuth>} />
                <Route path="/configure-multi" element={<RequireAuth><PageWrap><MultiSegmentConfig /></PageWrap></RequireAuth>} />
                <Route path="/sessions" element={<RequireAuth><PageWrap><Sessions /></PageWrap></RequireAuth>} />
                <Route path="/dashboard" element={<RequireAuth><PageWrap><Dashboard /></PageWrap></RequireAuth>} />
                <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
        </AnimatePresence>
    );
}

function PageWrap({ children }) {
    return (
        <Motion.div
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
        >
            {children}
        </Motion.div>
    );
}

export default function App() {
    return (
        <AuthProvider>
        <Router>
            <ProgressProvider>
                <AccountMenu />
                <Suspense fallback={<div className="auth-loading" role="status">Loading…</div>}>
                    <AnimatedRoutes />
                </Suspense>
            </ProgressProvider>
        </Router>
        </AuthProvider>
    );
}
