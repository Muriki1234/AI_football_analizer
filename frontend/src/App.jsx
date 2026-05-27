import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { ProgressProvider } from './components/ProgressBar';
import Welcome from './pages/Welcome';
import Upload from './pages/Upload';
import Trimmer from './pages/Trimmer';
import MultiSegmentConfig from './pages/MultiSegmentConfig';
import Sessions from './pages/Sessions';
import Dashboard from './pages/Dashboard';
// PlayerLibrary / PlayerProfile 是 100% mock data，没接后端，不暴露路由
// 直到真的有 player metadata pipeline 再上线。
import Login from './pages/Login';
import './index.css';

const pageVariants = {
    initial: { opacity: 0, y: 12 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.35, ease: 'easeOut' } },
    exit: { opacity: 0, y: -12, transition: { duration: 0.2 } },
};

function AnimatedRoutes() {
    const location = useLocation();
    return (
        <AnimatePresence mode="wait">
            <Routes location={location} key={location.key}>
                <Route path="/login" element={<PageWrap><Login /></PageWrap>} />
                <Route path="/" element={<PageWrap><Welcome /></PageWrap>} />
                <Route path="/upload" element={<PageWrap><Upload /></PageWrap>} />
                <Route path="/trim" element={<PageWrap><Trimmer /></PageWrap>} />
                {/* /configure (legacy single-pick) redirects to multi-segment */}
                <Route path="/configure" element={<PageWrap><MultiSegmentConfig /></PageWrap>} />
                <Route path="/configure-multi" element={<PageWrap><MultiSegmentConfig /></PageWrap>} />
                <Route path="/sessions" element={<PageWrap><Sessions /></PageWrap>} />
                <Route path="/dashboard" element={<PageWrap><Dashboard /></PageWrap>} />
                {/* /players 路由暂时下线（mock data，未对接后端） */}
            </Routes>
        </AnimatePresence>
    );
}

function PageWrap({ children }) {
    return (
        <motion.div
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
        >
            {children}
        </motion.div>
    );
}

export default function App() {
    return (
        <Router>
            <ProgressProvider>
                <AnimatedRoutes />
            </ProgressProvider>
        </Router>
    );
}
