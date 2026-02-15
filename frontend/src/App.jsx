import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { ProgressProvider } from './components/ProgressBar';
import Welcome from './pages/Welcome';
import Upload from './pages/Upload';
import Trimmer from './pages/Trimmer';
import Configuration from './pages/Configuration';
import Dashboard from './pages/Dashboard';
import PlayerLibrary from './pages/PlayerLibrary';
import PlayerProfile from './pages/PlayerProfile';
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
      <Routes location={location} key={location.pathname}>
        <Route path="/login" element={<PageWrap><Login /></PageWrap>} />
        <Route path="/" element={<PageWrap><Welcome /></PageWrap>} />
        <Route path="/upload" element={<PageWrap><Upload /></PageWrap>} />
        <Route path="/trim" element={<PageWrap><Trimmer /></PageWrap>} />
        <Route path="/configure" element={<PageWrap><Configuration /></PageWrap>} />
        <Route path="/dashboard" element={<PageWrap><Dashboard /></PageWrap>} />
        <Route path="/players" element={<PageWrap><PlayerLibrary /></PageWrap>} />
        <Route path="/players/:id" element={<PageWrap><PlayerProfile /></PageWrap>} />
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

function App() {
  return (
    <Router>
      <ProgressProvider>
        <AnimatedRoutes />
      </ProgressProvider>
    </Router>
  );
}

export default App;
