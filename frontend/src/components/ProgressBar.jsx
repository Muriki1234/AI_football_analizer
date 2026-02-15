import { useState, useEffect, useCallback, createContext, useContext } from 'react';
import { useLocation } from 'react-router-dom';

const ProgressContext = createContext();

export function useProgress() {
    return useContext(ProgressContext);
}

export function ProgressProvider({ children }) {
    const [progress, setProgress] = useState(0);
    const [visible, setVisible] = useState(false);
    const location = useLocation();

    const start = useCallback(() => {
        setVisible(true);
        setProgress(15);
        const t1 = setTimeout(() => setProgress(45), 200);
        const t2 = setTimeout(() => setProgress(70), 600);
        return () => { clearTimeout(t1); clearTimeout(t2); };
    }, []);

    const done = useCallback(() => {
        setProgress(100);
        setTimeout(() => {
            setVisible(false);
            setProgress(0);
        }, 400);
    }, []);

    // Auto-trigger on route change
    useEffect(() => {
        start();
        const timeout = setTimeout(done, 800);
        return () => clearTimeout(timeout);
    }, [location.pathname, start, done]);

    return (
        <ProgressContext.Provider value={{ start, done }}>
            {visible && (
                <div className="global-progress">
                    <div
                        className="global-progress__bar"
                        style={{ width: `${progress}%` }}
                    />
                </div>
            )}
            {children}
        </ProgressContext.Provider>
    );
}
