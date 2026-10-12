import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Toaster } from 'react-hot-toast';
import ErrorBoundary from './components/ErrorBoundary';
import App from './App.jsx';
import './index.css';

// Auto-reload when dynamic imports fail due to new version deployment
if (typeof window !== 'undefined') {
    window.addEventListener('vite:preloadError', () => {
        const key = 'chunk_reload_attempt';
        const lastAttempt = parseInt(sessionStorage.getItem(key) || '0', 10);
        const now = Date.now();
        if (now - lastAttempt > 15000) {
            sessionStorage.setItem(key, String(now));
            window.location.reload();
        }
    });
}

createRoot(document.getElementById('root')).render(
    <StrictMode>
        <ErrorBoundary>
            <App />
            <Toaster
                position="top-right"
                toastOptions={{
                    duration: 4500,
                    style: {
                        background: '#111827',
                        color: '#e5e7eb',
                        border: '1px solid #334155',
                        borderRadius: 10,
                    },
                    success: { iconTheme: { primary: '#22c55e', secondary: '#0b0f16' } },
                    error:   { iconTheme: { primary: '#ef4444', secondary: '#0b0f16' } },
                }}
            />
        </ErrorBoundary>
    </StrictMode>,
);
