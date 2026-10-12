import { Component } from 'react';

/**
 * App-wide ErrorBoundary. Catches render-time crashes so one broken page
 * doesn't blank out the whole app. Network errors still need per-page try/catch
 * with toasts — they don't bubble to render.
 */
export default class ErrorBoundary extends Component {
    constructor(props) {
        super(props);
        this.state = { error: null };
    }

    static getDerivedStateFromError(error) {
        return { error };
    }

    componentDidCatch(error, info) {
        // eslint-disable-next-line no-console
        console.error('ErrorBoundary caught:', error, info?.componentStack);

        const isChunkLoadError = Boolean(
            error?.message?.includes('Failed to fetch dynamically imported module') ||
            error?.message?.includes('Importing a module script failed') ||
            error?.message?.includes('Loading chunk') ||
            error?.toString()?.includes('dynamically imported module')
        );

        if (isChunkLoadError && typeof window !== 'undefined') {
            const key = 'chunk_reload_attempt';
            const lastAttempt = parseInt(sessionStorage.getItem(key) || '0', 10);
            const now = Date.now();
            if (now - lastAttempt > 15000) {
                sessionStorage.setItem(key, String(now));
                window.location.reload();
            }
        }
    }

    reset = () => this.setState({ error: null });

    render() {
        if (!this.state.error) return this.props.children;

        const isChunkLoadError = Boolean(
            this.state.error?.message?.includes('Failed to fetch dynamically imported module') ||
            this.state.error?.message?.includes('Importing a module script failed') ||
            this.state.error?.message?.includes('Loading chunk') ||
            this.state.error?.toString()?.includes('dynamically imported module')
        );

        return (
            <div
                style={{
                    minHeight: '100vh',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                    alignItems: 'center',
                    padding: '2rem',
                    color: '#e5e7eb',
                    background: '#0b0f16',
                    fontFamily: 'system-ui, -apple-system, sans-serif',
                }}
            >
                <div
                    style={{
                        maxWidth: 520,
                        width: '100%',
                        padding: '2rem',
                        border: '1px solid #334155',
                        borderRadius: 12,
                        background: '#111827',
                        boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5)',
                    }}
                >
                    {isChunkLoadError ? (
                        <>
                            <h1 style={{ margin: 0, fontSize: '1.4rem', color: '#60a5fa' }}>
                                新版本已更新 / App Updated
                            </h1>
                            <p style={{ color: '#cbd5e1', marginTop: 10, lineHeight: 1.6, fontSize: '0.95rem' }}>
                                系统刚刚发布了最新优化版本，当前页面正在请求旧版组件资源。点击下方按钮即可加载最新版本。
                                <br />
                                <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
                                    A new version has been deployed. Please reload the page to load the latest resources.
                                </span>
                            </p>
                            <div style={{ display: 'flex', gap: 10, marginTop: 20 }}>
                                <button
                                    onClick={() => window.location.reload()}
                                    style={{
                                        padding: '0.6rem 1.25rem',
                                        borderRadius: 8,
                                        border: 'none',
                                        background: '#2563eb',
                                        color: 'white',
                                        fontWeight: 600,
                                        cursor: 'pointer',
                                    }}
                                >
                                    立即刷新 / Reload
                                </button>
                                <button
                                    onClick={() => (window.location.href = '/')}
                                    style={{
                                        padding: '0.6rem 1.25rem',
                                        borderRadius: 8,
                                        border: '1px solid #334155',
                                        background: 'transparent',
                                        color: '#cbd5e1',
                                        cursor: 'pointer',
                                    }}
                                >
                                    返回主页 / Home
                                </button>
                            </div>
                        </>
                    ) : (
                        <>
                            <h1 style={{ margin: 0, fontSize: '1.5rem' }}>Something broke.</h1>
                            <p style={{ color: '#94a3b8', marginTop: 8 }}>
                                The page hit an unexpected error. Details below — screenshot it
                                if you need to report it.
                            </p>
                            <pre
                                style={{
                                    whiteSpace: 'pre-wrap',
                                    fontSize: 12,
                                    background: '#0b1220',
                                    padding: '0.75rem',
                                    borderRadius: 8,
                                    border: '1px solid #1e293b',
                                    overflow: 'auto',
                                    maxHeight: 240,
                                }}
                            >
                                {String(this.state.error?.stack || this.state.error)}
                            </pre>
                            <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
                                <button
                                    onClick={this.reset}
                                    style={{
                                        padding: '0.5rem 1rem',
                                        borderRadius: 8,
                                        border: 'none',
                                        background: '#22c55e',
                                        color: 'white',
                                        cursor: 'pointer',
                                    }}
                                >
                                    Try again
                                </button>
                                <button
                                    onClick={() => (window.location.href = '/')}
                                    style={{
                                        padding: '0.5rem 1rem',
                                        borderRadius: 8,
                                        border: '1px solid #334155',
                                        background: 'transparent',
                                        color: '#cbd5e1',
                                        cursor: 'pointer',
                                    }}
                                >
                                    Home
                                </button>
                            </div>
                        </>
                    )}
                </div>
            </div>
        );
    }
}
