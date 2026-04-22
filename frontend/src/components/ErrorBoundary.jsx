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
    }

    reset = () => this.setState({ error: null });

    render() {
        if (!this.state.error) return this.props.children;
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
                        padding: '2rem',
                        border: '1px solid #334155',
                        borderRadius: 12,
                        background: '#111827',
                    }}
                >
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
                </div>
            </div>
        );
    }
}
