import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { HiEnvelope, HiLockClosed, HiEye, HiEyeSlash } from 'react-icons/hi2';
import { IoFootball } from 'react-icons/io5';
import './Login.css';

export default function Login() {
    const [isSignUp, setIsSignUp] = useState(false);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [name, setName] = useState('');
    const [showPwd, setShowPwd] = useState(false);
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    const handleSubmit = (e) => {
        e.preventDefault();
        setLoading(true);
        setTimeout(() => {
            setLoading(false);
            navigate('/');
        }, 1200);
    };

    return (
        <div className="login-page">
            <div className="bg-grid" />

            {/* Decorative orbs */}
            <motion.div
                className="login-orb login-orb--1"
                animate={{ y: [0, -20, 0], x: [0, 10, 0] }}
                transition={{ duration: 7, repeat: Infinity, ease: 'easeInOut' }}
            />
            <motion.div
                className="login-orb login-orb--2"
                animate={{ y: [0, 15, 0], x: [0, -12, 0] }}
                transition={{ duration: 9, repeat: Infinity, ease: 'easeInOut' }}
            />

            <motion.div
                className="login-card"
                initial={{ opacity: 0, y: 30, scale: 0.96 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.5, ease: 'easeOut' }}
            >
                {/* Logo */}
                <div className="login-card__logo">
                    <IoFootball className="login-card__logo-icon" />
                    <span>PitchLogic AI</span>
                </div>

                <h1 className="login-card__title">
                    {isSignUp ? 'Create Account' : 'Welcome Back'}
                </h1>
                <p className="login-card__subtitle">
                    {isSignUp
                        ? 'Start analyzing football performance'
                        : 'Sign in to continue to your dashboard'}
                </p>

                <form className="login-form" onSubmit={handleSubmit}>
                    {isSignUp && (
                        <motion.div
                            className="login-field"
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                        >
                            <label>Full Name</label>
                            <div className="login-input-wrap">
                                <HiEnvelope className="login-input-icon" />
                                <input
                                    type="text"
                                    placeholder="Your name"
                                    value={name}
                                    onChange={(e) => setName(e.target.value)}
                                    required
                                />
                            </div>
                        </motion.div>
                    )}

                    <div className="login-field">
                        <label>Email</label>
                        <div className="login-input-wrap">
                            <HiEnvelope className="login-input-icon" />
                            <input
                                type="email"
                                placeholder="you@example.com"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                            />
                        </div>
                    </div>

                    <div className="login-field">
                        <label>Password</label>
                        <div className="login-input-wrap">
                            <HiLockClosed className="login-input-icon" />
                            <input
                                type={showPwd ? 'text' : 'password'}
                                placeholder="••••••••"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                            />
                            <button
                                type="button"
                                className="login-pwd-toggle"
                                onClick={() => setShowPwd(!showPwd)}
                            >
                                {showPwd ? <HiEyeSlash /> : <HiEye />}
                            </button>
                        </div>
                    </div>

                    {!isSignUp && (
                        <div className="login-forgot">
                            <a href="#">Forgot password?</a>
                        </div>
                    )}

                    <button
                        type="submit"
                        className={`btn btn-primary btn-lg login-submit ${loading ? 'btn--loading' : ''}`}
                        disabled={loading}
                    >
                        {loading ? (
                            <>
                                <span className="config-spinner" />
                                {isSignUp ? 'Creating...' : 'Signing in...'}
                            </>
                        ) : (
                            isSignUp ? 'Create Account' : 'Sign In'
                        )}
                    </button>
                </form>

                <div className="login-divider">
                    <span>or</span>
                </div>

                <button className="btn btn-secondary login-google">
                    <svg viewBox="0 0 24 24" width="18" height="18">
                        <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 01-2.2 3.32v2.76h3.56c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" />
                        <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.56-2.76c-.98.66-2.23 1.06-3.72 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
                        <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05" />
                        <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
                    </svg>
                    Continue with Google
                </button>

                <p className="login-toggle">
                    {isSignUp ? 'Already have an account?' : "Don't have an account?"}{' '}
                    <button type="button" onClick={() => setIsSignUp(!isSignUp)}>
                        {isSignUp ? 'Sign in' : 'Sign up'}
                    </button>
                </p>
            </motion.div>
        </div>
    );
}
