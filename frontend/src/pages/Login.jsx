import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { HiEnvelope, HiLockClosed, HiEye, HiEyeSlash } from 'react-icons/hi2';
import { IoFootball } from 'react-icons/io5';
import toast from 'react-hot-toast';
import { supabase } from '../lib/supabase';
import './Login.css';

export default function Login() {
    const [isSignUp, setIsSignUp] = useState(false);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [name, setName] = useState('');
    const [showPwd, setShowPwd] = useState(false);
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            if (isSignUp) {
                const { error } = await supabase.auth.signUp({
                    email,
                    password,
                    options: { data: { full_name: name } }
                });
                if (error) throw error;
                toast.success('Check your email for the confirmation link!');
            } else {
                const { error } = await supabase.auth.signInWithPassword({
                    email,
                    password
                });
                if (error) throw error;
                toast.success('Welcome back!');
                navigate('/');
            }
        } catch (error) {
            toast.error(error.message || 'Authentication failed');
        } finally {
            setLoading(false);
        }
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

                    {/* "Forgot password" 暂时下线 — 还没接 reset password 流程，
                        留个死链让用户点了什么都不会发生反而更糟。等接好
                        supabase.auth.resetPasswordForEmail() 再放出来。 */}

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

                {/* "Continue with Google" 暂时下线 — supabase.auth.signInWithOAuth
                    还没在 dashboard 配 Google provider，按钮点了无反应给用户错觉。
                    等 OAuth 配好再放出来。 */}

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
