import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import {
    HiHome, HiMagnifyingGlass, HiClock,
    HiCheckCircle, HiExclamationTriangle, HiArrowPath, HiTrash,
} from 'react-icons/hi2';
import { listMySessions, deleteSession } from '../services/api';
import './Sessions.css';

const STATUS_META = {
    uploaded:         { label: 'Uploaded',     icon: HiClock,                color: '#94a3b8' },
    queued:           { label: 'Queued',       icon: HiClock,                color: '#94a3b8' },
    tracking:         { label: 'Tracking',     icon: HiArrowPath,            color: '#fbbf24' },
    tracking_done:    { label: 'Tracking ✓',   icon: HiArrowPath,            color: '#fbbf24' },
    analyzing:        { label: 'Analyzing',    icon: HiArrowPath,            color: '#60a5fa' },
    analysis_done:    { label: 'Done',         icon: HiCheckCircle,          color: '#4ade80' },
    analysis_failed:  { label: 'Failed',       icon: HiExclamationTriangle,  color: '#f87171' },
    tracking_failed:  { label: 'Failed',       icon: HiExclamationTriangle,  color: '#f87171' },
};

// 后端只有一个 'uploaded' 状态，但用户能停在三个不同的页面：
//   - 没设过半场时间 → "Set periods"   (Trim 页)
//   - 设了半场没追踪人 → "Pick players" (MultiSegmentConfig 页)
//   - 其他 / 老数据    → "Uploaded"
function resolveMeta(s) {
    const base = STATUS_META[s.status] || STATUS_META.uploaded;
    if (s.status !== 'uploaded') return base;
    const periods = Array.isArray(s.match_periods_sec) ? s.match_periods_sec : null;
    if (!periods || periods.length === 0) {
        return { ...base, label: 'Set periods', color: '#a78bfa' };
    }
    return { ...base, label: 'Pick players', color: '#22d3ee' };
}

const formatRelative = (ts) => {
    const d = new Date(ts).getTime();
    if (!Number.isFinite(d)) return '';
    const diff = Math.max(0, Date.now() - d);
    const m = Math.floor(diff / 60000);
    if (m < 1) return 'just now';
    if (m < 60) return `${m}m ago`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}h ago`;
    const days = Math.floor(h / 24);
    if (days < 30) return `${days}d ago`;
    return new Date(ts).toLocaleDateString();
};

export default function Sessions() {
    const navigate = useNavigate();
    const [sessions, setSessions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [search, setSearch] = useState('');
    const [statusFilter, setStatusFilter] = useState('all');
    const [error, setError] = useState(null);

    useEffect(() => {
        let cancelled = false;
        setLoading(true);
        listMySessions({ limit: 200 })
            .then((rows) => { if (!cancelled) setSessions(rows); })
            .catch((e) => { if (!cancelled) setError(e.message); })
            .finally(() => { if (!cancelled) setLoading(false); });
        return () => { cancelled = true; };
    }, []);

    const filtered = useMemo(() => {
        const q = search.trim().toLowerCase();
        // 'uploaded' 不算 in_progress —— resolveMeta() 已经把它分成
        // "Set periods" 和 "Pick players" 两种子状态，意义是「等用户操作」
        // 而不是「服务端在跑」。继续算 in_progress 会让 Running tab 里塞
        // 一堆等 user 点的 session，跟标签名不符。
        // 'uploading' 保留 —— 上传中是真在传字节。
        const IN_PROGRESS = new Set([
            'uploading', 'queued',
            'tracking', 'tracking_done', 'samurai_multi_pending', 'samurai_done',
            'analyzing',
        ]);

        const matchesStatus = (status) => {
            switch (statusFilter) {
                case 'all':
                    return true;
                case 'analysis_done':
                case 'done':
                    return status === 'analysis_done';
                case 'analyzing':
                case 'in_progress':
                    return IN_PROGRESS.has(status);
                case 'failed':
                    return typeof status === 'string' && status.endsWith('_failed');
                default:
                    return status === statusFilter;
            }
        };

        const matchesSearch = (s) => {
            if (!q) return true;
            return (
                (s.fileName || '').toLowerCase().includes(q) ||
                s.id.toLowerCase().includes(q)
            );
        };

        return sessions.filter((s) => matchesStatus(s.status) && matchesSearch(s));
    }, [sessions, search, statusFilter]);

    const open = (id) => navigate(`/dashboard?sessionId=${encodeURIComponent(id)}`, {
        state: { sessionId: id, videoId: id },
    });

    const [deleting, setDeleting] = useState(null);  // session id currently being deleted

    const handleDelete = async (e, session) => {
        e.stopPropagation();   // don't bubble up to the row click
        const confirmed = window.confirm(
            `Delete "${session.fileName}"?\n\n` +
            `This removes the session record and all its tasks. The uploaded ` +
            `video file stays in storage until the nightly cleanup runs.\n\n` +
            `This cannot be undone.`
        );
        if (!confirmed) return;
        setDeleting(session.id);
        try {
            await deleteSession(session.id);
            setSessions((prev) => prev.filter((s) => s.id !== session.id));
            toast.success(`Deleted ${session.fileName}`);
        } catch (err) {
            toast.error(`Delete failed: ${err.message}`);
        } finally {
            setDeleting(null);
        }
    };

    return (
        <div className="sessions-page">
            <div className="bg-grid" />

            <motion.div
                className="sessions-page__topbar"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
            >
                <button className="btn btn-ghost" onClick={() => navigate('/')}>
                    <HiHome /> Home
                </button>
                <h1 className="sessions-page__title">My Sessions</h1>
                <button className="btn btn-primary" onClick={() => navigate('/upload')}>
                    + New Upload
                </button>
            </motion.div>

            <motion.div
                className="sessions-page__filters"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.1 }}
            >
                <div className="sessions-page__search">
                    <HiMagnifyingGlass />
                    <input
                        type="text"
                        placeholder="Search by filename or session id…"
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                    />
                </div>
                <div className="sessions-page__status-tabs">
                    {[
                        { v: 'all',            label: 'All' },
                        { v: 'analysis_done',  label: 'Done' },
                        { v: 'analyzing',      label: 'Running' },
                        { v: 'failed',         label: 'Failed' },
                    ].map((t) => (
                        <button
                            key={t.v}
                            className={`sessions-page__tab ${statusFilter === t.v ? 'is-active' : ''}`}
                            onClick={() => setStatusFilter(t.v)}
                        >
                            {t.label}
                        </button>
                    ))}
                </div>
            </motion.div>

            {error && (
                <p className="sessions-page__error">⚠ {error}</p>
            )}

            {loading ? (
                <p className="sessions-page__empty">Loading…</p>
            ) : filtered.length === 0 ? (
                <p className="sessions-page__empty">
                    {search || statusFilter !== 'all'
                        ? 'No sessions match this filter.'
                        : 'No sessions yet — upload your first video.'}
                </p>
            ) : (
                <motion.div
                    className="sessions-page__list"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.15 }}
                >
                    {filtered.map((s, i) => {
                        const meta = resolveMeta(s);
                        const Icon = meta.icon;
                        return (
                            <motion.button
                                key={s.id}
                                type="button"
                                className="sessions-page__row"
                                onClick={() => open(s.id)}
                                initial={{ opacity: 0, y: 5 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.02 * i }}
                                whileHover={{ x: 3 }}
                            >
                                <div className="sessions-page__row-icon" style={{ color: meta.color }}>
                                    <Icon />
                                </div>
                                <div className="sessions-page__row-main">
                                    <div className="sessions-page__row-name" title={s.fileName}>
                                        {s.fileName}
                                    </div>
                                    <div className="sessions-page__row-meta">
                                        <span>{formatRelative(s.created_at)}</span>
                                        <span className="sessions-page__row-sep">·</span>
                                        <span className="sessions-page__row-id">{s.id.slice(0, 8)}…</span>
                                        {s.status === 'analyzing' && s.progress != null && (
                                            <>
                                                <span className="sessions-page__row-sep">·</span>
                                                <span>{s.progress}% — {s.stage || ''}</span>
                                            </>
                                        )}
                                    </div>
                                </div>
                                <span
                                    className="sessions-page__row-status"
                                    style={{ color: meta.color, borderColor: `${meta.color}55` }}
                                >
                                    {meta.label}
                                </span>
                                <span
                                    role="button"
                                    tabIndex={0}
                                    aria-label="Delete session"
                                    className={`sessions-page__row-delete ${deleting === s.id ? 'is-deleting' : ''}`}
                                    onClick={(e) => handleDelete(e, s)}
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter' || e.key === ' ') {
                                            e.preventDefault();
                                            handleDelete(e, s);
                                        }
                                    }}
                                    title="Delete session"
                                >
                                    <HiTrash />
                                </span>
                            </motion.button>
                        );
                    })}
                </motion.div>
            )}
        </div>
    );
}
