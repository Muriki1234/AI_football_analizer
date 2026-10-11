import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import {
    HiHome, HiMagnifyingGlass, HiClock,
    HiCheckCircle, HiExclamationTriangle, HiArrowPath, HiTrash,
} from 'react-icons/hi2';
import { listMySessions, deleteSession } from '../services/api';
import { useLanguage } from '../i18n/LanguageContext';
import './Sessions.css';

const STATUS_ICONS = {
    uploaded:         { icon: HiClock,                color: '#94a3b8' },
    queued:           { icon: HiClock,                color: '#94a3b8' },
    tracking:         { icon: HiArrowPath,            color: '#fbbf24' },
    tracking_done:    { icon: HiArrowPath,            color: '#fbbf24' },
    analyzing:        { icon: HiArrowPath,            color: '#60a5fa' },
    analysis_done:    { icon: HiCheckCircle,          color: '#4ade80' },
    analysis_failed:  { icon: HiExclamationTriangle,  color: '#f87171' },
    tracking_failed:  { icon: HiExclamationTriangle,  color: '#f87171' },
};

function resolveMeta(s, t) {
    const iconMeta = STATUS_ICONS[s.status] || STATUS_ICONS.uploaded;
    let label = t('sessions.statusUploaded');

    switch (s.status) {
        case 'queued':
            label = t('sessions.statusQueued');
            break;
        case 'tracking':
            label = t('sessions.statusTracking');
            break;
        case 'tracking_done':
            label = t('sessions.statusTrackingDone');
            break;
        case 'analyzing':
            label = t('sessions.statusAnalyzing');
            break;
        case 'analysis_done':
            label = t('sessions.statusDone');
            break;
        case 'analysis_failed':
        case 'tracking_failed':
            label = t('sessions.statusFailed');
            break;
        default:
            label = t('sessions.statusUploaded');
    }

    // Check for zombies (dead workers stuck in running state)
    if (['tracking', 'samurai_multi_pending', 'samurai_done', 'analyzing', 'queued'].includes(s.status)) {
        const lastUpdated = new Date(s.updated_at || s.created_at).getTime();
        const minsSinceUpdate = (Date.now() - lastUpdated) / 60000;
        if (minsSinceUpdate > 20) {
            return { icon: HiExclamationTriangle, color: '#f87171', label: t('sessions.statusTimeout') };
        }
    }

    if (s.status !== 'uploaded') {
        return { ...iconMeta, label };
    }

    const periods = Array.isArray(s.match_periods_sec) ? s.match_periods_sec : null;
    if (!periods || periods.length === 0) {
        return { ...iconMeta, label: t('sessions.statusSetPeriods'), color: '#a78bfa' };
    }
    return { ...iconMeta, label: t('sessions.statusPickPlayers'), color: '#22d3ee' };
}

const formatRelative = (ts, language) => {
    const d = new Date(ts).getTime();
    if (!Number.isFinite(d)) return '';
    const diff = Math.max(0, Date.now() - d);
    const m = Math.floor(diff / 60000);
    if (m < 1) return language === 'zh' ? '刚刚' : 'just now';
    if (m < 60) return language === 'zh' ? `${m}分钟前` : `${m}m ago`;
    const h = Math.floor(m / 60);
    if (h < 24) return language === 'zh' ? `${h}小时前` : `${h}h ago`;
    const days = Math.floor(h / 24);
    if (days < 30) return language === 'zh' ? `${days}天前` : `${days}d ago`;
    return new Date(ts).toLocaleDateString();
};

export default function Sessions() {
    const navigate = useNavigate();
    const { t, language } = useLanguage();
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

    const open = (s) => {
        const id = s.id;
        const periods = Array.isArray(s.match_periods_sec) ? s.match_periods_sec : null;
        if (s.status === 'uploaded') {
            const dest = periods && periods.length > 0 ? `/configure-multi?sessionId=${encodeURIComponent(id)}` : `/trim?sessionId=${encodeURIComponent(id)}`;
            navigate(dest, { state: { sessionId: id, videoId: id, matchPeriods: periods } });
            return;
        }
        navigate(`/dashboard?sessionId=${encodeURIComponent(id)}`, {
            state: { sessionId: id, videoId: id },
        });
    };

    const [deleting, setDeleting] = useState(null);

    const handleDelete = async (e, session) => {
        e.stopPropagation();
        const confirmMsg = language === 'zh'
            ? `确定删除比赛 "${session.fileName}" 吗？\n\n此操作将删除该比赛的任务档案与分析记录。此操作无法撤销。`
            : `Delete "${session.fileName}"?\n\nThis removes the session record and all its analysis tasks. This cannot be undone.`;
        
        const confirmed = window.confirm(confirmMsg);
        if (!confirmed) return;
        setDeleting(session.id);
        try {
            await deleteSession(session.id);
            setSessions((prev) => prev.filter((s) => s.id !== session.id));
            toast.success(t('sessions.deleteSuccess', { name: session.fileName }));
        } catch (err) {
            toast.error(t('sessions.deleteFailed', { error: err.message }));
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
                    <HiHome /> {language === 'zh' ? '主页' : 'Home'}
                </button>
                <h1 className="sessions-page__title">{t('sessions.title')}</h1>
                <button className="btn btn-primary" onClick={() => navigate('/upload')}>
                    + {language === 'zh' ? '上传新比赛' : 'New Upload'}
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
                        placeholder={t('sessions.searchPlaceholder')}
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                    />
                </div>
                <div className="sessions-page__status-tabs">
                    {[
                        { v: 'all',            label: t('sessions.tabAll') },
                        { v: 'analysis_done',  label: t('sessions.tabCompleted') },
                        { v: 'analyzing',      label: t('sessions.tabInProgress') },
                        { v: 'failed',         label: t('sessions.tabFailed') },
                    ].map((tItem) => (
                        <button
                            key={tItem.v}
                            className={`sessions-page__tab ${statusFilter === tItem.v ? 'is-active' : ''}`}
                            onClick={() => setStatusFilter(tItem.v)}
                        >
                            {tItem.label}
                        </button>
                    ))}
                </div>
            </motion.div>

            {error && (
                <p className="sessions-page__error">
                    <HiExclamationTriangle style={{ display: 'inline', marginRight: '6px' }} />
                    {error}
                </p>
            )}

            {loading ? (
                <p className="sessions-page__empty">{t('common.loading')}</p>
            ) : filtered.length === 0 ? (
                <p className="sessions-page__empty">
                    {search || statusFilter !== 'all'
                        ? t('sessions.emptyMatches')
                        : (language === 'zh' ? '暂无比赛分析记录 — 上传第一场比赛。' : 'No matches yet — upload your first video.')}
                </p>
            ) : (
                <motion.div
                    className="sessions-page__list"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.15 }}
                >
                    {filtered.map((s, i) => {
                        const meta = resolveMeta(s, t);
                        const Icon = meta.icon;
                        return (
                            <motion.button
                                key={s.id}
                                type="button"
                                className="sessions-page__row"
                                onClick={() => open(s)}
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
                                        <span>{formatRelative(s.created_at, language)}</span>
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
                                    aria-label={t('sessions.deleteSession')}
                                    className={`sessions-page__row-delete ${deleting === s.id ? 'is-deleting' : ''}`}
                                    onClick={(e) => handleDelete(e, s)}
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter' || e.key === ' ') {
                                            e.preventDefault();
                                            handleDelete(e, s);
                                        }
                                    }}
                                    title={t('sessions.deleteSession')}
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
