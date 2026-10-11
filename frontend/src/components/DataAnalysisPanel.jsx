import React from 'react';
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, PieChart, Pie
} from 'recharts';
import {
    HiBolt,
    HiArrowTrendingUp,
    HiArrowsUpDown,
    HiGlobeAlt,
    HiArrowPath,
    HiForward,
    HiVideoCamera,
    HiIdentification,
    HiExclamationTriangle,
    HiInformationCircle
} from 'react-icons/hi2';
import { useLanguage } from '../i18n/LanguageContext';

const StatRow = ({ icon, label, value, sub }) => (
    <div className="stat-row">
        <div className="stat-row__icon">{icon}</div>
        <div className="stat-row__main">
            <div className="stat-row__label">{label}</div>
            <div className="stat-row__value">
                {value}
                {sub && (
                    <span
                        className="stat-row__sub"
                        style={{
                            marginLeft: '6px',
                            fontSize: '0.72rem',
                            color: '#f59e0b',
                            background: 'rgba(245, 158, 11, 0.15)',
                            padding: '2px 5px',
                            borderRadius: '4px',
                            fontWeight: 600,
                            verticalAlign: 'middle',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '3px'
                        }}
                    >
                        <HiExclamationTriangle style={{ fontSize: '0.8rem' }} /> {sub}
                    </span>
                )}
            </div>
        </div>
    </div>
);

const numberOrNull = (value) => {
    if (value === undefined || value === null || value === '') return null;
    const n = Number(value);
    return isNaN(n) ? null : n;
};

const formatMetric = (value, suffix = '', decimals = 1) => {
    const n = numberOrNull(value);
    if (n === null) return '-';
    return n.toFixed(decimals) + suffix;
};

const ChartTooltip = ({ active, payload, label, suffix = '', decimals = 1 }) => {
    if (!active || !payload || !payload.length) return null;
    return (
        <div className="chart-tooltip">
            <div className="chart-tooltip__label">{label}</div>
            {payload.map((entry, i) => {
                const val = (typeof entry.value === 'number')
                    ? entry.value.toFixed(decimals) + suffix
                    : entry.value;
                return (
                    <div key={i} className="chart-tooltip__row">
                        <span className="chart-tooltip__dot" style={{ background: entry.color || entry.fill }} />
                        {entry.name}
                        <strong>{val}</strong>
                    </div>
                );
            })}
        </div>
    );
};

const PossessionTooltip = ({ active, payload }) => {
    if (!active || !payload || !payload.length) return null;
    const item = payload[0];
    return (
        <div className="chart-tooltip" style={{ minWidth: 100 }}>
            <div className="chart-tooltip__row">
                <span className="chart-tooltip__dot" style={{ background: item.payload.fill }} />
                {item.name}
                <strong>{Number(item.value).toFixed(1)}%</strong>
            </div>
        </div>
    );
};

const PossessionBar = ({ team1, team2, neutral, t1Color, t2Color, neutralLabel }) => {
    const t1 = Math.max(0, Math.min(100, team1 ?? 0));
    const t2 = Math.max(0, Math.min(100, team2 ?? 0));
    const neu = Math.max(0, Math.min(100, neutral ?? Math.max(0, 100 - t1 - t2)));

    return (
        <div className="poss-bar">
            <div className="poss-bar__labels">
                <span>{t1.toFixed(1)}%</span>
                {neu > 0 && <span style={{ color: '#94a3b8', fontSize: '0.74rem' }}>{neutralLabel || 'Neutral'} {neu.toFixed(1)}%</span>}
                <span style={{ textAlign: 'right' }}>{t2.toFixed(1)}%</span>
            </div>
            <div className="poss-bar__track">
                <div className="poss-bar__fill poss-bar__fill--t1" style={{ width: `${t1}%`, background: t1Color || undefined }} />
                {neu > 0 && <div className="poss-bar__fill poss-bar__fill--neutral" style={{ width: `${neu}%` }} />}
                <div className="poss-bar__fill poss-bar__fill--t2" style={{ width: `${t2}%`, background: t2Color || undefined }} />
            </div>
        </div>
    );
};

export default function DataAnalysisPanel({ playerSummary }) {
    const { t, language } = useLanguage();

    if (!playerSummary) {
        return <p className="drawer__empty">{t('analytics.waitingStats')}</p>;
    }
    const overall = playerSummary.overall || playerSummary;
    const segments = playerSummary.by_segment || [];

    const t1 = Number(overall.team1_possession_pct ?? 0);
    const t2 = Number(overall.team2_possession_pct ?? 0);
    const neutral = Number(overall.neutral_possession_pct ?? Math.max(0, 100 - t1 - t2));
    const teamColors = overall.team_colors_hex || {};
    const t1Color = teamColors['1'] || '#3498db';
    const t2Color = teamColors['2'] || '#e74c3c';

    const possessionData = [
        { name: t('analytics.team1'), value: t1, fill: t1Color },
        { name: t('analytics.team2'), value: t2, fill: t2Color },
        ...(neutral > 0 ? [{ name: t('analytics.neutral'), value: neutral, fill: '#94a3b8' }] : []),
    ].filter((item) => item.value > 0);

    const speedData = [
        { name: language === 'zh' ? '平均速度' : 'Avg', value: numberOrNull(overall.avg_speed_kmh) ?? 0, fill: '#60a5fa' },
        { name: language === 'zh' ? '最高速度' : 'Max', value: numberOrNull(overall.max_speed_kmh) ?? 0, fill: '#f59e0b' },
    ];

    const periodData = segments.map((seg, i) => ({
        name: (seg.segment_type || `${language === 'zh' ? '片段' : 'Seg'} ${i + 1}`).replace('_', ' '),
        distance: numberOrNull(seg.total_distance_m) ?? 0,
        avg: numberOrNull(seg.avg_speed_kmh),
        max: numberOrNull(seg.max_speed_kmh),
    }));
    const isSprintsBlocked = overall.sprint_count === null || overall.analytics_contract?.sprints_count === 'EXPLICITLY_UNAVAILABLE';
    const isDistanceBlocked = overall.total_distance_m === null || overall.analytics_contract?.total_distance_m === 'EXPLICITLY_UNAVAILABLE';
    const speedFlag = overall.speed_reliability === 'suspect' || Number(overall.max_speed_kmh) >= 37.5;
    const speedSub = overall.max_speed_kmh === null && overall.max_speed_unavailable_reason
        ? (language === 'zh' ? '已屏蔽' : 'Disabled')
        : (speedFlag ? (language === 'zh' ? '待校验' : 'Verify') : null);
    const sprintsSub = isSprintsBlocked && overall.sprints_unavailable_reason ? (language === 'zh' ? '已停用' : 'Disabled') : null;

    // Detect truncated tracking / distance anomaly
    const durationSec = Number(overall.duration_s || (segments[0]?.end_sec ? (segments[segments.length - 1].end_sec - segments[0].start_sec) : 0));
    const totalDist = Number(overall.total_distance_m ?? 0);
    const trackingCov = overall.tracking_coverage_pct !== undefined ? Number(overall.tracking_coverage_pct) : null;
    const distanceRate = durationSec > 0 ? (totalDist / (durationSec / 60.0)) : 999;
    const isDistanceTruncated = !isDistanceBlocked && (
        Boolean(overall.distance_unavailable_reason) ||
        (durationSec >= 180 && totalDist > 0 && totalDist < 200 && (distanceRate < 20.0 || durationSec >= 600))
    );
    const distanceSub = isDistanceBlocked && overall.distance_unavailable_reason
        ? (language === 'zh' ? '已停用' : 'Disabled')
        : (isDistanceTruncated ? (trackingCov !== null && trackingCov < 25 ? (language === 'zh' ? '片段不完整' : 'Partial') : (language === 'zh' ? '数值偏低' : 'Low')) : null);

    // Detect sparse ball possession sample (< 5% of duration, or neutral >= 90% with low sample)
    const possSec = Number(overall.possession_seconds ?? 0);
    const isPossessionSparse = Boolean(overall.possession_unavailable_reason) ||
        (durationSec >= 10 && possSec > 0 && (possSec / durationSec) < 0.05) ||
        (neutral >= 90.0 && possSec < 15.0 && durationSec >= 10);

    return (
        <div className="drawer__section-body">
            {overall.video_confidence && (
                <div style={{
                    marginBottom: '12px',
                    padding: '8px 12px',
                    borderRadius: '6px',
                    background: overall.footage_quality_tier === 'TIER_5_SEVERELY_DEGRADED' ? 'rgba(239, 68, 68, 0.12)' : 'rgba(59, 130, 246, 0.10)',
                    border: `1px solid ${overall.footage_quality_tier === 'TIER_5_SEVERELY_DEGRADED' ? 'rgba(239, 68, 68, 0.3)' : 'rgba(59, 130, 246, 0.25)'}`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    fontSize: '0.8rem',
                }}>
                    <span style={{ fontWeight: 600, color: overall.footage_quality_tier === 'TIER_5_SEVERELY_DEGRADED' ? '#f87171' : '#93c5fd', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <HiVideoCamera /> {language === 'zh' ? (overall.video_confidence.tier_label_zh || overall.footage_quality_tier) : (overall.video_confidence.tier_label_en || overall.footage_quality_tier)}
                    </span>
                    <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                        {t('analytics.calibConfidence')}: {overall.video_confidence.overall_confidence_score?.toFixed(0)}{t('analytics.points')}
                    </span>
                </div>
            )}
            {overall.jersey_number !== undefined && overall.jersey_number !== null && (
                <div style={{
                    marginBottom: '12px',
                    padding: '8px 12px',
                    borderRadius: '6px',
                    background: 'rgba(16, 185, 129, 0.12)',
                    border: '1px solid rgba(16, 185, 129, 0.3)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    fontSize: '0.8rem',
                }}>
                    <span style={{ fontWeight: 600, color: '#34d399', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <HiIdentification /> {t('analytics.jerseyNumber')}: #{overall.jersey_number}
                    </span>
                    <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                        {t('analytics.confidence')}: {overall.jersey_confidence ? `${Math.round(overall.jersey_confidence * 100)}%` : t('analytics.confirmed')}
                    </span>
                </div>
            )}
            <div className="stat-grid">
                <StatRow icon={<HiBolt />} label={t('analytics.maxSpeed')} value={formatMetric(overall.max_speed_kmh, ' km/h')} sub={speedSub} />
                <StatRow icon={<HiArrowTrendingUp />} label={t('analytics.avgSpeed')} value={formatMetric(overall.avg_speed_kmh, ' km/h')} />
                <StatRow icon={<HiArrowsUpDown />} label={t('analytics.distance')} value={isDistanceBlocked ? '-' : formatMetric(overall.total_distance_m, ' m', 0)} sub={distanceSub} />
                <StatRow icon={<HiGlobeAlt />} label={t('analytics.possessionSec')} value={formatMetric(overall.possession_seconds, ' s')} />
                <StatRow icon={<HiArrowPath />} label={t('analytics.switches')} value={overall.possession_switches ?? '-'} />
                <StatRow icon={<HiForward />} label={t('analytics.sprints')} value={isSprintsBlocked ? '-' : (overall.sprint_count ?? overall.speed_telemetry?.sprint_count ?? '-')} sub={sprintsSub} />
            </div>
            {overall.max_speed_unavailable_reason && (
                <p className="drawer__note" style={{ color: '#f87171', borderLeft: '3px solid #ef4444', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <HiInformationCircle /> {overall.max_speed_unavailable_reason}
                </p>
            )}
            {overall.sprints_unavailable_reason && (
                <p className="drawer__note" style={{ color: '#f87171', borderLeft: '3px solid #ef4444', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <HiInformationCircle /> {overall.sprints_unavailable_reason}
                </p>
            )}
            {overall.distance_unavailable_reason && (
                <p className="drawer__note" style={{ color: '#f87171', borderLeft: '3px solid #ef4444', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <HiInformationCircle /> {overall.distance_unavailable_reason}
                </p>
            )}
            {isDistanceTruncated && !overall.distance_unavailable_reason && (
                <p className="drawer__note" style={{ color: '#fbbf24', borderLeft: '3px solid #f59e0b', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <HiInformationCircle /> {language === 'zh' ? `全场跑动数值偏低（当前仅记录到 ${totalDist}m），可能受低活动量或局部中断影响，已降级标注。` : `Low recorded distance (${totalDist}m), potentially due to low activity or occlusions.`}
                </p>
            )}
            {speedFlag && !overall.max_speed_unavailable_reason && (
                <p className="drawer__note" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <HiInformationCircle /> {language === 'zh' ? `峰值速度 (${overall.max_speed_kmh} km/h) 可能受镜头位移或追踪噪点影响，可靠度已降级标注。` : `Peak speed (${overall.max_speed_kmh} km/h) is flagged as likely tracking/camera noise.`}
                </p>
            )}

            <h4 className="drawer__subhead">{t('analytics.speedKmh')}</h4>
            <div className="chart-wrap" style={{ height: 140 }}>
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={speedData} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                        <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
                        <YAxis stroke="#94a3b8" fontSize={11} />
                        <Tooltip content={(props) => <ChartTooltip {...props} suffix=" km/h" />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
                        <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                            {speedData.map((entry) => (
                                <Cell key={entry.name} fill={entry.fill} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>

            <h4 className="drawer__subhead">{t('analytics.teamPossession')}</h4>
            {isPossessionSparse && (
                <p className="drawer__note" style={{ color: '#fbbf24', borderLeft: '3px solid #f59e0b', margin: '4px 0 10px 0', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <HiExclamationTriangle /> {overall.possession_unavailable_reason || (language === 'zh' ? `有效控球样本偏低（记录时长仅 ${possSec.toFixed(1)}s，占比赛 ${durationSec > 0 ? ((possSec / durationSec) * 100).toFixed(1) : '<1'}%），控球比例已降级提示，仅反映局部有效片段。` : `Low possession samples recorded (${possSec.toFixed(1)}s, ${durationSec > 0 ? ((possSec / durationSec) * 100).toFixed(1) : '<1'}% of match duration).`)}
                </p>
            )}
            {overall.pass_events_unavailable_reason && (
                <p className="drawer__note" style={{ color: '#fbbf24', borderLeft: '3px solid #f59e0b', margin: '4px 0 10px 0', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <HiInformationCircle /> {overall.pass_events_unavailable_reason}
                </p>
            )}
            <div className="poss-row">
                <div className="chart-wrap chart-wrap--donut">
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={possessionData}
                                innerRadius={32}
                                outerRadius={56}
                                paddingAngle={2}
                                dataKey="value"
                                stroke="none"
                            >
                                {possessionData.map((entry, i) => (
                                    <Cell key={i} fill={entry.fill} />
                                ))}
                            </Pie>
                            <Tooltip content={(props) => <PossessionTooltip {...props} />} />
                        </PieChart>
                    </ResponsiveContainer>
                </div>
                <div className="poss-row__legend">
                    <div><span className="poss-dot poss-dot--t1" style={t1Color ? { background: t1Color } : {}} /> {t('analytics.team1')} <strong>{t1.toFixed(1)}%</strong></div>
                    <div><span className="poss-dot poss-dot--t2" style={t2Color ? { background: t2Color } : {}} /> {t('analytics.team2')} <strong>{t2.toFixed(1)}%</strong></div>
                    {neutral > 0 && <div><span className="poss-dot poss-dot--neutral" /> {t('analytics.neutral')} <strong>{neutral.toFixed(1)}%</strong></div>}
                </div>
            </div>
            <PossessionBar team1={t1} team2={t2} neutral={neutral} t1Color={t1Color} t2Color={t2Color} neutralLabel={t('analytics.neutral')} />

            {periodData.length > 0 && (
                <>
                    <h4 className="drawer__subhead">{t('analytics.byPeriodDistance')}</h4>
                    <div className="chart-wrap" style={{ height: 140 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={periodData} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                                <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} />
                                <YAxis stroke="#94a3b8" fontSize={11} />
                                <Tooltip content={(props) => <ChartTooltip {...props} suffix=" m" decimals={0} />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
                                <Bar dataKey="distance" fill="#22d3ee" radius={[6, 6, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                    <table className="seg-table">
                        <thead>
                            <tr><th>{t('analytics.period')}</th><th>{t('analytics.distance')}</th><th>{t('analytics.avgSpeed')}</th><th>{t('analytics.maxSpeed')}</th></tr>
                        </thead>
                        <tbody>
                            {periodData.map((seg, i) => (
                                <tr key={i}>
                                    <td>{seg.name}</td>
                                    <td>{formatMetric(seg.distance, ' m', 0)}</td>
                                    <td>{formatMetric(seg.avg, ' km/h')}</td>
                                    <td>{formatMetric(seg.max, ' km/h')}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </>
            )}
        </div>
    );
}
