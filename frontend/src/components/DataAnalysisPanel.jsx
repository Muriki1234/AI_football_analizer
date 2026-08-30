import React from 'react';
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, PieChart, Pie
} from 'recharts';

const StatRow = ({ icon, label, value, sub }) => (
    <div className="stat-card">
        <div className="stat-card__icon">{icon}</div>
        <div className="stat-card__info">
            <span className="stat-card__label">{label}</span>
            <span className="stat-card__value">
                {value}
                {sub && <span className="stat-card__sub">{sub}</span>}
            </span>
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

const PossessionBar = ({ team1, team2, neutral, t1Color, t2Color }) => {
    const t1 = Math.max(0, Math.min(100, team1 ?? 0));
    const t2 = Math.max(0, Math.min(100, team2 ?? 0));
    const neu = Math.max(0, Math.min(100, neutral ?? Math.max(0, 100 - t1 - t2)));

    return (
        <div className="poss-bar">
            <div className="poss-bar__labels">
                <span>{t1.toFixed(1)}%</span>
                <span style={{ textAlign: 'right' }}>{t2.toFixed(1)}%</span>
            </div>
            <div className="poss-bar__track">
                <div className="poss-bar__fill poss-bar__fill--t1" style={{ width: `${t1}%`, background: t1Color || undefined }} />
                <div className="poss-bar__fill poss-bar__fill--t2" style={{ width: `${t2}%`, background: t2Color || undefined }} />
                {neu > 0 && <div className="poss-bar__fill poss-bar__fill--neutral" style={{ width: `${neu}%` }} />}
            </div>
        </div>
    );
};

export default function DataAnalysisPanel({ playerSummary }) {
    if (!playerSummary) {
        return <p className="drawer__empty">Stats will appear once analysis finishes.</p>;
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
        { name: 'Team 1', value: t1, fill: t1Color },
        { name: 'Team 2', value: t2, fill: t2Color },
        ...(neutral > 0 ? [{ name: 'Neutral', value: neutral, fill: '#94a3b8' }] : []),
    ].filter((item) => item.value > 0);

    const speedData = [
        { name: 'Avg', value: numberOrNull(overall.avg_speed_kmh) ?? 0, fill: '#60a5fa' },
        { name: 'Max', value: numberOrNull(overall.max_speed_kmh) ?? 0, fill: '#f59e0b' },
    ];

    const periodData = segments.map((seg, i) => ({
        name: (seg.segment_type || `Seg ${i + 1}`).replace('_', ' '),
        distance: numberOrNull(seg.total_distance_m) ?? 0,
        avg: numberOrNull(seg.avg_speed_kmh),
        max: numberOrNull(seg.max_speed_kmh),
    }));
    const speedFlag = overall.speed_reliability === 'suspect' || Number(overall.max_speed_kmh) >= 37.5;

    return (
        <div className="drawer__section-body">
            <div className="stat-grid">
                <StatRow icon="⚡" label="Max Speed" value={formatMetric(overall.max_speed_kmh, ' km/h')} sub={speedFlag ? 'verify' : null} />
                <StatRow icon="🏃" label="Avg Speed" value={formatMetric(overall.avg_speed_kmh, ' km/h')} />
                <StatRow icon="📏" label="Distance" value={formatMetric(overall.total_distance_m, ' m', 0)} />
                <StatRow icon="⚽" label="Possession" value={formatMetric(overall.possession_seconds, ' s')} />
                <StatRow icon="🔄" label="Switches" value={overall.possession_switches ?? '-'} />
            </div>
            {speedFlag && (
                <p className="drawer__note">
                    Peak speed is flagged as likely tracking/camera-motion noise.
                </p>
            )}

            <h4 className="drawer__subhead">Speed (km/h)</h4>
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

            <h4 className="drawer__subhead">Team Possession</h4>
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
                    <div><span className="poss-dot poss-dot--t1" style={t1Color ? { background: t1Color } : {}} /> Team 1 <strong>{t1.toFixed(1)}%</strong></div>
                    <div><span className="poss-dot poss-dot--t2" style={t2Color ? { background: t2Color } : {}} /> Team 2 <strong>{t2.toFixed(1)}%</strong></div>
                    {neutral > 0 && <div><span className="poss-dot poss-dot--neutral" /> Neutral <strong>{neutral.toFixed(1)}%</strong></div>}
                </div>
            </div>
            <PossessionBar team1={t1} team2={t2} neutral={neutral} t1Color={t1Color} t2Color={t2Color} />

            {periodData.length > 0 && (
                <>
                    <h4 className="drawer__subhead">By Period — Distance (m)</h4>
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
                            <tr><th>Period</th><th>Dist</th><th>Avg</th><th>Max</th></tr>
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
