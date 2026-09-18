import React, { useState } from 'react';
import PropTypes from 'prop-types';
import {
  HiChartBar,
  HiShieldCheck,
  HiArrowsPointingIn,
  HiPlayCircle,
  HiInformationCircle,
} from 'react-icons/hi2';
import './PitchZoneAnalysisPanel.css';

// 20-Zone Juego de Posición Grid Definition: 4 Rows x 5 Columns
// Rows (top to bottom: Opponent Goal -> Own Goal):
// 1. Penalty Box (攻方大禁区)
// 2. Attacking Third (前场进攻三区, 含 Zone 14)
// 3. Middle Third (中场控制三区)
// 4. Defensive Third (后场防守三区)
const PITCH_ROWS = [
  {
    id: 'box',
    label: '禁区',
    zones: [
      { id: 'box_lf', name: '左禁角', defaultPct: 4.2 },
      { id: 'box_lhs', name: '左肋禁区', defaultPct: 8.5, isHalfSpace: true },
      { id: 'box_c', name: '禁区核心', defaultPct: 11.2 },
      { id: 'box_rhs', name: '右肋禁区', defaultPct: 9.1, isHalfSpace: true },
      { id: 'box_rf', name: '右禁角', defaultPct: 3.8 },
    ],
  },
  {
    id: 'att',
    label: '前场三区',
    zones: [
      { id: 'att_lf', name: '左前翼', defaultPct: 7.4 },
      { id: 'att_lhs', name: '左半空间', defaultPct: 14.6, isHalfSpace: true },
      { id: 'zone_14', name: '14号位 (弧顶)', defaultPct: 12.8, isZone14: true },
      { id: 'att_rhs', name: '右半空间', defaultPct: 15.3, isHalfSpace: true },
      { id: 'att_rf', name: '右前翼', defaultPct: 8.1 },
    ],
  },
  {
    id: 'mid',
    label: '中场三区',
    zones: [
      { id: 'mid_lf', name: '左边中', defaultPct: 5.2 },
      { id: 'mid_lhs', name: '左半中场', defaultPct: 9.8, isHalfSpace: true },
      { id: 'mid_c', name: '中场中枢', defaultPct: 14.1 },
      { id: 'mid_rhs', name: '右半中场', defaultPct: 10.5, isHalfSpace: true },
      { id: 'mid_rf', name: '右边中', defaultPct: 6.0 },
    ],
  },
  {
    id: 'def',
    label: '后场三区',
    zones: [
      { id: 'def_lf', name: '左后卫', defaultPct: 3.1 },
      { id: 'def_lhs', name: '左肋防区', defaultPct: 5.0, isHalfSpace: true },
      { id: 'def_c', name: '后防中枢', defaultPct: 7.2 },
      { id: 'def_rhs', name: '右肋防区', defaultPct: 5.4, isHalfSpace: true },
      { id: 'def_rf', name: '右后卫', defaultPct: 3.5 },
    ],
  },
];

export default function PitchZoneAnalysisPanel({ onSeekTimestamp, zoneStats }) {
  const [hoveredZone, setHoveredZone] = useState(null);

  const handleSeek = (timeStr) => {
    if (!onSeekTimestamp) return;
    const match = timeStr.match(/(\d{2}):(\d{2})/);
    if (match) {
      const sec = parseInt(match[1], 10) * 60 + parseInt(match[2], 10);
      onSeekTimestamp(sec);
    }
  };

  // Metrics (computed or fallbacks)
  const halfSpacePct = zoneStats?.halfSpaceOccupancyPct ?? 48.4;
  const zone14Entries = zoneStats?.zone14EntriesCount ?? 14;
  const restDefenseStructure = zoneStats?.restDefenseStructure ?? '3+2 稳固';
  const teamDepth = zoneStats?.teamDepthM ?? '23.6m';

  const analystReportText = `本场比赛球队在空间分配上呈现出鲜明的现代位置主义体系。进攻重心高度倾斜于双侧半空间（占前场触球的 ${halfSpacePct}%），尤其是右半空间与前场翼侧的肋部超载，制造了多次防线穿透。14 号黄金区域共完成了 ${zone14Entries} 次有效连接，重点关注 [04:12] 的肋部直塞与 [14:32] 的弧顶分边。防守端，反击遏制结构（Restabsicherung）保持在 ${restDefenseStructure}，全队平均纵向压缩在 ${teamDepth}，极好地切断了对手快速纵深通过的路线。`;

  return (
    <div className="pitch-zone-panel">
      {/* Panel Header */}
      <div className="pitch-zone-panel__header">
        <div className="pitch-zone-panel__title-group">
          <HiChartBar className="pitch-zone-panel__icon" />
          <div>
            <h3 className="pitch-zone-panel__title">20区空间战术雷达 (JDP 4×5)</h3>
            <p className="pitch-zone-panel__subtitle">Juego de Posición 4纵深×5通道（含14号黄金位与双肋半空间）战术网格</p>
          </div>
        </div>
        <span className="pitch-zone-panel__badge">Pro Analytics</span>
      </div>

      {/* 20-Zone Pitch Visual Grid */}
      <div className="pitch-grid-wrapper">
        <div className="pitch-grid-header">
          <span className="pitch-grid-header__label">
            {hoveredZone ? `当前高亮: ${hoveredZone.name} (${hoveredZone.pct}%)` : '球场20分区热力网格 (攻方朝上)'}
          </span>
          <div className="pitch-grid-legend">
            <span className="pitch-grid-legend__dot" style={{ background: 'rgba(245, 158, 11, 0.8)' }} />
            <span>14号黄金位</span>
            <span className="pitch-grid-legend__dot" style={{ background: 'rgba(0, 180, 216, 0.6)' }} />
            <span>半空间(肋部)</span>
          </div>
        </div>

        <div className="pitch-grid-board">
          {PITCH_ROWS.map((row) =>
            row.zones.map((z) => {
              const pct = zoneStats?.zonePcts?.[z.id] ?? z.defaultPct;
              let extraClass = '';
              if (z.isZone14) extraClass = 'pitch-grid-cell--zone14';
              else if (z.isHalfSpace) extraClass = 'pitch-grid-cell--halfspace';

              return (
                <div
                  key={z.id}
                  className={`pitch-grid-cell ${extraClass}`}
                  onMouseEnter={() => setHoveredZone({ name: z.name, pct })}
                  onMouseLeave={() => setHoveredZone(null)}
                >
                  <span className="pitch-grid-cell__name">{z.name}</span>
                  <span className="pitch-grid-cell__pct">{pct}%</span>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* KPI Metric Cards */}
      <div className="pitch-metrics-grid">
        <div className="pitch-metric-card">
          <div className="pitch-metric-card__header">
            <span className="pitch-metric-card__label">半空间利用率</span>
            <HiInformationCircle style={{ color: 'var(--color-text-muted)' }} />
          </div>
          <div className="pitch-metric-card__val pitch-metric-card__val--accent">{halfSpacePct}%</div>
          <span className="pitch-metric-card__sub">左右肋部高频渗透</span>
        </div>

        <div className="pitch-metric-card">
          <div className="pitch-metric-card__header">
            <span className="pitch-metric-card__label">14号位渗透次数</span>
            <HiInformationCircle style={{ color: 'var(--color-text-muted)' }} />
          </div>
          <div className="pitch-metric-card__val pitch-metric-card__val--warning">{zone14Entries} 次</div>
          <span className="pitch-metric-card__sub">大禁区弧顶核心触球</span>
        </div>

        <div className="pitch-metric-card">
          <div className="pitch-metric-card__header">
            <span className="pitch-metric-card__label">防守防反结构</span>
            <HiShieldCheck style={{ color: 'var(--color-accent)' }} />
          </div>
          <div className="pitch-metric-card__val">{restDefenseStructure}</div>
          <span className="pitch-metric-card__sub">Restabsicherung 体系</span>
        </div>

        <div className="pitch-metric-card">
          <div className="pitch-metric-card__header">
            <span className="pitch-metric-card__label">纵向紧凑度</span>
            <HiArrowsPointingIn style={{ color: 'var(--color-text-muted)' }} />
          </div>
          <div className="pitch-metric-card__val">{teamDepth}</div>
          <span className="pitch-metric-card__sub">全队纵深防守间距</span>
        </div>
      </div>

      {/* Professional Tactical Analyst Report */}
      <div className="pitch-analyst-section">
        <div className="pitch-analyst-section__title">
          <HiChartBar />
          <span>战术分析师客观复盘</span>
        </div>
        <p className="pitch-analyst-section__text">
          {analystReportText.split(/(\[\d{2}:\d{2}\])/g).map((part, i) => {
            if (/\[\d{2}:\d{2}\]/.test(part)) {
              return (
                <button
                  key={i}
                  className="pitch-timestamp-btn"
                  onClick={() => handleSeek(part)}
                  title={`点击跳转至 ${part}`}
                >
                  <HiPlayCircle />
                  {part}
                </button>
              );
            }
            return part;
          })}
        </p>
      </div>
    </div>
  );
}

PitchZoneAnalysisPanel.propTypes = {
  onSeekTimestamp: PropTypes.func,
  zoneStats: PropTypes.shape({
    halfSpaceOccupancyPct: PropTypes.number,
    zone14EntriesCount: PropTypes.number,
    restDefenseStructure: PropTypes.string,
    teamDepthM: PropTypes.string,
    zonePcts: PropTypes.objectOf(PropTypes.number),
  }),
};
