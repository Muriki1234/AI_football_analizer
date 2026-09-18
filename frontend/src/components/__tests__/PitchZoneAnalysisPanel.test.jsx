import React from 'react';
import { render, fireEvent, screen } from '@testing-library/react';
import PitchZoneAnalysisPanel from '../PitchZoneAnalysisPanel';
import { describe, it, expect, vi } from 'vitest';

describe('PitchZoneAnalysisPanel Prototype', () => {
    it('renders the 20-zone tactical grid with pro analytics badge', () => {
        const { container } = render(<PitchZoneAnalysisPanel />);
        expect(screen.getByText('20区空间战术雷达 (JDP 4×5)')).toBeInTheDocument();
        expect(screen.getByText('Pro Analytics')).toBeInTheDocument();
        
        // 4 rows x 5 cols = 20 cells in the JDP matrix
        const cells = container.querySelectorAll('.pitch-grid-cell');
        expect(cells.length).toBe(20);
    });

    it('correctly marks Zone 14 and Half-space cells', () => {
        const { container } = render(<PitchZoneAnalysisPanel />);
        const zone14Cell = container.querySelector('.pitch-grid-cell--zone14');
        expect(zone14Cell).toBeInTheDocument();
        expect(zone14Cell).toHaveTextContent('14号位 (弧顶)');

        const halfSpaceCells = container.querySelectorAll('.pitch-grid-cell--halfspace');
        expect(halfSpaceCells.length).toBe(8); // 2 half-spaces per each of the 4 bands
    });

    it('updates header on cell hover and clears on mouse leave', () => {
        const { container } = render(<PitchZoneAnalysisPanel />);
        const zone14Cell = container.querySelector('.pitch-grid-cell--zone14');
        
        fireEvent.mouseEnter(zone14Cell);
        expect(screen.getByText(/当前高亮: 14号位 \(弧顶\)/)).toBeInTheDocument();

        fireEvent.mouseLeave(zone14Cell);
        expect(screen.getByText('球场20分区热力网格 (攻方朝上)')).toBeInTheDocument();
    });

    it('renders custom zone stats and KPI cards', () => {
        const customStats = {
            halfSpaceOccupancyPct: 52.3,
            zone14EntriesCount: 19,
            restDefenseStructure: '2+3 结构',
            teamDepthM: '21.4m',
            zonePcts: {
                'zone_14': 18.5
            }
        };
        const { container } = render(<PitchZoneAnalysisPanel zoneStats={customStats} />);
        expect(screen.getByText('52.3%')).toBeInTheDocument();
        expect(screen.getByText('19 次')).toBeInTheDocument();
        expect(screen.getByText('2+3 结构')).toBeInTheDocument();
        expect(screen.getByText('21.4m')).toBeInTheDocument();

        const zone14Cell = container.querySelector('.pitch-grid-cell--zone14');
        expect(zone14Cell).toHaveTextContent('18.5%');
    });

    it('invokes onSeekTimestamp with parsed seconds when clicking report timestamps', () => {
        const onSeek = vi.fn();
        const { container } = render(<PitchZoneAnalysisPanel onSeekTimestamp={onSeek} />);
        
        const timeBtns = container.querySelectorAll('.pitch-timestamp-btn');
        expect(timeBtns.length).toBe(2);

        // Click [04:12] -> 4*60 + 12 = 252s
        fireEvent.click(timeBtns[0]);
        expect(onSeek).toHaveBeenCalledWith(252);

        // Click [14:32] -> 14*60 + 32 = 872s
        fireEvent.click(timeBtns[1]);
        expect(onSeek).toHaveBeenCalledWith(872);
    });
});
