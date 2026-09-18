import React from 'react';
import { render, fireEvent, screen } from '@testing-library/react';
import TelestrationCanvas from '../TelestrationCanvas';
import { describe, it, expect, vi, beforeEach } from 'vitest';

describe('TelestrationCanvas', () => {
    beforeEach(() => {
        vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue({
            top: 0, left: 0, bottom: 450, right: 800, width: 800, height: 450, x: 0, y: 0
        });
    });

    it('renders the canvas and toolbar when active', () => {
        const { container } = render(<TelestrationCanvas active={true} width={800} height={450} />);
        
        const canvas = container.querySelector('canvas');
        expect(canvas).toBeInTheDocument();
        expect(canvas).not.toHaveClass('telestration-canvas--inactive');
        
        const toolbar = container.querySelector('.draw-toolbar');
        expect(toolbar).toHaveClass('is-visible');
    });

    it('hides toolbar and makes canvas inactive when active is false', () => {
        const { container } = render(<TelestrationCanvas active={false} width={800} height={450} />);
        
        const canvas = container.querySelector('canvas');
        expect(canvas).toHaveClass('telestration-canvas--inactive');
        
        const toolbar = container.querySelector('.draw-toolbar');
        expect(toolbar).not.toHaveClass('is-visible');
    });

    it('changes tool on toolbar button click', () => {
        const { container } = render(<TelestrationCanvas active={true} width={800} height={450} />);
        
        const arrowBtn = container.querySelector('button[title="箭头"]');
        fireEvent.click(arrowBtn);
        expect(arrowBtn).toHaveClass('is-active');
        
        const penBtn = container.querySelector('button[title="画笔"]');
        expect(penBtn).not.toHaveClass('is-active');
    });

    it('clears strokes on clear button click', () => {
        const parentRef = React.createRef();
        const { container } = render(<TelestrationCanvas active={true} parentRef={parentRef} initialStrokes={[{ points: [{x:0, y:0}, {x:10, y:10}] }]} />);
        
        expect(parentRef.current.getStrokes()).toHaveLength(1);
        
        const clearBtn = container.querySelector('button[title="全部清除"]');
        fireEvent.click(clearBtn);
        
        expect(parentRef.current.getStrokes()).toHaveLength(0);
    });
    

});
