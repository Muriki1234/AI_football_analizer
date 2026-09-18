import React from 'react';
import { render, waitFor, fireEvent, act } from '@testing-library/react';
import MinimapOverlay from '../MinimapOverlay';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

describe('MinimapOverlay', () => {
    let videoRef;
    let mockFetch;

    beforeEach(() => {
        videoRef = {
            current: {
                currentTime: 0.0,
            }
        };
        mockFetch = vi.fn().mockResolvedValue({
            json: () => Promise.resolve({
                fps: 25,
                pitch: { length: 10500, width: 6800 },
                frames: [
                    [{ id: 1, x: 5000, y: 3400, t: 1 }]
                ],
                sample_stride: 1,
                sample_indices: [0]
            }),
        });
        global.fetch = mockFetch;
        // Mock localStorage
        const localStorageMock = (function() {
            let store = {};
            return {
                getItem(key) { return store[key] || null; },
                setItem(key, value) { store[key] = value.toString(); },
                clear() { store = {}; },
                removeItem(key) { delete store[key]; }
            };
        })();
        Object.defineProperty(window, 'localStorage', { value: localStorageMock });
        window.localStorage.clear();
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('renders and fetches data', async () => {
        const { container } = render(
            <MinimapOverlay dataUrl="minimap.json" videoRef={videoRef} visible={true} />
        );
        expect(mockFetch).toHaveBeenCalledWith('minimap.json');
        
        const minimapDiv = container.querySelector('.minimap-overlay');
        expect(minimapDiv).toBeInTheDocument();
    });

    it('returns null if not visible', () => {
        const { container } = render(
            <MinimapOverlay dataUrl="minimap.json" videoRef={videoRef} visible={false} />
        );
        expect(container.firstChild).toBeNull();
    });

    it('calls onExpand when expand button is clicked', async () => {
        const onExpandMock = vi.fn();
        const { container } = render(
            <MinimapOverlay dataUrl="minimap.json" videoRef={videoRef} visible={true} onExpand={onExpandMock} />
        );
        
        await waitFor(() => {
            const btn = container.querySelector('.minimap-expand-btn');
            expect(btn).toBeInTheDocument();
        });
        
        const btn = container.querySelector('.minimap-expand-btn');
        fireEvent.click(btn);
        expect(onExpandMock).toHaveBeenCalledTimes(1);
    });

    it('handles dragging functionality', async () => {
        const { container } = render(
            <MinimapOverlay dataUrl="minimap.json" videoRef={videoRef} visible={true} />
        );
        
        await waitFor(() => {
            expect(container.querySelector('.minimap-overlay')).toBeInTheDocument();
        });
        
        const wrapper = container.querySelector('.minimap-overlay');
        
        act(() => {
            fireEvent.mouseDown(wrapper, { clientX: 100, clientY: 100 });
        });
        
        expect(wrapper).toHaveClass('is-dragging');
        
        act(() => {
            fireEvent.mouseMove(window, { clientX: 150, clientY: 150 });
        });
        
        act(() => {
            fireEvent.mouseUp(window);
        });
        
        expect(wrapper).not.toHaveClass('is-dragging');
    });
});
