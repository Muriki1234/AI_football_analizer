import React from 'react';
import { render, waitFor } from '@testing-library/react';
import CanvasOverlay from '../CanvasOverlay';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

describe('CanvasOverlay', () => {
    let videoRef;
    let mockFetch;

    beforeEach(() => {
        videoRef = {
            current: {
                paused: false,
                ended: false,
                videoWidth: 1280,
                videoHeight: 720,
                currentTime: 0.0,
                addEventListener: vi.fn(),
                removeEventListener: vi.fn(),
            }
        };
        mockFetch = vi.fn().mockResolvedValue({
            json: () => Promise.resolve({
                fps: 25,
                frames: [
                    // Frame 0
                    [
                        [10, 10, 20, 20], // targetBbox
                        [[1, 0, 0, 10, 10, 1, false]], // players: id, x1, y1, x2, y2, team, has_ball
                        [5, 5, 8, 8] // ballBbox
                    ]
                ],
                team_colors: { "1": "#ff0000" }
            }),
        });
        global.fetch = mockFetch;
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('renders a canvas element with correct opacity based on visible prop', () => {
        const { container, rerender } = render(<CanvasOverlay dataUrl="test.json" videoRef={videoRef} visible={false} />);
        let canvas = container.querySelector('canvas');
        expect(canvas).toBeInTheDocument();
        expect(canvas).toHaveStyle({ opacity: '0' });

        rerender(<CanvasOverlay dataUrl="test.json" videoRef={videoRef} visible={true} />);
        expect(canvas).toHaveStyle({ opacity: '1' });
    });

    it('fetches overlay data when dataUrl is provided', async () => {
        render(<CanvasOverlay dataUrl="test-data.json" videoRef={videoRef} visible={true} />);
        expect(mockFetch).toHaveBeenCalledWith('test-data.json');
        
        await waitFor(() => {
            expect(videoRef.current.addEventListener).toHaveBeenCalledWith('seeked', expect.any(Function));
            expect(videoRef.current.addEventListener).toHaveBeenCalledWith('timeupdate', expect.any(Function));
        });
    });

    it('cleans up event listeners on unmount', async () => {
        const { unmount } = render(<CanvasOverlay dataUrl="test-data.json" videoRef={videoRef} visible={true} />);
        
        await waitFor(() => {
            expect(videoRef.current.addEventListener).toHaveBeenCalled();
        });

        unmount();

        expect(videoRef.current.removeEventListener).toHaveBeenCalledWith('seeked', expect.any(Function));
        expect(videoRef.current.removeEventListener).toHaveBeenCalledWith('timeupdate', expect.any(Function));
    });
});
