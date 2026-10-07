import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
const mockAuth = vi.hoisted(() => ({ getSession: vi.fn(), getUser: vi.fn(), signOut: vi.fn(), signInAnonymously: vi.fn() }));
vi.mock('../src/lib/supabase', () => ({ supabase: { auth: mockAuth } }));
import { uploadVideo, queueFeature } from '../src/services/api';

describe('authenticated analysis requests', () => {
    beforeEach(() => { vi.clearAllMocks(); mockAuth.signOut.mockResolvedValue({ error: null }); vi.stubGlobal('fetch', vi.fn()); });
    afterEach(() => vi.unstubAllGlobals());
    it('requires a verified user before upload and never silently creates a guest', async () => {
        mockAuth.getUser.mockResolvedValue({ data: { user: null } });
        await expect(uploadVideo({ size: 100, name: 'sample.mp4' })).rejects.toThrow('Please sign in');
        expect(mockAuth.signInAnonymously).not.toHaveBeenCalled(); expect(fetch).not.toHaveBeenCalled();
    });
    it('refuses to send a paid feature request with an anonymous token', async () => {
        mockAuth.getSession.mockResolvedValue({ data: { session: { access_token: 'guest-token', user: { is_anonymous: true } } } });
        await expect(queueFeature('session', 'heatmap')).rejects.toThrow('Please sign in'); expect(fetch).not.toHaveBeenCalled();
    });
    it('attaches the customer token and ends local state after a server 401 without retrying', async () => {
        mockAuth.getSession.mockResolvedValue({ data: { session: { access_token: 'customer-token', user: { id: 'a', email_confirmed_at: 'today' } } } });
        fetch.mockResolvedValue({ status: 401, ok: false });
        await expect(queueFeature('session', 'heatmap')).rejects.toThrow('expired');
        expect(fetch).toHaveBeenCalledTimes(1);
        expect(fetch).toHaveBeenCalledWith(expect.stringContaining('/api/analyze'), expect.objectContaining({ headers: expect.objectContaining({ Authorization: 'Bearer customer-token' }) }));
        expect(mockAuth.signOut).toHaveBeenCalledWith({ scope: 'local' });
    });
});
