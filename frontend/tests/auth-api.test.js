// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from 'vitest';
const fake = vi.hoisted(() => ({ getUser: vi.fn(), rpc: vi.fn(), maybeSingle: vi.fn() }));
vi.mock('@supabase/supabase-js', () => ({ createClient: vi.fn(() => ({
    auth: { getUser: fake.getUser }, rpc: fake.rpc,
    from: () => ({ select: () => ({ eq: () => ({ maybeSingle: fake.maybeSingle }) }) }),
})) }));
vi.stubEnv('SUPABASE_URL', 'https://test.supabase.co');
vi.stubEnv('SUPABASE_ANON_KEY', 'public-key');
const { requireSupabaseUser, requireSessionOwner } = await import('../api/_authMiddleware.js');
const { createJobTicket, verifyJobTicket } = await import('../api/_jobTicket.js');
const response = () => { const res = { setHeader: vi.fn(), status: vi.fn(), json: vi.fn() }; res.status.mockReturnValue(res); res.json.mockReturnValue(res); return res; };
const req = { headers: { authorization: 'Bearer signed-token' } };

describe('Vercel customer authentication', () => {
    beforeEach(() => { vi.clearAllMocks(); fake.rpc.mockResolvedValue({ data: true }); });
    it('blocks missing JWT before contacting Supabase', async () => {
        const res = response(); expect(await requireSupabaseUser({ headers: {} }, res)).toBeNull();
        expect(res.status).toHaveBeenCalledWith(401); expect(fake.getUser).not.toHaveBeenCalled();
    });
    it.each([{ is_anonymous: true, email_confirmed_at: 'today' }, { is_anonymous: false }])('blocks anonymous and unverified identities', async (user) => {
        fake.getUser.mockResolvedValue({ data: { user } }); const res = response();
        expect(await requireSupabaseUser(req, res)).toBeNull(); expect(res.status).toHaveBeenCalledWith(403);
    });
    it('blocks a retained JWT after its auth session is revoked', async () => {
        fake.getUser.mockResolvedValue({ data: { user: { id: 'a', email_confirmed_at: 'today' } } });
        fake.rpc.mockResolvedValue({ data: false }); const res = response();
        expect(await requireSupabaseUser(req, res)).toBeNull(); expect(res.status).toHaveBeenCalledWith(401);
    });
    it('fails closed during database validation failure without claiming a bad password', async () => {
        fake.getUser.mockResolvedValue({ data: { user: { id: 'a', email_confirmed_at: 'today' } } });
        fake.rpc.mockResolvedValue({ error: { code: 'unavailable' } }); const res = response();
        expect(await requireSupabaseUser(req, res)).toBeNull(); expect(res.status).toHaveBeenCalledWith(503);
    });
    it('rejects a different owner even if a broken RLS policy returns their row', async () => {
        fake.maybeSingle.mockResolvedValue({ data: { id: 'session-b', user_id: 'b' } }); const res = response();
        expect(await requireSessionOwner(req, res, 'session-b', 'jwt-a', 'a')).toBeNull(); expect(res.status).toHaveBeenCalledWith(403);
    });
    it('accepts only verified active customers and their own analysis', async () => {
        const user = { id: 'a', phone_confirmed_at: 'today' };
        fake.getUser.mockResolvedValue({ data: { user } });
        expect(await requireSupabaseUser(req, response())).toEqual(user);
        fake.maybeSingle.mockResolvedValue({ data: { id: 'session-a', user_id: 'a' } });
        expect(await requireSessionOwner(req, response(), 'session-a', 'jwt-a', 'a')).toEqual({ id: 'session-a', user_id: 'a' });
    });
});

describe('RunPod result tickets', () => {
    beforeEach(() => vi.stubEnv('RUNPOD_API_KEY', 'test-signing-key'));
    it('binds results to customer, session and exact job', () => {
        const ticket = createJobTicket('cpu:job-1', 'a', 'session-a');
        expect(verifyJobTicket(ticket, 'cpu:job-1', 'a')?.sessionId).toBe('session-a');
        expect(verifyJobTicket(ticket, 'cpu:job-1', 'b')).toBeNull();
        expect(verifyJobTicket(ticket, 'job-1', 'a')).toBeNull();
    });
    it('rejects tampering, expired tickets and missing signatures', () => {
        vi.useFakeTimers(); const ticket = createJobTicket('job', 'a', 'session');
        const [payload, sig] = ticket.split('.');
        const changed = Buffer.from(JSON.stringify({ ...JSON.parse(Buffer.from(payload, 'base64url')), userId: 'b' })).toString('base64url');
        expect(verifyJobTicket(`${changed}.${sig}`, 'job', 'b')).toBeNull();
        expect(verifyJobTicket(payload, 'job', 'a')).toBeNull();
        vi.advanceTimersByTime(86400001); expect(verifyJobTicket(ticket, 'job', 'a')).toBeNull(); vi.useRealTimers();
    });
});
