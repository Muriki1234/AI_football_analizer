import { createHmac, timingSafeEqual } from 'node:crypto';

const signature = (payload) => {
    if (!process.env.RUNPOD_API_KEY) throw new Error('Job signing unavailable');
    return createHmac('sha256', process.env.RUNPOD_API_KEY).update(`footnova-job-v1:${payload}`).digest();
};

export function createJobTicket(jobId, userId, sessionId) {
    const payload = Buffer.from(JSON.stringify({ jobId, userId, sessionId, exp: Math.floor(Date.now() / 1000) + 86400 })).toString('base64url');
    return `${payload}.${signature(payload).toString('base64url')}`;
}

export function verifyJobTicket(ticket, jobId, userId) {
    if (typeof ticket !== 'string' || ticket.length > 2048) return null;
    try {
        const parts = ticket.split('.');
        if (parts.length !== 2) return null;
        const expected = signature(parts[0]);
        const actual = Buffer.from(parts[1], 'base64url');
        if (expected.length !== actual.length || !timingSafeEqual(expected, actual)) return null;
        const claims = JSON.parse(Buffer.from(parts[0], 'base64url').toString());
        if (claims.jobId !== jobId || claims.userId !== userId || !Number.isFinite(claims.exp) || claims.exp <= Date.now() / 1000 || typeof claims.sessionId !== 'string') return null;
        return claims;
    } catch { return null; }
}
