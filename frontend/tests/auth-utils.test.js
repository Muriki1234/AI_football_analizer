// @vitest-environment jsdom
import { describe, expect, it, beforeEach } from 'vitest';
import { isCustomer, normalizePhone, safeReturnPath } from '../src/lib/auth';
import { setRecentSessionUser, getRecentSessions, addRecentSession } from '../src/lib/recentSessions';

describe('customer identities and return destinations', () => {
    it('rejects guests and unverified users', () => {
        expect(isCustomer({ id: 'a', is_anonymous: true, email_confirmed_at: 'today' })).toBe(false);
        expect(isCustomer({ id: 'a', email: 'test@example.com' })).toBe(false);
        expect(isCustomer({ id: 'a', phone_confirmed_at: 'today' })).toBe(true);
    });
    it('keeps the analysis link while rejecting external and auth redirects', () => {
        expect(safeReturnPath('/dashboard?sessionId=abc')).toBe('/dashboard?sessionId=abc');
        for (const url of ['//evil.com', '/\\evil.com', 'https://evil.com', '/login', '/auth/callback', '/reset-password', '/\nevil.com']) expect(safeReturnPath(url)).toBe('/');
    });
    it('normalizes mainland and international numbers without truncating E.164', () => {
        expect(normalizePhone('138 0013 8000')).toBe('+8613800138000');
        expect(normalizePhone('021 123 4567', '+64')).toBe('+64211234567');
        expect(normalizePhone('+44 7700 900123')).toBe('+447700900123');
        expect(() => normalizePhone('123')).toThrow();
        expect(() => normalizePhone('+8611111111111')).toThrow();
    });
});

describe('shared-browser recent analyses', () => {
    beforeEach(() => {
        const values = new Map();
        Object.defineProperty(globalThis, 'localStorage', { value: { getItem: (k) => values.get(k) ?? null, setItem: (k,v) => values.set(k,v), removeItem: (k) => values.delete(k), clear: () => values.clear() }, configurable: true });
        localStorage.clear(); setRecentSessionUser(null); });
    it('never shows the previous customer or legacy anonymous cache', () => {
        localStorage.setItem('pitchlogic.recentSessions', JSON.stringify([{ id: 'legacy', videoUrl: 'private' }]));
        setRecentSessionUser('a'); addRecentSession({ id: 'a-video', videoUrl: 'a-private' });
        expect(getRecentSessions()[0].id).toBe('a-video');
        setRecentSessionUser('b'); expect(getRecentSessions()).toEqual([]);
        expect(localStorage.getItem('pitchlogic.recentSessions.a')).toBeNull();
        expect(localStorage.getItem('pitchlogic.recentSessions')).toBeNull();
        addRecentSession({ id: 'b-video' }); setRecentSessionUser(null);
        expect(getRecentSessions()).toEqual([]);
    });
});
