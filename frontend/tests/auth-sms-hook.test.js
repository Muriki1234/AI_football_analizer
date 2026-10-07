import { Readable } from 'node:stream';
import { Webhook } from 'standardwebhooks';
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
const fake = vi.hoisted(() => ({ send: vi.fn() }));
vi.mock('tencentcloud-sdk-nodejs-sms', () => ({ default: { sms: { v20210111: { Client: class { SendSms = fake.send; } } } } }));
import handler from '../api/auth-sms-hook';
const secret = Buffer.alloc(32, 7).toString('base64');
const res = () => { const r = { setHeader: vi.fn(), status: vi.fn(), json: vi.fn() }; r.status.mockReturnValue(r); r.json.mockReturnValue(r); return r; };
function request(phone = '8613800138000', signed = true, timestamp = new Date()) {
    const body = JSON.stringify({ user: { phone }, sms: { otp: '123456' } });
    const req = Readable.from([body]); req.method = 'POST';
    req.headers = signed ? { 'webhook-id': 'hook-1', 'webhook-timestamp': String(Math.floor(timestamp.getTime() / 1000)), 'webhook-signature': new Webhook(secret).sign('hook-1', timestamp, body) } : {};
    return req;
}
describe('signed Supabase SMS delivery', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        for (const [key, value] of Object.entries({ AUTH_PHONE_OTP_ENABLED: 'true', AUTH_SMS_HOOK_SECRET: `v1,whsec_${secret}`, AUTH_SMS_ALLOWED_PREFIXES: '+86,+64', TENCENT_SMS_SECRET_ID: 'fake', TENCENT_SMS_SECRET_KEY: 'fake', TENCENT_SMS_APP_ID: 'fake', TENCENT_SMS_TEMPLATE_CN: 'domestic', TENCENT_SMS_SIGN_CN: 'FootNova', TENCENT_SMS_TEMPLATE_INTL: 'international' })) vi.stubEnv(key, value);
        fake.send.mockResolvedValue({ SendStatusSet: [{ Code: 'Ok' }] });
    });
    afterEach(() => vi.unstubAllEnvs());
    it('rejects unsigned or expired requests before contacting the provider', async () => {
        for (const req of [request('8613800138000', false), request('8613800138000', true, new Date(Date.now() - 600000))]) {
            const r = res(); await handler(req, r); expect(r.status).toHaveBeenCalledWith(401);
        }
        expect(fake.send).not.toHaveBeenCalled();
    });
    it.each([['8613800138000', 'domestic'], ['64211234567', 'international']])('routes the verified number to its domestic or overseas template', async (phone, template) => {
        const r = res(); await handler(request(phone), r);
        expect(r.status).toHaveBeenCalledWith(200);
        expect(fake.send).toHaveBeenCalledWith(expect.objectContaining({ PhoneNumberSet: [`+${phone}`], TemplateId: template, TemplateParamSet: ['123456'] }));
    });
    it('blocks unconfigured destination regions and missing provider credentials', async () => {
        const r = res(); await handler(request('447700900123'), r); expect(r.status).toHaveBeenCalledWith(400);
        vi.stubEnv('TENCENT_SMS_SECRET_KEY', ''); const r2 = res(); await handler(request(), r2); expect(r2.status).toHaveBeenCalledWith(503);
        expect(fake.send).not.toHaveBeenCalled();
    });
    it('reports provider failure without revealing the OTP, phone or credential', async () => {
        fake.send.mockRejectedValue(new Error('123456 fake-secret 8613800138000')); const r = res(); await handler(request(), r);
        expect(r.status).toHaveBeenCalledWith(503);
        expect(JSON.stringify(r.json.mock.calls)).not.toContain('123456');
    });
});
