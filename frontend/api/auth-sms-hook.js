import { Webhook } from 'standardwebhooks';
import tencentcloud from 'tencentcloud-sdk-nodejs-sms';

export const config = { api: { bodyParser: false } };

async function readBody(req) {
    let size = 0;
    const chunks = [];
    for await (const chunk of req) {
        const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
        size += bytes.length;
        if (size > 16384) throw new Error('Body too large');
        chunks.push(bytes);
    }
    return Buffer.concat(chunks).toString('utf8');
}

// Supabase owns OTP generation, expiry and verification. This endpoint only
// sends its signed request; public callers cannot choose a phone number/code.
export default async function handler(req, res) {
    res.setHeader('Cache-Control', 'no-store');
    if (req.method !== 'POST') {
        res.setHeader('Allow', 'POST');
        return res.status(405).json({ error: { http_code: 405, message: 'Method not allowed' } });
    }
    const secret = process.env.AUTH_SMS_HOOK_SECRET;
    if (!secret || process.env.AUTH_PHONE_OTP_ENABLED !== 'true') {
        return res.status(503).json({ error: { http_code: 503, message: 'Phone sign-in is temporarily unavailable.' } });
    }
    let payload;
    try {
        const body = await readBody(req);
        payload = new Webhook(secret.replace(/^v1,whsec_/, '')).verify(body, {
            'webhook-id': req.headers['webhook-id'],
            'webhook-timestamp': req.headers['webhook-timestamp'],
            'webhook-signature': req.headers['webhook-signature'],
        });
    } catch {
        return res.status(401).json({ error: { http_code: 401, message: 'Invalid hook signature.' } });
    }
    const rawPhone = payload?.user?.phone;
    const phone = typeof rawPhone === 'string' ? `+${rawPhone.replace(/^\+/, '')}` : '';
    const code = payload?.sms?.otp;
    const prefixes = (process.env.AUTH_SMS_ALLOWED_PREFIXES || '').split(',').map((s) => s.trim()).filter((s) => /^\+[1-9]\d{0,3}$/.test(s));
    if ((phone.startsWith('+86') && !/^\+861[3-9]\d{9}$/.test(phone)) || !/^\+[1-9]\d{7,14}$/.test(phone) || typeof code !== 'string' || !/^\d{6}$/.test(code) || !prefixes.some((prefix) => phone.startsWith(prefix))) {
        return res.status(400).json({ error: { http_code: 400, message: 'Phone sign-in is unavailable for this number. Please use email.' } });
    }
    const mainland = phone.startsWith('+86');
    const template = mainland ? process.env.TENCENT_SMS_TEMPLATE_CN : process.env.TENCENT_SMS_TEMPLATE_INTL;
    const sign = mainland ? process.env.TENCENT_SMS_SIGN_CN : process.env.TENCENT_SMS_SIGN_INTL;
    const { TENCENT_SMS_SECRET_ID: secretId, TENCENT_SMS_SECRET_KEY: secretKey, TENCENT_SMS_APP_ID: appId } = process.env;
    if (!secretId || !secretKey || !appId || !template || (mainland && !sign)) {
        return res.status(503).json({ error: { http_code: 503, message: 'Phone sign-in is temporarily unavailable.' } });
    }
    try {
        const client = new tencentcloud.sms.v20210111.Client({
            credential: { secretId, secretKey },
            region: process.env.TENCENT_SMS_REGION || 'ap-guangzhou',
            profile: { httpProfile: { endpoint: 'sms.tencentcloudapi.com', reqTimeout: 8 } },
        });
        const result = await client.SendSms({
            PhoneNumberSet: [phone], SmsSdkAppId: appId,
            TemplateId: template, TemplateParamSet: [code],
            ...(sign ? { SignName: sign } : {}),
            ...(!mainland && process.env.TENCENT_SMS_SENDER_ID ? { SenderId: process.env.TENCENT_SMS_SENDER_ID } : {}),
        });
        if (result.SendStatusSet?.[0]?.Code !== 'Ok') throw new Error('Provider rejected message');
        return res.status(200).json({});
    } catch {
        // Never log hook bodies, phone numbers, OTPs or provider credentials.
        console.error('[auth-sms] provider delivery failed');
        return res.status(503).json({ error: { http_code: 503, message: 'Unable to send a code. Please try email or retry later.' } });
    }
}
