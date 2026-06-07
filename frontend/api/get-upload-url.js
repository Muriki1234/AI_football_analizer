// Vercel serverless: generate a presigned R2 PUT URL so the browser can
// upload the video directly to Cloudflare R2, bypassing Supabase Storage's
// 50 MB free-tier limit.
//
// Auth: requires a valid Supabase JWT (same as all other /api/* routes).
// The presigned URL is scoped to a single object key and expires in 1 hour.

import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import { requireSupabaseUser } from './_authMiddleware.js';

const {
    R2_ACCOUNT_ID,
    R2_BUCKET_NAME,
    R2_ACCESS_KEY_ID,
    R2_SECRET_ACCESS_KEY,
    R2_ENDPOINT_URL,
    R2_PUBLIC_URL,
} = process.env;

let _s3 = null;
function getS3() {
    if (!_s3) {
        if (!R2_ACCESS_KEY_ID || !R2_SECRET_ACCESS_KEY || !R2_BUCKET_NAME) {
            throw new Error('R2 credentials not configured on Vercel');
        }
        const endpoint = R2_ENDPOINT_URL
            || `https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com`;
        _s3 = new S3Client({
            region: 'auto',
            endpoint,
            credentials: {
                accessKeyId: R2_ACCESS_KEY_ID,
                secretAccessKey: R2_SECRET_ACCESS_KEY,
            },
        });
    }
    return _s3;
}

export default async function handler(req, res) {
    // Only POST
    if (req.method !== 'POST') {
        res.setHeader('Allow', 'POST');
        return res.status(405).json({ error: 'Method not allowed' });
    }

    // Auth — same Supabase JWT check as analyze.js / detect_frame.js
    const user = await requireSupabaseUser(req, res);
    if (!user) return; // 401 already sent

    const { sessionId, fileName, contentType } = req.body || {};
    if (!sessionId || !fileName) {
        return res.status(400).json({ error: 'sessionId and fileName required' });
    }

    // Sanitise file name — same logic as the frontend's uploadVideo()
    const safeName = (fileName || 'video.mp4')
        .replace(/[^A-Za-z0-9._-]+/g, '_')
        .replace(/^[-.]+/, '');
    const key = `${sessionId}/${safeName}`;

    try {
        const s3 = getS3();
        const cmd = new PutObjectCommand({
            Bucket: R2_BUCKET_NAME,
            Key: key,
            ContentType: contentType || 'video/mp4',
        });
        // Presigned PUT URL — browser will PUT the file body directly here
        const uploadUrl = await getSignedUrl(s3, cmd, { expiresIn: 3600 });

        // The permanent video URL for DB storage / backend download.
        // Prefer the free public URL (no egress cost on R2); fall back to
        // constructing one from the endpoint (works if public access is on).
        let videoUrl;
        if (R2_PUBLIC_URL) {
            videoUrl = `${R2_PUBLIC_URL.replace(/\/+$/, '')}/${key}`;
        } else {
            // No public URL configured — generate a long-lived presigned GET
            const { GetObjectCommand } = await import('@aws-sdk/client-s3');
            const getCmd = new GetObjectCommand({
                Bucket: R2_BUCKET_NAME,
                Key: key,
            });
            videoUrl = await getSignedUrl(s3, getCmd, {
                expiresIn: 60 * 60 * 24 * 7,  // 7 days
            });
        }

        return res.status(200).json({ uploadUrl, videoUrl, key });
    } catch (e) {
        console.error('[get-upload-url] R2 presign failed:', e);
        return res.status(500).json({ error: 'Failed to generate upload URL' });
    }
}
