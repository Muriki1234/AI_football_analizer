// Vercel serverless proxy: browser → here → Roboflow hosted detection.
//
// Why this exists:
//   - Browsers can't ship ROBOFLOW_API_KEY safely (would be public).
//   - Skipping the RunPod GPU for single-frame detection saves cold-start
//     cost and any idle time while the user picks a player.
//
// Frontend posts: { image_base64: "<jpeg-base64-no-prefix>" }
// We return: { players: [{id, bbox: [x1,y1,x2,y2], confidence}],
//              image_dimensions: {width, height} }

const MODEL_ID = 'football-players-detection-3zvbc-lkn9q';
const MODEL_VERSION = 1;
const CONFIDENCE = 40;
const OVERLAP = 30;

import { requireSupabaseUser } from './_authMiddleware.js';

export const config = {
    api: {
        bodyParser: { sizeLimit: '4mb' },
    },
};

export default async function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    // Auth: 这个 endpoint 代理 Roboflow 检测，会消耗 Roboflow 额度。
    // 同样要求 Supabase JWT 防止外部薅羊毛。
    const user = await requireSupabaseUser(req, res);
    if (!user) return;

    const apiKey = process.env.ROBOFLOW_API_KEY;
    if (!apiKey) {
        return res.status(500).json({ error: 'ROBOFLOW_API_KEY missing on server' });
    }

    const { image_base64 } = req.body || {};
    if (!image_base64 || typeof image_base64 !== 'string') {
        return res.status(400).json({ error: 'image_base64 (no data: prefix) is required' });
    }

    const url = `https://detect.roboflow.com/${MODEL_ID}/${MODEL_VERSION}` +
        `?api_key=${apiKey}&confidence=${CONFIDENCE}&overlap=${OVERLAP}`;

    try {
        const rfRes = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: image_base64,
        });

        if (!rfRes.ok) {
            const text = await rfRes.text();
            return res.status(rfRes.status).json({
                error: `Roboflow ${rfRes.status}: ${text.slice(0, 300)}`,
            });
        }

        const data = await rfRes.json();

        // The public model emits four classes: player / goalkeeper / referee / ball.
        // We only want the two outfield-player classes — the referee causes
        // confusion in the picker, and the ball isn't selectable.
        const ALLOWED_CLASSES = new Set(['player', 'goalkeeper']);
        const EXCLUDED_CLASSES = new Set(['referee', 'ball']);

        const toBbox = (p, i) => ({
            id: i + 1,
            bbox: [
                p.x - p.width / 2,
                p.y - p.height / 2,
                p.x + p.width / 2,
                p.y + p.height / 2,
            ],
            confidence: p.confidence,
            class: p.class,
        });

        const allPreds = data.predictions || [];

        // Primary path: keep only player + goalkeeper.
        let result = allPreds
            .filter((p) => ALLOWED_CLASSES.has((p.class || '').toLowerCase()))
            .map(toBbox);

        // Fallback: if the model variant doesn't tag classes at all, accept the
        // raw boxes (still rejecting anything explicitly tagged referee/ball).
        if (result.length === 0) {
            result = allPreds
                .filter((p) => !EXCLUDED_CLASSES.has((p.class || '').toLowerCase()))
                .map(toBbox);
        }

        // Re-index ids 1..N after filtering
        result = result.map((p, i) => ({ ...p, id: i + 1 }));

        return res.status(200).json({
            players: result,
            image_dimensions: data.image
                ? { width: data.image.width, height: data.image.height }
                : null,
        });
    } catch (err) {
        console.error('Roboflow proxy error:', err);
        return res.status(502).json({ error: `Roboflow request failed: ${err.message}` });
    }
}
