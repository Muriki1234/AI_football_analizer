import { requireSupabaseUser } from './_authMiddleware.js';

export default async function handler(req, res) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Auth: 必须是 Supabase 已登录（含匿名）用户。防止外部直接打这个 endpoint
  // 薅 RunPod GPU 羊毛。requireSupabaseUser 失败时已经写过 res，直接 return。
  const user = await requireSupabaseUser(req, res);
  if (!user) return;

  // Get the secret keys from Vercel's environment variables
  const runpodUrl = process.env.RUNPOD_ENDPOINT_URL;
  const runpodKey = process.env.RUNPOD_API_KEY;

  if (!runpodUrl || !runpodKey) {
    return res.status(500).json({ error: 'RunPod configuration is missing on the server.' });
  }

  try {
    // Forward the request body to RunPod
    const response = await fetch(runpodUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${runpodKey}`
      },
      body: JSON.stringify(req.body)
    });

    const data = await response.json();

    // Return the RunPod response (like the Job ID) to our frontend
    return res.status(response.status).json(data);
  } catch (error) {
    console.error('RunPod proxy error:', error);
    return res.status(500).json({ error: 'Failed to communicate with AI server.' });
  }
}
