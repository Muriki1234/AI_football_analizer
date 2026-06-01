import {
  requireSupabaseUser,
  requireSessionOwner,
  extractJwt,
} from './_authMiddleware.js';

export default async function handler(req, res) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // 1) JWT 校验：必须是登录用户（包含匿名）
  const user = await requireSupabaseUser(req, res);
  if (!user) return;

  // 2) Ownership 校验：req.body.input.session_id 必须属于当前 user。
  //    RLS 在数据库层会兜底，但这里显式查能给清晰的 403 报错。
  const input = req.body?.input || {};
  const sessionId = input.session_id;
  if (!sessionId) {
    return res.status(400).json({ error: 'input.session_id is required' });
  }

  const jwt = extractJwt(req);
  const session = await requireSessionOwner(req, res, sessionId, jwt);
  if (!session) return;

  // 3) 不信任前端传来的 video_url：从 DB 取服务端可信版本。
  //    之前用户 A 能 POST {input: {session_id: B 的 id, video_url: 任意 URL}} —
  //    后端 SSRF 防线还会再拦一次，但这里直接断绝。
  const serverInput = {
    ...input,
    session_id: sessionId,
    video_url: session.video_url,
  };

  // 4) Forward to RunPod —— 路由到 GPU 或 CPU pool。
  //
  // CPU pool 处理 ai_summary + 几个 matplotlib 图表 (heatmap / charts)。这些
  // 任务在 GPU worker 上跑会让 GPU 闲置 1-3 min，烧钱。CPU worker $0.05/hr
  // vs GPU $0.50/hr = 省 90%。
  //
  // 如果 RUNPOD_CPU_ENDPOINT_URL 没设，全部 fallback 到 GPU endpoint（保持
  // 向后兼容；只有一个 endpoint 的部署不受影响）。
  // 这个集合必须跟 server/handler.py 里的 _CPU_FEATURES 保持一致。
  const CPU_FEATURES = new Set([
    'ai_summary',
    'heatmap',
    'speed_chart',
    'possession',
    'sprint_analysis',
    'defensive_line',
  ]);

  const isCpuTask =
    serverInput.action === 'feature' &&
    CPU_FEATURES.has(serverInput.feature);

  const gpuUrl  = process.env.RUNPOD_ENDPOINT_URL;
  const cpuUrl  = process.env.RUNPOD_CPU_ENDPOINT_URL;
  const runpodKey = process.env.RUNPOD_API_KEY;

  const runpodUrl = isCpuTask && cpuUrl ? cpuUrl : gpuUrl;

  if (!runpodUrl || !runpodKey) {
    return res.status(500).json({ error: 'RunPod configuration is missing on the server.' });
  }

  try {
    const response = await fetch(runpodUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${runpodKey}`,
      },
      body: JSON.stringify({ input: serverInput }),
    });
    const data = await response.json();
    return res.status(response.status).json(data);
  } catch (error) {
    console.error('RunPod proxy error:', error);
    return res.status(500).json({ error: 'Failed to communicate with AI server.' });
  }
}
