export default async function handler(req, res) {
  // We expect a GET request with ?id=xxxx
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const jobId = req.query.id;
  if (!jobId) {
    return res.status(400).json({ error: 'Job ID is required' });
  }

  const runpodUrl = process.env.RUNPOD_ENDPOINT_URL;
  const runpodKey = process.env.RUNPOD_API_KEY;

  if (!runpodUrl || !runpodKey) {
    return res.status(500).json({ error: 'RunPod configuration is missing on the server.' });
  }

  try {
    // RunPod's status URL is usually: https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}
    // We construct it by replacing "/run" with "/status/{job_id}"
    const statusUrl = runpodUrl.replace('/run', `/status/${jobId}`);

    const response = await fetch(statusUrl, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${runpodKey}`
      }
    });

    const data = await response.json();
    return res.status(response.status).json(data);
  } catch (error) {
    console.error('RunPod status proxy error:', error);
    return res.status(500).json({ error: 'Failed to fetch status from AI server.' });
  }
}
