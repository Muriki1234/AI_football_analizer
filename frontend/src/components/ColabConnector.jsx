/**
 * ColabConnector.jsx
 * PitchLogic — Colab 后端连接组件
 *
 * 放置位置：src/components/ColabConnector.jsx
 *
 * 用途：提供一个可复用的 UI 组件，供用户输入 Colab URL 并验证连接
 * 可嵌入任何需要 Colab 后端的页面顶部
 */

import { useState, useEffect } from 'react'
import colab from '../services/colabService'

export default function ColabConnector({ onConnected, onDisconnected }) {
  const [url, setUrl]         = useState(colab.getUrl())
  const [status, setStatus]   = useState('idle')   // idle | connecting | connected | error
  const [gpuInfo, setGpuInfo] = useState(null)
  const [error, setError]     = useState('')

  // 页面加载时，如果已有缓存 URL 则自动验证
  useEffect(() => {
    if (colab.isConfigured()) {
      handleConnect(colab.getUrl())
    }
  }, [])

  async function handleConnect(targetUrl = url) {
    if (!targetUrl.trim()) return
    setStatus('connecting')
    setError('')
    try {
      colab.setUrl(targetUrl.trim())
      const health = await colab.ping()
      setGpuInfo(health.gpu)
      setStatus('connected')
      onConnected?.(health)
    } catch (e) {
      setStatus('error')
      setError(e.message || 'Connection failed')
      colab.clearUrl()
      onDisconnected?.()
    }
  }

  function handleDisconnect() {
    colab.clearUrl()
    setStatus('idle')
    setGpuInfo(null)
    setError('')
    onDisconnected?.()
  }

  // ── 样式（匹配 PitchLogic 深色工业风）──────────────────────────────────
  const styles = {
    container: {
      background: 'var(--card, #1a1a2e)',
      border: `1px solid ${status === 'connected' ? '#27ae60' : status === 'error' ? '#e74c3c' : '#333'}`,
      borderRadius: '8px',
      padding: '14px 18px',
      display: 'flex',
      alignItems: 'center',
      gap: '12px',
      flexWrap: 'wrap',
    },
    label: {
      color: '#aaa',
      fontSize: '12px',
      fontWeight: 600,
      letterSpacing: '0.08em',
      textTransform: 'uppercase',
      whiteSpace: 'nowrap',
    },
    input: {
      flex: 1,
      minWidth: '260px',
      background: '#0d0d1a',
      border: '1px solid #333',
      borderRadius: '6px',
      padding: '7px 12px',
      color: '#fff',
      fontSize: '13px',
      fontFamily: 'monospace',
      outline: 'none',
    },
    btn: {
      padding: '7px 16px',
      borderRadius: '6px',
      border: 'none',
      cursor: 'pointer',
      fontSize: '13px',
      fontWeight: 600,
    },
    connectBtn: {
      background: '#3498db',
      color: '#fff',
    },
    disconnectBtn: {
      background: '#333',
      color: '#ccc',
    },
    gpuBadge: {
      display: 'flex',
      alignItems: 'center',
      gap: '6px',
      background: '#0d1f12',
      border: '1px solid #27ae60',
      borderRadius: '20px',
      padding: '4px 12px',
      fontSize: '12px',
      color: '#2ecc71',
      whiteSpace: 'nowrap',
    },
    errorText: {
      color: '#e74c3c',
      fontSize: '12px',
      width: '100%',
    },
  }

  return (
    <div style={styles.container}>
      <span style={styles.label}>⚡ Colab GPU</span>

      {status !== 'connected' ? (
        <>
          <input
            style={styles.input}
            type="text"
            placeholder="https://xxxx.trycloudflare.com"
            value={url}
            onChange={e => setUrl(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleConnect()}
            disabled={status === 'connecting'}
          />
          <button
            style={{ ...styles.btn, ...styles.connectBtn }}
            onClick={() => handleConnect()}
            disabled={status === 'connecting'}
          >
            {status === 'connecting' ? 'Connecting...' : 'Connect'}
          </button>
        </>
      ) : (
        <>
          {gpuInfo?.available ? (
            <div style={styles.gpuBadge}>
              <span>●</span>
              <span>{gpuInfo.name}</span>
              <span style={{ opacity: 0.7 }}>
                {gpuInfo.free_gb}GB free / {gpuInfo.memory_gb}GB
              </span>
            </div>
          ) : (
            <div style={{ ...styles.gpuBadge, borderColor: '#f39c12', color: '#f39c12' }}>
              <span>●</span>
              <span>CPU only — GPU not detected</span>
            </div>
          )}
          <button
            style={{ ...styles.btn, ...styles.disconnectBtn }}
            onClick={handleDisconnect}
          >
            Disconnect
          </button>
        </>
      )}

      {error && <span style={styles.errorText}>⚠ {error}</span>}
    </div>
  )
}
