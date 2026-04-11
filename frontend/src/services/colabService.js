/**
 * colabService.js
 * PitchLogic — Colab GPU 后端服务层
 *
 * 放置位置：src/services/colabService.js
 *
 * 职责：
 *   - 管理 Colab URL（运行时设置，localStorage 持久化）
 *   - 封装所有与 Colab 后端的通信
 *   - 提供视频传输、任务轮询等工具函数
 *
 * 使用方式：
 *   import colab from '@/services/colabService'
 *   colab.setUrl('https://xxxx.trycloudflare.com')
 *   const gpu = await colab.ping()
 */

import axios from 'axios'

// ── 常量 ──────────────────────────────────────────────────────────────────
const STORAGE_KEY = 'pitchlogic_colab_url'
const POLL_INTERVAL = 2500   // ms
const POLL_TIMEOUT = 30 * 60 * 1000  // 30 分钟（防止无限轮询）

// ── 内部状态 ─────────────────────────────────────────────────────────────
let _baseURL = localStorage.getItem(STORAGE_KEY) || ''

// ── axios 实例（动态 baseURL）────────────────────────────────────────────
const http = axios.create({ timeout: 30_000 })

// 每次请求前注入最新 baseURL（支持运行时更换 URL）
http.interceptors.request.use(cfg => {
  if (!_baseURL) throw new Error('Colab URL not set. Call colabService.setUrl() first.')
  cfg.baseURL = _baseURL
  return cfg
})


// ═══════════════════════════════════════════════════════════════════════
// URL 管理
// ═══════════════════════════════════════════════════════════════════════

/**
 * 设置 Colab 后端 URL，并持久化到 localStorage
 * @param {string} url  例：https://xxxx.trycloudflare.com
 */
function setUrl(url) {
  _baseURL = url.replace(/\/$/, '')  // 去掉末尾斜杠
  localStorage.setItem(STORAGE_KEY, _baseURL)
}

/** 获取当前 Colab URL */
function getUrl() {
  return _baseURL
}

/** 是否已配置 URL */
function isConfigured() {
  return Boolean(_baseURL)
}

/** 清除 URL */
function clearUrl() {
  _baseURL = ''
  localStorage.removeItem(STORAGE_KEY)
}


// ═══════════════════════════════════════════════════════════════════════
// 健康检查
// ═══════════════════════════════════════════════════════════════════════

/**
 * Ping Colab 后端，返回 GPU 信息
 * @returns {{ status: string, gpu: { available: boolean, name: string, memory_gb: number } }}
 */
async function ping() {
  const { data } = await http.get('/health')
  return data
}


// ═══════════════════════════════════════════════════════════════════════
// 视频传输
// ═══════════════════════════════════════════════════════════════════════

/**
 * 将视频文件发送到 Colab（直接上传 multipart）
 *
 * @param {string}   sessionId   来自本地 Flask upload 的 video_id
 * @param {File}     file        原始视频 File 对象
 * @param {Function} onProgress  (percent: number) => void
 */
async function sendVideo(sessionId, file, onProgress) {
  const form = new FormData()
  form.append('file', file)

  const { data } = await http.post(
    `/api/${sessionId}/receive_video`,
    form,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 10 * 60 * 1000,  // 10 分钟（大视频）
      onUploadProgress: e => {
        if (onProgress && e.total) {
          onProgress(Math.round((e.loaded / e.total) * 100))
        }
      },
    }
  )
  return data
}

/**
 * 告知 Colab 视频已在服务器本地路径（共享存储场景，无需重传）
 * @param {string} sessionId
 * @param {string} videoPath  服务器上的绝对路径
 */
async function registerVideoPath(sessionId, videoPath) {
  const { data } = await http.post(`/api/${sessionId}/receive_video`, { video_path: videoPath })
  return data
}


// ═══════════════════════════════════════════════════════════════════════
// 分析流水线
// ═══════════════════════════════════════════════════════════════════════


/**
 * 启动 SAMURAI 追踪
 * @param {string} sessionId
 * @param {{ x1, y1, x2, y2, frame }} bbox  前端选框坐标
 */
async function startTracking(sessionId, bbox) {
  const { data } = await http.post(`/api/${sessionId}/track`, bbox)
  return data
}

/**
 * 启动 YOLO 全量分析（追踪完成后调用）
 */
async function startAnalysis(sessionId) {
  const { data } = await http.post(`/api/${sessionId}/analyze`)
  return data
}

/**
 * 获取 session 状态
 * @returns {{ status, progress, stage_label, error, available_features }}
 */
async function getStatus(sessionId) {
  const { data } = await http.get(`/api/${sessionId}/status`)
  return data
}


// ═══════════════════════════════════════════════════════════════════════
// 按需生成
// ═══════════════════════════════════════════════════════════════════════

/** 生成热力图，返回 task_id */
async function generateHeatmap(sessionId) {
  const { data } = await http.post(`/api/${sessionId}/generate/heatmap`)
  return data.task_id
}

/** 生成速度/距离图表，返回 task_id */
async function generateSpeedChart(sessionId) {
  const { data } = await http.post(`/api/${sessionId}/generate/speed_chart`)
  return data.task_id
}

/** 生成控球率饼图，返回 task_id */
async function generatePossession(sessionId) {
  const { data } = await http.post(`/api/${sessionId}/generate/possession`)
  return data.task_id
}

/** 生成小地图回放视频，返回 task_id */
async function generateMinimapReplay(sessionId) {
  const { data } = await http.post(`/api/${sessionId}/generate/minimap_replay`)
  return data.task_id
}

/** 生成全量回放视频，返回 task_id */
async function generateFullReplay(sessionId) {
  const { data } = await http.post(`/api/${sessionId}/generate/full_replay`)
  return data.task_id
}

/** 生成冲刺爆发分析图，返回 task_id */
async function generateSprintAnalysis(sessionId) {
  const { data } = await http.post(`/api/${sessionId}/generate/sprint_analysis`)
  return data.task_id
}

/** 生成防线渗透统计图，返回 task_id */
async function generateDefensiveLine(sessionId) {
  const { data } = await http.post(`/api/${sessionId}/generate/defensive_line`)
  return data.task_id
}

/**
 * 查询任务状态
 * @returns {{ status, progress, file_path, url, result, error }}
 */
async function getTask(sessionId, taskId) {
  const { data } = await http.get(`/api/${sessionId}/task/${taskId}`)
  return data
}

/** 获取分析摘要（球员速度/距离/控球率等）*/
async function getSummary(sessionId) {
  const { data } = await http.get(`/api/${sessionId}/summary`)
  return data
}

/**
 * 返回任务文件的完整 URL（用于 <img src> 或 <video src>）
 * @param {string} sessionId
 * @param {string} taskId
 */
function getFileUrl(sessionId, taskId) {
  return `${_baseURL}/api/${sessionId}/file/${taskId}`
}


/**
 * 视频已经在 upload 阶段发给了 Colab，这里只需要 mock 成功即可
 */
async function registerSession(sessionId, videoPath) {
  return { status: "success", message: "Video already synced during upload" }
}

/**
 * 在 Colab GPU 上通过 YOLO 进行单帧分析（检测球员）
 */
async function analyzeFrame(sessionId, timeInSeconds) {
  const { data } = await http.post(`/api/${sessionId}/analyze_frame`, {
    time_in_seconds: timeInSeconds
  })
  return data
}


/**
 * 在 Colab 上使用 ffmpeg 裁剪视频
 */
async function trimVideo(sessionId, start, end) {
  const { data } = await http.post(`/api/${sessionId}/trim`, { start, end })
  return data
}


// ═══════════════════════════════════════════════════════════════════════
// 轮询工具
// ═══════════════════════════════════════════════════════════════════════

/**
 * 轮询 session 状态，直到目标状态或失败
 *
 * @param {string}   sessionId
 * @param {string[]} targetStatuses  例：['tracking_done', 'analysis_done']
 * @param {Function} onProgress      ({ status, progress, stage_label }) => void
 * @returns {Promise<object>}        最终 session 状态对象
 *
 * @example
 * await colab.pollStatus(sessionId, ['tracking_done'], ({ progress, stage_label }) => {
 *   setProgress(progress)
 *   setLabel(stage_label)
 * })
 */
function pollStatus(sessionId, targetStatuses, onProgress) {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + POLL_TIMEOUT
    const interval = setInterval(async () => {
      try {
        const s = await getStatus(sessionId)
        onProgress?.(s)

        if (targetStatuses.includes(s.status)) {
          clearInterval(interval)
          resolve(s)
        } else if (s.status.endsWith('_failed') || s.status.endsWith('_error')) {
          clearInterval(interval)
          reject(new Error(s.error || `Task failed: ${s.status}`))
        } else if (Date.now() > deadline) {
          clearInterval(interval)
          reject(new Error('Polling timeout (30 min)'))
        }
      } catch (err) {
        clearInterval(interval)
        reject(err)
      }
    }, POLL_INTERVAL)
  })
}

/**
 * 轮询任务（按需生成任务），直到 done 或 failed
 *
 * @param {string}   sessionId
 * @param {string}   taskId
 * @param {Function} onProgress  (progress: number) => void
 * @returns {Promise<object>}    任务对象（含 url/result）
 */
function pollTask(sessionId, taskId, onProgress) {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + POLL_TIMEOUT
    const interval = setInterval(async () => {
      try {
        const t = await getTask(sessionId, taskId)
        onProgress?.(t.progress ?? 0)

        if (t.status === 'done') {
          clearInterval(interval)
          // 修正 url：把相对路径转成 Colab 绝对 URL
          if (t.url && !t.url.startsWith('http')) {
            t.url = `${_baseURL}${t.url}`
          }
          resolve(t)
        } else if (t.status === 'failed') {
          clearInterval(interval)
          reject(new Error(t.error || 'Task failed'))
        } else if (Date.now() > deadline) {
          clearInterval(interval)
          reject(new Error('Task polling timeout'))
        }
      } catch (err) {
        clearInterval(interval)
        reject(err)
      }
    }, POLL_INTERVAL)
  })
}


// ═══════════════════════════════════════════════════════════════════════
// 完整流程封装（高层 API）
// ═══════════════════════════════════════════════════════════════════════

/**
 * 完整分析流程：上传视频 → 追踪 → 分析
 *
 * @param {object} options
 * @param {string} options.sessionId     来自本地 Flask 的 video_id
 * @param {File}   options.videoFile     原始视频文件
 * @param {object} options.bbox          { x1, y1, x2, y2, frame }
 * @param {Function} options.onStep      (step: string, detail: object) => void
 *   step 可能值：'uploading' | 'tracking' | 'analyzing' | 'done'
 *
 * @returns {Promise<{ summary: object, availableFeatures: string[] }>}
 *
 * @example
 * const result = await colab.runFullAnalysis({
 *   sessionId: videoId,
 *   videoFile: file,
 *   bbox: { x1: 100, y1: 200, x2: 160, y2: 360, frame: 0 },
 *   onStep: (step, detail) => console.log(step, detail),
 * })
 */
async function runFullAnalysis({ sessionId, videoFile, bbox, onStep }) {
  // 1. 上传视频
  onStep?.('uploading', { progress: 0 })
  await sendVideo(sessionId, videoFile, pct => {
    onStep?.('uploading', { progress: pct })
  })

  // 2. 启动追踪
  onStep?.('tracking', { progress: 0, stage_label: 'Starting SAMURAI...' })
  await startTracking(sessionId, bbox)
  await pollStatus(sessionId, ['tracking_done'], s => {
    onStep?.('tracking', s)
  })

  // 3. 启动分析
  onStep?.('analyzing', { progress: 0, stage_label: 'Starting YOLO analysis...' })
  await startAnalysis(sessionId)
  await pollStatus(sessionId, ['analysis_done'], s => {
    onStep?.('analyzing', s)
  })

  // 4. 取摘要
  const summary = await getSummary(sessionId)
  onStep?.('done', { summary })

  return { summary, availableFeatures: ['heatmap', 'speed_chart', 'possession', 'minimap_replay', 'full_replay'] }
}


// ── 导出 ──────────────────────────────────────────────────────────────────
const colabService = {
  // URL 管理
  setUrl,
  getUrl,
  isConfigured,
  clearUrl,

  // 健康
  ping,

  // 视频
  sendVideo,
  registerVideoPath,
  registerSession,
  trimVideo,

  // 流水线
  analyzeFrame,
  startTracking,
  startAnalysis,
  getStatus,

  // 按需生成
  generateHeatmap,
  generateSpeedChart,
  generatePossession,
  generateMinimapReplay,
  generateFullReplay,
  generateSprintAnalysis,
  generateDefensiveLine,
  getTask,
  getSummary,
  getFileUrl,

  // 轮询
  pollStatus,
  pollTask,

  // 高层封装
  runFullAnalysis,
}

export default colabService
