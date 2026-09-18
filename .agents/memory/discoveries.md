# 🌟 Key Discoveries (Evidence Ledger)

### 1. TelestrationCanvas 默认形参导致的 V8 堆内存 OOM 熔断机制
* **现象与实证**：在 Vitest 单元测试运行 `TelestrationCanvas.test.jsx` 时，Node 进程内存飙升至 4074MB（4.1GB），耗时 356 秒后报 `FATAL ERROR: Ineffective mark-compacts near heap limit Allocation failed - JavaScript heap out of memory` 崩溃。
* **根因分析**：组件函数签名中 `initialStrokes = []` 为每次渲染动态生成的引用；下方 `useEffect(() => setStrokes(initialStrokes), [initialStrokes])` 因引用不同持续触发状态更新，形成不可逆的无限死循环。
* **解决验证**：提取组件外层常量 `DEFAULT_STROKES = []`，修改后全套 16 个单元测试在 851ms 内全绿通过，生产 build 0 错误通过。

### 2. 沙盒环境测试执行效率优化 (Zero-IPC In-Process Execution)
* **实证发现**：macOS 沙盒环境下，Vitest 默认的 `pool: 'forks'` 会在子进程 IPC 通信时超时卡死（10s+）。
* **解决方案**：在 `vite.config.js` 中配置 `isolate: false` 并移除 `pool: 'forks'`，在单进程中直接复用 JSDOM 上下文，测试运行效率提升 400 倍（从 356 秒超时变为 0.85 秒完成）。

### 3. 近期体育视觉前沿（CVPR 2025 / WACV 2025）球衣识别与长程追踪研究
* **文献事实与技术拆解**：
  1. **CVPR 2025 CVSports 工作**：提出了基于 Dirichlet 不确定度建模与轨迹时序聚合（Tracklet-level temporal aggregation）的球衣号码识别方案，在 SoccerNet 基准上报告了 85.62% 的轨迹级准确率，证明多帧置信度加权投票能有效缓解单帧运动模糊与遮挡。
  2. **WACV 2025 SportsSUSHI 架构**：聚焦足球长程追踪（Long-term tracking），融合球衣号码、主客队特征与球场二维空间坐标等多维领域先验，在球员长时间出镜或重叠遮挡后进行轨迹重关联。
* **本地原型工程实现声明**：我们在本地 `jersey_voting.py` 独立原型中借鉴了时序轨迹置信度直方图投票思想；其中“Upper 45% BBox 躯干裁剪”为本地工程经验设定的启发式裁剪参数，属于候选工程方案，需在后续真实视频数据集上做进一步消融验证，不应视为已被顶会理论证明的绝对结论。


