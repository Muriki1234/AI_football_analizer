# 巡检报告 — 2026-04-21

## 检查文件列表

**核心文件（本次检查）：**
- backend/app/pipeline/analysis_core.py
- backend/app/pipeline/tasks.py
- colab_backend.py
- frontend/src/services/colabService.js
- frontend/src/pages/Dashboard.jsx

**本次 commit 改动（HEAD: e85e190）：**
- backend/app/pipeline/analysis_core.py
- colab_backend.py
- frontend/src/components/VideoOverlayPlayer.css
- frontend/src/components/VideoOverlayPlayer.jsx
- frontend/src/pages/Dashboard.jsx

---

## 语法检查

backend/app/ 下所有 Python 文件语法检查通过（python3 -m py_compile 全部退出码 0）。

---

## 静态分析

### P1（严重）

**问题 1：tasks.run_auto_detect_and_track 被调用但未定义 — /auto_start 路由在运行时必定崩溃**

位置：colab_backend.py line 282

```python
_fire(tasks.run_auto_detect_and_track, sid, session, sm)
```

问题描述：`run_auto_detect_and_track` 在 `tasks.py` 及整个 backend 目录中完全不存在。
`_fire()` 把该调用放在 daemon 线程里执行，线程启动后立即触发
`AttributeError: module has no attribute 'run_auto_detect_and_track'`，
异常在 daemon 线程内静默消失，session 状态永远卡在 `tracking`，前端无法感知。

建议修复：在 tasks.py 实现该函数（YOLO 检测首帧 → 自动选球员 → 调用 run_samurai_tracking），
或如果 auto_start 流程暂不支持，删除该路由并从前端移除对应调用。

### P2（中等）

**问题 2：_available() 未包含 sprint_analysis / defensive_line / ai_summary**

位置：colab_backend.py line 52

```python
if st == 'analysis_done': return ['heatmap', 'speed_chart', 'possession', 'minimap_replay', 'full_replay', 'summary']
```

问题描述：前端在 `available_features` 中检查 `sprint_analysis`、`defensive_line`、`ai_summary`（Dashboard.jsx line 108 逻辑），
但这三个 feature 从未出现在 `_available()` 返回列表里。对应路由在 colab_backend.py 已注册且 tasks.py 中有完整实现，
唯独 available_features 不通知前端，导致用户在分析完成后看不到这三个功能按钮。

建议修复：将 `'sprint_analysis'`、`'defensive_line'`、`'ai_summary'` 加入 `analysis_done` 的返回列表。

**问题 3：CameraMovementEstimator 光流掩膜中硬编码 900:1050 像素列**

位置：analysis_core.py line 655

```python
mask[:, 900:1050] = 1
```

问题描述：掩膜固定取第 900-1050 像素列，假设视频宽度约 1080p（1920px）。
4K 视频（3840px）或 720p（1280px）时，该范围要么漏掉大片有效边缘区域，
要么完全落在画面中间（720p 时 900px 已在画面中段），导致光流估计偏差，
摄像机运动补偿失准，进而影响球员坐标转换和小地图精度。

建议修复：改为按帧宽度比例计算，例如 `mask[:, int(w*0.47):int(w*0.55)] = 1`。

**问题 4：AccurateSpeedEstimator.fps 硬编码为 24，未接收真实视频 fps**

位置：analysis_core.py line 1042

```python
self.fps = 24
```

tasks.py line 455 实例化时未传入 fps：

```python
speed_est = AccurateSpeedEstimator()
```

问题描述：此时 `fps` 已从视频元数据中读出并存在 `fps` 变量，但没有传给速度估算器。
对于 25fps、30fps 或 60fps 视频，计算出的速度会有 4%–150% 的系统性误差。

建议修复：`AccurateSpeedEstimator.__init__` 接受 `fps` 参数，tasks.py 中改为
`speed_est = AccurateSpeedEstimator(fps=fps)`。

### P3（轻微/建议）

- analysis_core.py line 243：局部变量 `pd` 覆盖了模块顶层的 pandas import 别名 `pd`。
  在 `detect_segments()` 方法内部 `pd = players[fi]`，虽然该方法内部不使用 pandas，
  但阅读时容易造成混淆，建议改为 `frame_data` 或 `fp`。

---

## 近期 commit 审查（最近 5 次）

- e85e190 fix: faststart remux + Range 流式播放 — 逻辑正确，原子替换文件，ffmpeg 失败有兜底
- 849daee revert minimap_replay 直接 URL 播放 — 正常回滚
- d474b98 / f518fb9 fix: pre-buffer video to blob — 前端改动，无后端影响
- 258f53d fix: 对齐 minimap 坐标系 120×70 — 配合 _PITCH_LEN/_PITCH_WID 一致，正确

未发现新增 regression。

---

## GitHub Issue 状态

gh CLI 未登录，无法自动创建 Issue。

以下问题建议 Hans 手动创建 Issue：

1. [bug] tasks.run_auto_detect_and_track 未定义，/auto_start 路由运行时崩溃
   — 文件：colab_backend.py:282

2. [bug] _available() 缺少 sprint_analysis/defensive_line/ai_summary，功能按钮不显示
   — 文件：colab_backend.py:52

3. [enhancement] CameraMovementEstimator 光流掩膜硬编码像素列，4K/720p 视频补偿失准
   — 文件：analysis_core.py:655

4. [bug] AccurateSpeedEstimator.fps 硬编码 24，非标准帧率视频速度计算有系统误差
   — 文件：analysis_core.py:1042, tasks.py:455

---

## 需要 Hans 关注

- P1：colab_backend.py /auto_start 路由引用了不存在的 tasks 函数，会静默崩溃
- P2：_available() 遗漏三个已实现功能，前端显示不出来（影响用户体验）
- P2：光流掩膜硬编码，非 1080p 视频摄像机补偿会偏差
- P2：速度估算器 fps 未从视频元数据传入，非 24fps 视频速度计算有误
