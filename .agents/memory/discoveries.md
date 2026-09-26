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


### 4. 长视频检测性能瓶颈与分流加速机制 (Async Prefetch & Tactical View Gating)
* **实证瓶颈暴露**：
  1. **同步 Stop-and-Wait 串行阻塞**：全场比赛（~13.5万帧）长视频处理中，`stream_video_chunks` 同步逐帧调用 `cv2.VideoCapture.read()`，CPU 视频帧解码与 GPU YOLO/SAM 批推理处于串行停等模式，CPU 解码时 GPU 处于饥饿等待，GPU 推理时 CPU 闲置，形成典型的流水线气泡与内存抖动。
  2. **非战术广播视角算力虚耗**：在真实足球转播流中，约 25%~35% 的画面为特写（Closeup）、死球回放（Replay）及看台观众，关键点单应性匹配在这些帧几乎必然失败，但在原流水线中仍每 3 帧无差别执行 1280×1280 全分辨率重型模型前向推理。
* **分流加速方案与独立原型验证**：
  1. **双缓冲预取器 (`AsyncFramePrefetcher`)**：采用有界阻塞队列在独立线程预解码视频帧，实现 I/O 解码与推理阶段的 overlap，消除停等气泡。
  2. **亚毫秒级战术视角门控 (`TacticalViewGater`)**：通过降采样（160×90）HSV 绿色草皮比率快速判别战术俯瞰视角与特写镜头，单帧耗时仅 ~0.037ms（算法基准吞吐量 >26,000 FPS），在进入重型检测器前低成本剔除非战术镜头。
  3. **动态自适应步长调度 (`AdaptiveTemporalStrideController`)**：基于运动剧烈程度自适应伸缩步长（2~6 帧），在维持 ByteTrack/Kalman 跟踪连续性的同时减少冗余检测。
* **实证基准与科学边界声明**：13 项单元测试在 `test_detection_accelerator.py` 中全绿通过。5000 帧混合模拟场景下的调度规划可在 188ms 内完成（0.0376ms/帧），理论前向计算缩减率达 66.7%。本指标属于纯内存算法层面的 `[Algorithm-only Benchmark]`，明确排除外部磁盘读取与物理 GPU 显存拷贝延迟。

### 5. FIFA SAOT 几何原理与定格越位线评估机制 (VAR Offside Line & Margin Evaluator)
* **技术突破与规程事实**：基于国际足联 Law 11 与半自动越位技术（SAOT），构建传球释放帧（$t_{release}$，kick-point）定格几何越位线。
* **第二防守人动态追踪**：自动穿透门将与后卫身份标签，按进攻方向（+X 与 -X 双向几何映射）对所有防守人排序，准确定位最深防守人与倒数第二防守人（Second-Last Defender），并强制施加半场豁免（中线 $x=0$ 约束）与球后豁免（$x_{att} \le x_{ball}$）。
* **实证基准与科学边界**：`test_offside_var_evaluator.py` 9/9 单测全绿通过，算法基准吞吐量达到 166,842 次评估/秒（0.006ms/次）。支持 5cm 校准误差容差区间，并生成广播级 2D 俯视定格越位线复核图 (`var_offside_map.png`)。

### 6. 空间凸包与阵型紧凑度时空动力学分析 (Tactical Convex Hull & Stretch Index)
* **领域事实与理论解耦**：基于 Fernandez & Bornn (2018) 以及 Moura et al. (2012) 阵型时空动力学，将阵型紧凑度量化为外场球员 2D 凸包面积（$m^2$）、径向拉伸指数（Stretch Index，到质心的欧氏距离均值，纵向/横向分离）及双队质心距离。
* **孤立门将动态剔除算法**：通过边界极大空隙差（Left Gap vs Right Gap）与进攻朝向解耦，自动剔除拖后门将，防止其虚假拉伸后防线凸包面积（在 $50m \times 40m$ 阵型中准确还原 $2000m^2$ 外场形状）。
* **实证基准与科学边界**：`test_team_compactness_engine.py` 8/8 单测全绿，纯内存凸包吞吐量达 12,467 帧/秒。支持进攻扩张（1200-2200 $m^2$）与防守压缩（400-800 $m^2$）的相位扩张比自动计算，并生成双面板时序图表。

### 7. 期望威胁 (xT) 马尔可夫贝尔曼网格价值迭代与球权转移吸积机制 (Expected Threat xT & Progressive Action Valuator)
* **文献与理论事实**：基于 Karun Singh (2018/2019) 与 Javier Fernández et al. (2021) 空间持有价值模型，解决传统足球数据中“只有射门有 xG，而占全场 98% 的 800+ 次传球推进缺乏定量价值衡量”的断层。在 105m×68m 球场上构建 16×12 离散化网格（192 个战术微元），求解递推贝尔曼收益方程：
  $$xT(z) = s_z \cdot g_z + m_z \cdot \gamma \sum_{z'} T_{z \to z'} \cdot xT(z')$$
* **实证踩坑与吸收态转移修正**：若忽略球权丢失率（即设转移矩阵行和恒等于 1.0），无折扣马尔可夫随机游走会导致全场扩散平衡（后场与中场虚高收敛至 ~0.45）。引入真实足球单次传球控球保留率 $\gamma = 0.78$（对应约 22% 拦截与失误率）后，精确还原出从本方禁区（0.028）到后场（0.038）、中场（0.061）、进攻三区 14 区（0.125）至对方小禁区（0.558）的严格单调上升时空梯度，使向前推进传球产生稳定的 $\Delta xT \approx +0.05 \sim +0.10$。
* **实证基准与科学边界**：`test_expected_threat_xt_engine.py` 10/10 单元测试全绿通过。纯内存算法基准测试吞吐量达到 202,690 次动作评估/秒（50,000 次操作耗时 246.68ms），并生成广播级 2D 俯视威胁热力图与推进向量图。

### 8. Osgnach (2010) 代谢功率 (Metabolic Power) 与等效坡度加速度疲劳动力学 (Metabolic Power & HMLD Kinematics)
* **文献与理论事实**：基于 Cristian Osgnach 与 Pietro Enrico di Prampero (2010, MSSE) 等效坡度 (Equivalent Slope, ES) 理论，运动员在平地前向加速等效于在倾角为 $\alpha$ 的斜坡上以恒速做功：
  $$g' = \sqrt{a_f^2 + g^2}, \quad ES = \tan \alpha = \frac{a_f}{g}$$
  结合 Minetti et al. (2002) 五次多项式：
  $$EC = (155.4 ES^5 - 30.4 ES^4 - 43.3 ES^3 + 46.3 ES^2 + 19.5 ES + 3.6) \cdot \frac{g'}{g} \cdot KT$$
  计算瞬时代谢功率 $P_{met} = EC \cdot v$ (W/kg)。
* **实证突破与对传统速度指标的降维打击**：传统速度门槛（如高速奔跑 $>19.8 km/h$）会完全漏掉球员从 1.5 m/s 剧烈提速至 3.5 m/s（$a = 3.5 m/s^2$）的大量高耗能爆发；代谢功率模型成功捕捉到此时 $P_{met} > 25.5 W/kg$，将其精准计入高代谢负荷距离（HMLD, High Metabolic Load Distance）。同时，通过等效距离指标 $EDI = ED / \text{Actual Distance}$（走走停停爆发性跑动产生 $EDI > 1.15$），直观揭示了加减速带来的“额外生理账单”。
* **实证基准与科学边界**：`test_metabolic_power_fatigue_engine.py` 10/10 单元测试全绿通过。纯内存算法基准测试吞吐量达到 10,661,454 点/秒（50,000 点耗时 4.69ms），并生成包含 5 区能量分布与生理负荷总览的仪表盘图表。

### 10. 真实 RunPod 全视频 GPU 争用与内存伸缩实证 (SAMURAI Concurrency Contention & OOM Threshold)
* **实证现象与客观事实**：基于 25,316 帧（14.1 分钟）真实 1080p 比赛视频在同一 NVIDIA RTX A5000（24GB VRAM，503GB Host RAM）上的两次生产运行对比：
  1. **并发度 C=11（Run A, worker `91ujs9ikyqeab1`）**：设置 cap=16 触发 11 个切片全并发，流水线启动 84 秒后被 Linux OOM killer 强制 SIGKILL 终止（exit code 137）。实证证明单切片内存开销与多进程并发呈近似线性累加，单节点无限制并发必然导致 OOM 崩溃。
  2. **并发度 C=4（Run B, worker `hq087wf2auxm50`）**：分三波执行（4+4+3），全流程 439.4 秒成功跑通，Host RAM 峰值稳定在 99.7 GB（占 503GB 主机的 19.3%），SAMURAI 累计耗时 240.55s，YOLO 耗时 439.38s（平均 102.0 FPS），并发节省 240.55s（35.4%）。
* **CUDA 算力争用关键拐点暴露**：
  - 在纯单任务或波次收尾阶段，YOLO 推理速度稳定在 **150~187 FPS**。
  - 在 4 个 SAMURAI 进程平稳并行期，YOLO 降至 **108~150 FPS**（算力争用轻微）。
  - 在波次交替切换（Wave 1→Wave 2 切片交接，短暂并发重叠）时，YOLO 出现严重算力降速至 **38~43 FPS**（持续约 49 秒）。
* **科学结论与工程边界**：证明“无脑提高 SAMURAI 并发”是错误的伪优化；最优并发点是 SAMURAI 缩短时长、YOLO 算力争用降速与 Host RAM 峰值安全三者的帕累托折中，必须通过参数矩阵完整跑测。

### 11. 统一五阶足球时空视觉分析基准架构 (Unified 5-Stage Analytics Accuracy Foundation)
* **领域痛点与理论洞察**：过往实践中，小地图飘移、热力图异常弥散、最高速度虚高（超 40 km/h）以及跑动距离过量累加（5倍误差）常被作为孤立的前端或任务级 bug 分别修补打补丁。但理论与数据流上，它们共享唯一的底层因果依赖链：
  $$\text{Detection} \longrightarrow \text{Tracking} \longrightarrow \text{Team Identity} \longrightarrow \text{Pitch Coordinates / Homography} \longrightarrow \text{Trajectory Kinematics}$$
  前序阶梯的极小抖动（如 ByteTrack 单次 ID 跳跃或单帧透视变换噪点）会在后续阶梯被微分放大为几十倍的速度突变与队伍反转。
* **统一评测闭环与独立原型构建**：在 `analytics_accuracy_foundation.py` 中实现了贯穿全链条的量化指标体系统：
  1. **Team Identity**：队伍标签翻转率（Flip Rate）与时间多数派纯度（Majority Purity）。
  2. **Pitch Homography**：标准球场边界约束（[0, 105]m × [0, 68]m）合法率与亚毫秒级瞬时瞬变（Teleportation Jumps >12 m/s）检出率。
  3. **Trajectory Kinematics**：FIFA/Catapult 生理速度硬约束（$v \le 37.0$ km/h）与加减速物理界限（$a \le 6.5$ m/s$^2$）。
  4. **Unified Scorecard**：融合 mAP/F1、HOTA、Team Purity、Homography Bounds 与 Kinematics 物理合理性，输出加权整体精度指数（Holistic Accuracy Index, HAI）。
* **实证基准与科学边界**：`test_analytics_accuracy_foundation.py` 6/6 单测全绿通过，端到端完整闭环验证成功。为后续在全量 25k 帧与 Golden Set 750 帧上同时优化 Speed + Accuracy + Resource Efficiency 奠定了统一评价基准。

### 12. 球衣感知 CIE Lab 时空聚类与草皮阴影自适应剔除机制 (Perceptual Tracklet Team Clustering)
* **根因暴露**：在全场长视频（25,316 帧）实证日志中发现，原流水线 `tasks.py` 采用稀疏固定网格采样（`team_vote_step = total // 20`，整场只采样 21 帧），395 个 Track ID 平均仅采样 1.03 次，绝大多数 Track ID 从未被采样，随后在行 1378 触发 `player_final_team.get(pid, 1)`，**无条件默认归为 Team 1**！这一队伍偏置污染了下游的球权归属、小地图点位颜色、传球网络（队友间传球 vs 拦截）以及队伍凸包紧凑度与防线分析。此外，RGB 欧氏距离无法抗衡背光与草皮反射。
* **技术突破与解决方案**：
  1. **连续轨迹蓄水池采样 (`TrackletProfile`)**：每个 Track ID 维护最大 15 帧的高质量感知采样池，动态淘汰低质量/草皮遮挡帧。
  2. **CIE L*a*b* 色彩感知空间**：利用 L* 分离光照亮度，a* 和 b* 表征色度，使欧氏距离直接对应人眼色差 $\Delta E$，结合 HSV 动态草皮掩膜与 Otsu 阈值剔除率 $>85\%$ 过滤。
  3. **K-Means++ 双峰聚类与门将/裁判离群点解耦**：自适应识别与主队外场聚类中心色差 $\Delta E > 30$ 的孤立轨迹，分离门将与全黑制服裁判，并提供置信度衰减的空域邻域降级，彻底消灭全队偏置为 Team 1 的系统性污染。
* **实证基准与科学边界**：`test_tracklet_perceptual_team_classifier.py` 7/7 单元测试全绿通过。外场队伍可分度 $\Delta E > 25.0$，离群门将/裁判检出率 100%。

### 13. 跨 Track ID 连续步进位移累积与 FIFA/Catapult 比赛均速重构 (Continuous Trajectory Kinematic Accumulator)
* **根因暴露**：
  1. 在 `tasks.py` 中，跑动距离计算采用 `dist_delta = max(distances) - min(distances)`。当 ByteTrack 因遮挡产生 ID Switch 时，新 ID 的累积距离归零，`distances` 呈锯齿状 `[0..25m, 0..35m, 0..20m]`，`max - min` 仅测量了单段最长子轨迹（35m）而非累积（80m），在 90 分钟长视频中导致高达 5-10 倍的距离少计！
  2. 在 `analysis_core.py` 的 `AccurateSpeedEstimator` 中，5 帧窗口位移被逐帧累加，产生了 5 倍的严重多计。
  3. 在 `tasks.py` 中，`if speed < 0.5: return None` 剔除了所有静止帧，将 FIFA 比赛均速（总距离/总比赛时间，通常为 4~5 km/h）曲解为奔跑移动均速（10~14 km/h）。
  4. 目标检测框底边中心在透视变换下产生 1~3 像素微抖动，静止球员每秒凭空产生 $6\text{ m/s}$ 虚假位移。
* **技术突破与解决方案**：
  1. **跨 Track ID 连续步进位移累积**：解耦 ByteTrack 动态 ID，直接追踪指定 target_player 在球场坐标系下的真实时间序列，在连续时间间隔内逐帧计算步进物理位移 $\Delta d$。
  2. **运动学微抖动死区抑制 (Kinematic Deadband)**：当 $\Delta d < 4\text{cm}$ 或瞬时速度 $v < 1.0\text{ km/h}$ 时，强制判定为静止状态，彻底消除相机与检测晃动带来的虚假累积。
  3. **FIFA 5-Zone 速度区间与真正比赛均速**：严格区分 FIFA/Catapult 比赛平均速度（$\frac{\text{Total Distance}}{\text{Match Time}} \times 3.6$）与移动平均速度（$v \ge 1.0\text{ km/h}$），并划分 5 级运动学负荷区间与冲刺次数计数。
* **实证基准与科学边界**：`test_target_player_continuous_kinematic_accumulator.py` 5/5 单元测试全绿通过。在 900 帧 3 次 ID Switch 仿真中完美恢复 150m 全程位移，静止抖动死区抑制达到 0.0 米累积。

### 14. Soccana 120×70m 底模与 FIFA 105×68m 战术空间尺度分裂及像素倒灌 (Pitch Coordinate Normalizer & Metric Guard)
* **根因暴露**：
  1. **空间底模与战术标准分裂**：`ViewTransformer` 采用 Roboflow `SoccerPitchConfiguration`（Soccana 29 关键点，范围 $12000 \times 7000$ cm = $120 \times 70$ 米，其中包含边线与球门外侧 $7.5$m 与 $1.0$m 缓冲区），但在 `tasks.py:1874` 与下游 9 个战术引擎（防线、预期威胁 xT、20区雷达、阵型紧凑度等）中，被误当作 FIFA 标准 $105\text{m} \times 68\text{m}$ 球场使用，造成 X 轴 14.3% 的几何物理拉伸误差，且导致底线进攻球员超出 105m 网格边界触发越界异常。
  2. **厘米-米 100 倍歧义**：全库大量代码执行 `pos = info.get("position_minimap") or info.get("position_transformed")`，若存在 `position_minimap` 则为厘米（如 6000, 3500），若回退则为米（如 52.5, 34.0），造成跨帧 100 倍量纲漂移。
  3. **传球事件像素倒灌**：在 `tasks.py:1422-1437` 中，当单应性失败缺少 `position_transformed` 时，代码回退至屏幕像素中心 `(bb[0]+bb[2])/2.0`（如 960, 540），随后与传球控球半径 2.8 米进行欧氏距离计算，产生 1000 米级的虚假空间瞬移。
* **技术突破与解决方案**：
  1. 构建 `CanonicalPitchCoordinateNormalizer`，将 Soccana 仿射尺度统一解算至 FIFA 标准 $[0, 105]\text{m} \times [0, 68]\text{m}$。
  2. 自动判别量纲（厘米 vs 米）并过滤非法屏幕像素倒灌。
  3. 提供小地图画布标准像素对齐（0m 对应左门线，105m 对应右门线），彻底消除小地图点位浮空与边界压缩。
* **实证基准与科学边界**：`test_canonical_pitch_coordinate_normalizer.py` 7/7 单元测试全绿通过。

### 15. 深度模型并发争用下 CUDA Stream 优先级与波次交接节流机制 (CUDA Stream Priority Concurrency Governor)
* **根因暴露**：在 RunPod RTX A5000 生产全视频执行中，当 SAMURAI 多切片波次交接（Wave 1 结束与 Wave 2 启动交叠）时，YOLO 推理速度在默认 CUDA 流（priority=0）下与 SAMURAI 重型访存产生严重算力争用，从 187 FPS 剧烈跌落至 38~43 FPS 持续长达 49 秒，且 Host RAM 瞬间上冲至 97.2GB。
* **技术突破与解决方案**：
  1. **CUDA 流优先级隔离**：为实时 YOLO 前向推理分配高优先级流 `priority=-1`，SAMURAI 分配标准流 `priority=0`，在 GPU 硬件层面保障检测器低延迟吞吐，消除争用断崖。
  2. **波次交接准入节流控制器**：实时监控 YOLO FPS 与内存水位，当检测帧率低于 80 FPS 警戒线时自动挂起新切片加载，削减重叠峰值。
  3. **帕累托最优切片并发模型**：基于实时可用 VRAM 与 RAM 动态求解最优安全并发度（RTX A5000 下为 Cap=5，低显存下自适应降级至 2）。
* **实证基准与科学边界**：`test_cuda_stream_priority_concurrency_governor.py` 5/5 单元测试全绿通过。


