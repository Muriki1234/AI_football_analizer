# 🪦 Failed Paths & Dead Ends (避坑索引)

- [PITFALL-01] 混淆算法内存基准与端到端 Pipeline 延时：在评估 20 区空间投射和时序投票时，若未显式剥离视频解帧、YOLO 检测与 SAM2 分割开销直接宣称全链路 1~3ms，会导致工程落地预期严重失真。后续所有性能测试必须显式前缀 "Algorithm-only Benchmark" 并标注适用边界。
