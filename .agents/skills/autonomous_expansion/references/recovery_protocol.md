# 上下文恢复协议 (Context Recovery Protocol)

当出现以下情况时，必须无条件执行本协议：
1. 经历系统重启或新 Session；
2. 检测到底层 Context Window 发生压缩（Compaction / Truncation）；
3. 长时间暂停后被唤醒；
4. 发生未知中断或错误恢复。

## 恢复 8 步仪式 (严格按序执行，严禁凭空重新规划)
1. 运行 `python3 .agents/skills/autonomous_expansion/scripts/checkpoint.py get`，检视当前状态；
2. 读取 `.agents/memory/session_log.md` 最近 20 行，回忆刚才停在哪；
3. 运行 `python3 .agents/skills/autonomous_expansion/scripts/opportunity_graph.py list`，检视当前机会图谱；
4. 读取 `.agents/memory/failed_paths.md`，确认已排除的死胡同；
5. 确认当前认领的主线任务（Mainline）；
6. 确认上一个已经完成的动作（Last Completed Action）；
7. 确认紧接着必须执行的动作（Next Action）；
8. 正式接管并继续执行，记入日志。
