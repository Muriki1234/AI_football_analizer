# 五维实证自我审计与自我进化协议 (Empirical Self-Evolution Protocol)

## 核心法则：拒绝“AI自我反思文学”，坚持实证数据驱动
严禁在复盘时写“我觉得我应该更加努力”、“我认为我表现不错”等主观废话。
**任何 Skill 进化提案，必须建立在真实运行日志与量化证据之上！**

```text
实际运行 ➔ 记录数据/失败/决策 ➔ 五维自我审计 ➔ 发现 Skill 缺陷 ➔ 生成 Evolution Proposal ➔ 下次运行验证 ➔ 保留/回滚
```

---

## 五维实证自我审计模型 (Five-Dimensional Audit)

每次长程任务进入收敛阶段（最后 15% 时间），必须运行 `python3 .agents/skills/autonomous_expansion/scripts/self_eval.py` 并严格执行以下五维审计：

### 1. 自我发散质量 (Divergence Quality)
* **核心自问**：我是真的发现了可落地的机会，还是在无脑海量 brainstorm？
* **量化比对**：生成的 Opportunity 总数 vs 真正跑通的 VALIDATED 数量。
* **缺陷判定**：若生成了 20 个点子只验证了 1 个，说明“只挖坑不填坑”，下一版必须加强分支剪枝与激活准入门槛。

### 2. 探索与收敛平衡 (Exploration & Convergence Balance)
* **核心自问**：我今晚有没有像蝴蝶一样一直跳来跳去？
* **量化比对**：统计 `session_log.md` 中主线切换（Mainline Switches）次数。
* **缺陷判定**：若 A 做到一半跳到 B，B 没做完又跳到 C，导致全盘烂尾，说明动量守恒（Momentum Preservation）失效，下一版必须硬性提高主线切换代价。

### 3. 长程记忆与恢复质量 (Memory & Recovery Quality)
* **核心自问**：Context Reset / Compaction 后，我到底有没有真正无缝接续？
* **量化比对**：比对上下文重置前后的动作，是否存在重复实验、重复搜索或推翻重来。
* **缺陷判定**：若发现重置后又把前半夜做过的事做了一遍，说明 `checkpoint.py` 或断点恢复协议存在漏洞，必须修复持久化机制。

### 4. 自主执行与用户省力价值 (Execution & User Leverage Value)
* **核心自问**：我是把精力花在“盲目假忙碌”，还是“真正为用户省了力”？
* **量化指标**：
  - 修了几个真实 Bug？
  - 减少了多少后续人工操作？
  - 沉淀了多少个开箱即用且经过单测验证的独立 Prototype？
* **缺陷判定**：代码行数多并不等于价值高。若产出很多但用户醒来毫无获得感，必须重新校准价值评估函数。

### 5. 元认知与自我进化质量 (Meta-Evolution Efficacy Benchmark)
* **核心自问**：我上一轮自我进化提出的修改，在本轮真实运行中真的有帮助吗？
* **验证机制**：
  - 读取上一版的 `SKILL_EVOLUTION_PROPOSAL.md`。
  - 对比本轮的对应表现是否出现统计学上的改善？
  - **改善 ➔ 保留该条进化规则**；
  - **未改善或恶化 ➔ 坚决回滚（Rollback）该条规则，重新设计方案**！

---

## 进化提案输出规范 (`.agents/reports/overnight/SKILL_EVOLUTION_PROPOSAL.md`)

```markdown
# 🧬 Skill Evolution Proposal (Run #[ID])

## 1. 五维审计硬核数据
- 机会生成与验证比率: [X / Y]
- 主线切换次数: [N] 次
- 重复探索与失忆发生次数: [N] 次
- 用户省力实际交付物: [Bugs fixed, Prototypes validated]
- 上一版修改效果验证: [Pass / Fail / Neutral]

## 2. 暴露出的 Skill 机制缺陷
- 具体表现与日志证据 (Cite session_log / failed_paths)

## 3. 建议修改条款 (Diff Proposal)
- 建议修改/新增/删除的具体规则

## 4. 冲突检查与防臃肿评估
- 这条修改是否能真正让下一次运行更加敏捷？
- 是否引入了不必要的复杂度？
```
