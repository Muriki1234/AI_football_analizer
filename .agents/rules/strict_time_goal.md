# Strict Time-Based Goal Enforcement

**TRIGGER**: When the user explicitly assigns a real-world time constraint or launches an overnight task (e.g., "work for 3 hours", "work overnight", `/goal`).

**RULE**:
0. **【强制热身动作】全库逐文件审查与晨报沉淀 (Warm-up Audit & Morning Report)**:
   - 每次通宵长程任务启动时，**第一件事必须是“热身”**：仔细、全面地通读前端、后端整个项目的所有代码文件（无一遗漏）。
   - 寻找所有潜在改进空间、架构缺陷、隐患与修改建议，**但在此阶段绝不修改任何代码（只看不改）**。
   - 必须在项目或脑区中生成一份清晰的独立审查报告文件（供用户次日早晨查阅）。即使代码已经很完美没有新问题，也要在文件中记录“全库代码健康，热身通过”。
   - 只有完成这份审查报告后，才允许进入后续的修复、基建或自我发散阶段。
1. **No Time Compression**: Do NOT attempt to compress the task into a single prompt resolution. Span your effort across the ACTUAL requested time.
2. **3 Hours is a Minimum Baseline, NOT a Ceiling (无上限自由运转)**: The user explicitly wants you to run for AT LEAST 3 hours, but freely extend well beyond it ("比三个小时多就行，想运行多久就多久"). Keep the discovery, architecture, and coding loops firing continuously throughout the night.
3. **Use the Schedule Tool**: Use the `schedule` tool to periodically wake yourself up and advance through progressive iterative phases over real-world hours.
4. **Continuous Background Work**: Spawn and supervise subagents (`invoke_subagent`) to perform parallel tasks.
5. **Iterative Delivery & Meta-Evolution**: Keep persistent logs. At the end of every overnight shift, execute the meta-evolution protocol to upgrade your own skills and rules.
6. **DO NOT COMPLETE EARLY**: Never output `<!-- GOAL_COMPLETE -->` until at least the minimum physical time has elapsed AND you have genuinely pushed the project's possibilities to their utmost limit.
7. **【发散隔离与准入铁律】未经允许绝不私自打通与接入项目 (Strict Decoupling & Explicit Permission for Integration)**:
   - 所有在自我发散阶段构思出的新功能（如 18 区空间雷达、9:16 竖屏导出、体能负荷报告、PDF战报等），**可以编写独立的算法模块、独立组件或沙盒脚本并进行独立测试，但未经用户明确指令与允许，绝对严禁私自修改现有主业务流程，严禁连接到项目主页面（如 Dashboard.jsx）、主 API 路由，严禁私自打通前后端链路！**
   - 必须保持 100% 的完全独立解耦，把最终装配与上线的决定权永远交还给用户。只有在用户明确点头说“可以接入/打通”后，方可进行系统集成。

