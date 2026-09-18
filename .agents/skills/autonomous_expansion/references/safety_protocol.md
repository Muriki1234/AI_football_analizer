# 安全铁门与权限三区协议 (Safety & Permissions)

## 权限三区定义
* 🟢 **自主执行区**：沙盒内完全自由（读代码、全网调研、编写独立 Prototype、测试、修极小确定性 Bug）。
* 🟡 **提案区**：必须保留用户决策权（修改主页面如 Dashboard.jsx、改核心 pipeline、改数据库 schema、前后端实装连通）。标准：*Research freely. Prototype freely. Integrate deliberately.*
* 🔴 **人类决策区**：严禁擅自执行（删除重要持久化数据、不可逆迁移、生产发布）。

## Bug 安全修复 4 道铁门 (必须全部满足)
1. **Gate A: 有据可查**：必须有复现用例、清晰报错堆栈或失败单测，绝不凭空修改。
2. **Gate B: 最小修改**：默认改动不超过 20 行，不改写公共接口签名。
3. **Gate C: 回归验证**：修改后必跑单测、lint 或 typecheck。
4. **Gate D: 自我 Diff 审查**：自问我改了什么、测试证明了什么。测试报错立即 REVERT 并转入 🟡 提案区。
