#!/usr/bin/env python3
"""
overnight_report.py - Outcome-Driven Morning Intelligence Report Generator

Only evidence-backed bugs enter BUGS KILLED.
Only VALIDATED nodes enter STANDALONE PROTOTYPES.
PROPOSED nodes remain proposals; PARKED nodes remain backlog.
"""

import json
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../memory"))
REPORTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../reports/overnight"))
STATE_FILE = os.path.join(MEMORY_DIR, "active_state.json")
GRAPH_FILE = os.path.join(MEMORY_DIR, "opportunity_graph.json")
SESSION_LOG_FILE = os.path.join(MEMORY_DIR, "session_log.md")
DISCOVERIES_FILE = os.path.join(MEMORY_DIR, "discoveries.md")
FAILED_PATHS_FILE = os.path.join(MEMORY_DIR, "failed_paths.md")
BUGS_FILE = os.path.join(MEMORY_DIR, "bugs_resolved.json")


def load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def load_verified_bugs() -> List[Dict[str, Any]]:
    data = load_json(BUGS_FILE, {})
    bugs = data.get("resolved_bugs", []) if isinstance(data, dict) else data
    if not isinstance(bugs, list):
        return []
    return [
        bug
        for bug in bugs
        if isinstance(bug, dict)
        and bug.get("tests_passed") is True
        and not bug.get("rollback", False)
    ]


def read_text(path: str, fallback: str = "暂无") -> str:
    if not os.path.exists(path):
        return fallback
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return content if content else fallback
    except OSError:
        return fallback


def get_session_op_ids_by_target(session_id: str, target_statuses: List[str]) -> List[str]:
    if not os.path.exists(SESSION_LOG_FILE):
        return []
    op_ids: List[str] = []
    targets = {f"to={st}" for st in target_statuses}
    with open(SESSION_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if f"[{session_id}]" in line and "[EVENT=OPPORTUNITY_TRANSITION]" in line:
                if any(t in line for t in targets):
                    m = re.search(r"opportunity=([^\s]+)", line)
                    if m and m.group(1) not in op_ids:
                        op_ids.append(m.group(1))
    return op_ids


def get_session_validated_op_ids(session_id: str) -> List[str]:
    return get_session_op_ids_by_target(session_id, ["VALIDATED", "VALIDATED_SANDBOX"])


def resolve_lifecycle_header(state: Dict[str, Any], graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified Lifecycle Resolver (Defect 2 Minimal Fix).
    Ensures report header does not display split-brain states (e.g. 'exploring' when
    all opportunities are validated or budget has expired).
    """
    nodes = graph.get("nodes", {}) if isinstance(graph, dict) else {}
    active_ops = [n for n in nodes.values() if n.get("status") in {"ACTIVATED", "EXPLORING"}]
    active_m = [n["id"] for n in active_ops if n.get("tier") == "main"]
    active_s = [n["id"] for n in active_ops if n.get("tier") == "secondary"]

    mainline = active_m[0] if active_m else (state.get("mainline") if active_ops else None)
    secondary = active_s[0] if active_s else (state.get("secondary") if active_ops else None)

    # Budget & deadline evaluation
    max_runtime = float(state.get("max_runtime_minutes", 240.0))
    started_at_str = state.get("started_at")
    started_dt = None
    if started_at_str:
        try:
            started_dt = datetime.fromisoformat(started_at_str.replace("Z", "+00:00"))
        except Exception:
            started_dt = None

    deadline_str = state.get("budget_deadline")
    deadline_dt = None
    if deadline_str:
        try:
            deadline_dt = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
        except Exception:
            deadline_dt = None

    if not deadline_dt and started_dt:
        deadline_dt = started_dt + timedelta(minutes=max_runtime)

    now_utc = datetime.now(timezone.utc)
    elapsed = max(0.0, (now_utc - started_dt).total_seconds() / 60.0) if started_dt else 0.0
    budget_exhausted = (deadline_dt is not None and now_utc >= deadline_dt) or (elapsed >= max_runtime)

    blocked_reason = state.get("blocked_reason")
    raw_phase = state.get("phase", "completed")
    overnight = state.get("overnight_mode", False)

    if blocked_reason:
        phase = f"blocked ({blocked_reason})"
    elif active_ops:
        phase = "exploring"
    else:
        # No active nodes in graph
        if budget_exhausted:
            phase = "completed (budget_exhausted)"
        elif not overnight:
            phase = "completed (all opportunities validated)"
        else:
            phase = "rediscovery"

    return {
        "phase": phase,
        "mainline": mainline or "None",
        "secondary": secondary or "None",
        "elapsed_minutes": elapsed,
        "max_runtime_minutes": max_runtime,
        "budget_exhausted": budget_exhausted,
    }


def generate() -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(REPORTS_DIR, f"morning_report_{stamp}.md")

    state = load_json(STATE_FILE, {})
    graph = load_json(GRAPH_FILE, {"nodes": {}})
    nodes = graph.get("nodes", {}) if isinstance(graph, dict) else {}

    session_id = state.get("session_id", "session_default")
    header_info = resolve_lifecycle_header(state, graph)

    all_integrated_ops = [n for n in nodes.values() if n.get("status") == "INTEGRATED"]
    all_candidate_ops = [n for n in nodes.values() if n.get("status") == "PRODUCTION_CANDIDATE"]
    all_sandbox_ops = [n for n in nodes.values() if n.get("status") in {"VALIDATED_SANDBOX", "VALIDATED"}]
    proposed_ops = [n for n in nodes.values() if n.get("status") == "PROPOSED"]
    parked_ops = [n for n in nodes.values() if n.get("status") == "PARKED"]
    active_ops = [n for n in nodes.values() if n.get("status") in {"ACTIVATED", "EXPLORING"}]

    all_verified_bugs = load_verified_bugs()
    discoveries = read_text(DISCOVERIES_FILE)
    failed = read_text(FAILED_PATHS_FILE)

    # Session delta isolation
    session_int_ids = set(get_session_op_ids_by_target(session_id, ["INTEGRATED"]))
    session_cand_ids = set(get_session_op_ids_by_target(session_id, ["PRODUCTION_CANDIDATE"]))
    session_sand_ids = set(get_session_validated_op_ids(session_id))

    session_integrated_ops = [op for op in all_integrated_ops if op["id"] in session_int_ids or op.get("created_session_id") == session_id]
    session_candidate_ops = [op for op in all_candidate_ops if op["id"] in session_cand_ids or op.get("created_session_id") == session_id]
    session_sandbox_ops = [op for op in all_sandbox_ops if op["id"] in session_sand_ids or op.get("created_session_id") == session_id]
    session_bugs = [b for b in all_verified_bugs if b.get("session_id") == session_id]

    leverage_index = round(
        len(all_integrated_ops) * 5.0
        + len(all_candidate_ops) * 4.0
        + len(all_sandbox_ops) * 3.0
        + len(all_verified_bugs) * 4.0
        + (1.5 if discoveries != "暂无" else 0.0)
        + (1.0 if failed != "暂无" else 0.0),
        1,
    )

    lines: List[str] = []
    lines.append("# 🌅 OVERNIGHT AUTONOMOUS R&D REPORT")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**会话 ID**: `{session_id}`  ")
    lines.append(
        f"**当前状态**: Phase: `{header_info['phase']}` | Main: `{header_info['mainline']}` | Sec: `{header_info['secondary']}`  "
    )
    lines.append(
        f"**运行预算与时钟**: 已耗时 `{header_info['elapsed_minutes']:.1f}m` / 上限 `{header_info['max_runtime_minutes']:.1f}m` "
        f"({'已超时/截止' if header_info['budget_exhausted'] else '预算充足'}) [Wall-Clock Deadline Model]"
    )
    lines.extend([
        "",
        "---",
        "",
        f"## 🎯 NORTH STAR DELTA: 本会话新增交付 (Current Session Delta: `{session_id}`)",
        "",
        f"> 本节严格仅展示当前会话 `{session_id}` 内完成验证并落盘的成果，杜绝将历史总资产冒充为单夜产出。",
        "",
        f"### 1. 🛠️ 本会话修复 Bug (Bug Safety Envelope Enforced, {len(session_bugs)} 项)",
        "",
    ])

    if session_bugs:
        for bug in session_bugs:
            lines.append(
                f"- **[{bug.get('bug_id', 'BUG')}] {bug.get('title', 'Untitled')}**:"
            )
            lines.append(f"  - **复现证据**: `{bug.get('evidence', 'N/A')}`")
            lines.append(f"  - **代码变更**: `{', '.join(bug.get('files_changed', []))}`")
            lines.append(
                f"  - **回归测试**: `{bug.get('tests_run', 'N/A')}` (状态: `PASSED ✅`, 回滚: `None`)"
            )
    else:
        lines.append("- 本会话无新增 Bug 修复。")

    lines.extend([
        "",
        f"### 2. 🔗 本会话生产已合入特性 (INTEGRATED - 人工授权, {len(session_integrated_ops)} 项)",
        "",
    ])
    if session_integrated_ops:
        for op in session_integrated_ops:
            lines.append(
                f"- **[{op['id']}] {op['title']}**: {op.get('description', '')} (状态: `INTEGRATED 🔗`)"
            )
    else:
        lines.append("- 本会话无新增生产合入特性（自主运行期间严禁自授权合入）。")

    lines.extend([
        "",
        f"### 3. 🚀 本会话生产候选特性 (PRODUCTION_CANDIDATE - 等待用户接入授权, {len(session_candidate_ops)} 项)",
        "",
    ])
    if session_candidate_ops:
        for op in session_candidate_ops:
            lines.append(
                f"- **[{op['id']}] {op['title']}**: {op.get('description', '')} (状态: `PRODUCTION_CANDIDATE 🚀`)"
            )
            notes = op.get("notes", [])
            cand_notes = [n.get("note", "") for n in notes if n.get("status") == "PRODUCTION_CANDIDATE"]
            if cand_notes and cand_notes[-1]:
                lines.append(f"  - **实证审查结果**: `{cand_notes[-1]}`")
    else:
        lines.append("- 本会话无新增生产候选特性。")

    lines.extend([
        "",
        f"### 4. 📦 本会话实跑验证独立沙盒原型 (VALIDATED_SANDBOX, {len(session_sandbox_ops)} 项)",
        "",
    ])

    if session_sandbox_ops:
        for op in session_sandbox_ops:
            lines.append(
                f"- **[{op['id']}] {op['title']}**: {op.get('description', '')} (状态: `VALIDATED_SANDBOX 📦`)"
            )
            notes = op.get("notes", [])
            val_notes = [n.get("note", "") for n in notes if n.get("status") in {"VALIDATED_SANDBOX", "VALIDATED"}]
            if val_notes and val_notes[-1]:
                lines.append(f"  - **实测验证**: `{val_notes[-1]}`")
            res = op.get("external_research")
            if res and isinstance(res, dict) and res.get("searched"):
                lines.append(
                    f"  - **外部情报研判 (External Research Gate)**: 决议=`{res.get('decision', '').upper()}` | 来源=`{', '.join(res.get('sources', [])) or 'N/A'}`"
                )
                lines.append(f"    - 情报分析: {res.get('findings', 'N/A')}")
                lines.append(f"    - 选型理由: {res.get('reason', 'N/A')}")
    else:
        lines.append("- 本会话无新增独立验证原型。")

    lines.extend([
        "",
        "---",
        "",
        "## 🏛️ CUMULATIVE PROJECT ASSETS: 项目历史累计总资产 (Project Lifetime Totals)",
        "",
        f"> 包含项目启动以来全部历史会话累积沉淀（全量已合入生产: `{len(all_integrated_ops)}` 项，全量生产候选: `{len(all_candidate_ops)}` 项，全量独立沙盒原型: `{len(all_sandbox_ops)}` 项，全量已验证 Bug 修复: `{len(all_verified_bugs)}` 项）。",
        "",
        f"### 1. 🔗 全量生产已集成特性清单 ({len(all_integrated_ops)} 项)",
        "",
    ])

    if all_integrated_ops:
        for op in all_integrated_ops:
            lines.append(
                f"- **[{op['id']}] {op['title']}**: {op.get('description', '')} (状态: `INTEGRATED 🔗`)"
            )
    else:
        lines.append("- 暂无生产已合入特性。")

    lines.extend([
        "",
        f"### 2. 🚀 全量生产候选特性清单 ({len(all_candidate_ops)} 项)",
        "",
    ])
    if all_candidate_ops:
        for op in all_candidate_ops:
            lines.append(
                f"- **[{op['id']}] {op['title']}**: {op.get('description', '')} (状态: `PRODUCTION_CANDIDATE 🚀`)"
            )
    else:
        lines.append("- 暂无生产候选特性。")

    lines.extend([
        "",
        f"### 3. 📦 全量独立沙盒原型清单 ({len(all_sandbox_ops)} 项)",
        "",
    ])

    if all_sandbox_ops:
        for op in all_sandbox_ops:
            lines.append(
                f"- **[{op['id']}] {op['title']}**: {op.get('description', '')} (状态: `VALIDATED_SANDBOX 📦`)"
            )
            notes = op.get("notes", [])
            val_notes = [n.get("note", "") for n in notes if n.get("status") in {"VALIDATED_SANDBOX", "VALIDATED"}]
            if val_notes and val_notes[-1]:
                lines.append(f"  - **实测验证**: `{val_notes[-1]}`")
    else:
        lines.append("- 暂无独立验证完毕的原型。")

    lines.extend([
        "",
        f"### 4. 🛠️ 全量真实已验证 Bug 修复清单 ({len(all_verified_bugs)} 项)",
        "",
    ])

    if all_verified_bugs:
        for bug in all_verified_bugs:
            lines.append(
                f"- **[{bug.get('bug_id', 'BUG')}] {bug.get('title', 'Untitled')}** (会话: `{bug.get('session_id', 'unknown')}`):"
            )
            if bug.get("evidence"):
                lines.append(f"  - **复现证据**: `{bug.get('evidence')}`")
            if bug.get("files_changed"):
                lines.append(f"  - **代码变更**: `{', '.join(bug.get('files_changed'))}`")
    else:
        lines.append("- 暂无历史 Bug 修复。")

    lines.extend([
        "",
        "### 5. 🚀 CONFIRMED DISCOVERIES (实证有效的发现)",
        "",
        discoveries,
        "",
        "### 6. 💡 PRODUCT PROPOSALS (留给用户做选择题的候选提案)",
        "",
    ])

    if proposed_ops:
        for op in proposed_ops:
            lines.append(
                f"- **[{op['id']}] {op['title']}**: {op.get('description', '')} (状态: `PROPOSED 💡`)"
            )
    else:
        lines.append("- 暂无待决提案。")

    lines.extend(["", "### 7. 🗃️ OPPORTUNITY BACKLOG & ACTIVE (待命与活跃图谱)", ""])
    if active_ops:
        lines.append("**当前推进中**:")
        for op in active_ops:
            lines.append(
                f"- **[{op['id']}] {op['title']}** [{op.get('tier', '').upper()}] (状态: `{op['status']}`)"
            )
    if parked_ops:
        lines.append("")
        lines.append("**冷冻待命区**:")
        for op in parked_ops:
            lines.append(f"- **[{op['id']}] {op['title']}**: {op.get('description', '')}")

    lines.extend([
        "",
        "### 8. 🪦 FAILED EXPERIMENTS & KNOWLEDGE (踩坑记录与被排除的死胡同)",
        "",
        failed,
        "",
        "### 📊 TRANSPARENT LEVERAGE METRIC (双轨价值指数)",
        "",
        f"- **本会话新增交付**: 已合入 `{len(session_integrated_ops)}` | 候选 `{len(session_candidate_ops)}` | 原型 `{len(session_sandbox_ops)}` | 修复 Bug `{len(session_bugs)}`",
        f"- **历史累计总资产**: 已合入 `{len(all_integrated_ops)}` | 候选 `{len(all_candidate_ops)}` | 原型 `{len(all_sandbox_ops)}` | 修复 Bug `{len(all_verified_bugs)}`",
        f"- **累计估算省力指数 (Heuristic Leverage Index)**: `{leverage_index} pts` *(启发式指标，非机械秒表工时)*",
        "",
        "### 🧠 SELF-EVOLUTION CHECK (五维实证自我审计)",
        "",
        "- 请查阅配套生成的 `SKILL_EVOLUTION_PROPOSAL.md`，其中包含与上一运行的历史 Delta。",
        "",
        "---",
        "",
        "*Report auto-generated by overnight_report.py*",
        "",
    ])

    # Atomic-ish report generation: write temp then replace.
    temp_file = report_file + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_file, report_file)

    print(f"✅ Morning Intelligence Report generated at: {report_file}")
    return report_file


if __name__ == "__main__":
    generate()
