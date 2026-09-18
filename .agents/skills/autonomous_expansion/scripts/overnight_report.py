#!/usr/bin/env python3
"""
overnight_report.py - Outcome-Driven Morning Intelligence Report Generator

Only evidence-backed bugs enter BUGS KILLED.
Only VALIDATED nodes enter STANDALONE PROTOTYPES.
PROPOSED nodes remain proposals; PARKED nodes remain backlog.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../memory"))
REPORTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../reports/overnight"))
STATE_FILE = os.path.join(MEMORY_DIR, "active_state.json")
GRAPH_FILE = os.path.join(MEMORY_DIR, "opportunity_graph.json")
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


def generate() -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(REPORTS_DIR, f"morning_report_{stamp}.md")

    state = load_json(STATE_FILE, {})
    graph = load_json(GRAPH_FILE, {"nodes": {}})
    nodes = graph.get("nodes", {}) if isinstance(graph, dict) else {}

    validated_ops = [n for n in nodes.values() if n.get("status") == "VALIDATED"]
    proposed_ops = [n for n in nodes.values() if n.get("status") == "PROPOSED"]
    parked_ops = [n for n in nodes.values() if n.get("status") == "PARKED"]
    active_ops = [n for n in nodes.values() if n.get("status") in {"ACTIVATED", "EXPLORING"}]

    verified_bugs = load_verified_bugs()
    discoveries = read_text(DISCOVERIES_FILE)
    failed = read_text(FAILED_PATHS_FILE)

    leverage_index = round(
        len(validated_ops) * 3.0
        + len(verified_bugs) * 4.0
        + (1.5 if discoveries != "暂无" else 0.0)
        + (1.0 if failed != "暂无" else 0.0),
        1,
    )

    lines: List[str] = []
    lines.append("# 🌅 OVERNIGHT AUTONOMOUS R&D REPORT")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(
        f"**当前状态**: Phase: `{state.get('phase', 'completed')}` | Main: `{state.get('mainline', 'N/A')}` | Sec: `{state.get('secondary', 'None')}`"
    )
    lines.extend([
        "",
        "---",
        "",
        "## 🎯 NORTH STAR DELTA: 用户因此获得了什么？",
        "",
        "> 本次长程运行只把有证据的结果视为已完成交付，严格区分验证结果与待决提案。",
        "",
        "### 1. 🛠️ BUGS KILLED (真实已验证修复清单)",
        "",
    ])

    if verified_bugs:
        for bug in verified_bugs:
            lines.append(
                f"- **[{bug.get('bug_id', 'BUG')}] {bug.get('title', 'Untitled')}**:"
            )
            lines.append(f"  - **复现证据**: `{bug.get('evidence', 'N/A')}`")
            lines.append(f"  - **代码变更**: `{', '.join(bug.get('files_changed', []))}`")
            lines.append(
                f"  - **回归测试**: `{bug.get('tests_run', 'N/A')}` (状态: `PASSED ✅`, 回滚: `None`)"
            )
    else:
        lines.append("- 暂无符合严格证据链的 Bug 修复。")

    lines.extend([
        "",
        "### 2. 🚀 CONFIRMED DISCOVERIES (实证有效的发现)",
        "",
        discoveries,
        "",
        "### 3. 📦 STANDALONE PROTOTYPES (实跑已验证的独立沙盒原型)",
        "",
    ])

    if validated_ops:
        for op in validated_ops:
            lines.append(
                f"- **[{op['id']}] {op['title']}**: {op.get('description', '')} (状态: `VALIDATED ✅`)"
            )
            notes = op.get("notes", [])
            val_notes = [n.get("note", "") for n in notes if n.get("status") == "VALIDATED"]
            if val_notes and val_notes[-1]:
                lines.append(f"  - **实测验证**: `{val_notes[-1]}`")
    else:
        lines.append("- 暂无独立验证完毕的原型。")

    lines.extend(["", "### 💡 PRODUCT PROPOSALS & CANDIDATES (留给用户做选择题的候选提案)", ""])
    if proposed_ops:
        for op in proposed_ops:
            lines.append(
                f"- **[{op['id']}] {op['title']}**: {op.get('description', '')} (状态: `PROPOSED 💡`)"
            )
    else:
        lines.append("- 暂无待决提案。")

    lines.extend(["", "### 🗃️ OPPORTUNITY BACKLOG & ACTIVE (待命与活跃图谱)", ""])
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
        "### 🪦 FAILED EXPERIMENTS & KNOWLEDGE (踩坑记录与被排除的死胡同)",
        "",
        failed,
        "",
        "### 📊 TRANSPARENT LEVERAGE METRIC (用户价值指数)",
        "",
        f"- **实证确权产出**: 原型 `{len(validated_ops)}` 个 | 修复 Bug `{len(verified_bugs)}` 个",
        f"- **估算省力指数 (Heuristic Leverage Index)**: `{leverage_index} pts` *(启发式指标，非机械秒表工时)*",
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
