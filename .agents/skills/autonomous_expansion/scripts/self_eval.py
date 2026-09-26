#!/usr/bin/env python3
"""
self_eval.py - Five-Dimensional Empirical Self-Audit Harness

This version consumes structured telemetry rather than inferring branch switches
from free-form phrases such as "Transitioned to" or "Activated".
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../memory"))
REPORTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../reports/overnight"))
GRAPH_FILE = os.path.join(MEMORY_DIR, "opportunity_graph.json")
STATE_FILE = os.path.join(MEMORY_DIR, "active_state.json")
LOG_FILE = os.path.join(MEMORY_DIR, "session_log.md")
DISCOVERIES_FILE = os.path.join(MEMORY_DIR, "discoveries.md")
FAILED_PATHS_FILE = os.path.join(MEMORY_DIR, "failed_paths.md")
FAILED_PATHS_JSON = os.path.join(MEMORY_DIR, "failed_paths.json")
BUGS_FILE = os.path.join(MEMORY_DIR, "bugs_resolved.json")
EVOLUTION_HISTORY_FILE = os.path.join(MEMORY_DIR, "evolution_history.json")
EVOLUTION_PROPOSAL_FILE = os.path.join(REPORTS_DIR, "SKILL_EVOLUTION_PROPOSAL.md")

EVENT_RE = re.compile(
    r"^\[(?P<timestamp>[^\]]+)\] \[(?P<session>[^\]]+)\] "
    r"\[(?P<phase>[^\]]+)\] \[EVENT=(?P<event>[A-Z0-9_]+)\] (?P<message>.*)$"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_current_skill_version() -> str:
    version_file = os.path.abspath(os.path.join(SCRIPT_DIR, "../VERSION"))
    if os.path.exists(version_file):
        try:
            with open(version_file, "r", encoding="utf-8") as f:
                v = f.read().strip()
                if v:
                    return v if v.startswith("v") else f"v{v}"
        except OSError:
            pass

    skill_file = os.path.abspath(os.path.join(SCRIPT_DIR, "../SKILL.md"))
    if os.path.exists(skill_file):
        try:
            with open(skill_file, "r", encoding="utf-8") as f:
                content = f.read()
                m = re.search(r"^version:\s*[\"']?([vV]?\d+(?:\.\d+)*)[\"']?", content, re.MULTILINE)
                if m:
                    v = m.group(1)
                    return v if v.startswith("v") else f"v{v}"
                m2 = re.search(r"[vV](\d+(?:\.\d+)*)", content)
                if m2:
                    return f"v{m2.group(1)}"
        except OSError:
            pass
    return "v5.1"


def atomic_write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_file = path + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_file, path)


def count_markdown_items(filepath: str) -> int:
    if not os.path.exists(filepath):
        return 0
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith(("- ", "* ")):
                if "暂无" not in stripped and "placeholder" not in stripped.lower():
                    count += 1
            elif re.match(r"^\d+\.\s+", stripped):
                count += 1
    return count


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


def load_verified_pitfalls() -> List[Dict[str, Any]]:
    data = load_json(FAILED_PATHS_JSON, [])
    if isinstance(data, dict):
        data = data.get("pitfalls", [])
    if not isinstance(data, list):
        return []
    return [
        p
        for p in data
        if isinstance(p, dict)
        and p.get("id")
        and p.get("evidence")
        and p.get("action_taken")
        and p.get("result")
    ]


def load_evolution_history() -> List[Dict[str, Any]]:
    data = load_json(EVOLUTION_HISTORY_FILE, [])
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        runs = data.get("runs", [])
        return runs if isinstance(runs, list) else []
    return []


def get_active_baseline_run(history: List[Dict[str, Any]], current_version: str) -> Optional[Dict[str, Any]]:
    # Search backwards for the last run that was ACTIVE_BASELINE or CONFIRMED from a prior version
    for run in reversed(history):
        if run.get("status") in {"ACTIVE_BASELINE", "CONFIRMED", "BASELINE_INITIALIZED"}:
            if run.get("version") != current_version:
                return run
    for run in reversed(history):
        if run.get("version") != current_version:
            return run
    return history[0] if history else None


def save_evolution_run(record: Dict[str, Any]) -> None:
    history = load_evolution_history()
    history.append(record)
    atomic_write_json(EVOLUTION_HISTORY_FILE, history)


def load_session_context() -> Tuple[Dict[str, Any], str]:
    state = load_json(STATE_FILE, {})
    session_id = state.get("session_id", "session_default") if isinstance(state, dict) else "session_default"
    return state if isinstance(state, dict) else {}, session_id


def parse_events(session_id: Optional[str] = None) -> List[Dict[str, str]]:
    if not os.path.exists(LOG_FILE):
        return []
    events: List[Dict[str, str]] = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            match = EVENT_RE.match(line.strip())
            if not match:
                continue
            if session_id is not None and match.group("session") != session_id:
                continue
            item = match.groupdict()
            events.append(item)
    return events


def normalized_action(message: str) -> str:
    text = re.sub(r"\d+", "", message).strip().lower()
    return re.sub(r"\s+", " ", text)


def audit() -> Dict[str, Any]:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MEMORY_DIR, exist_ok=True)

    state, curr_session_id = load_session_context()
    events = parse_events(curr_session_id)
    all_events = parse_events(None)

    graph = load_json(GRAPH_FILE, {"nodes": {}})
    nodes = graph.get("nodes", {}) if isinstance(graph, dict) else {}
    total_ops = len(nodes)
    validated_ops = sum(
        1 for node in nodes.values()
        if node.get("status") in {"VALIDATED_SANDBOX", "VALIDATED", "PRODUCTION_CANDIDATE", "INTEGRATED"}
    )
    by_status: Dict[str, int] = {}
    for node in nodes.values():
        status = node.get("status", "PARKED")
        by_status[status] = by_status.get(status, 0) + 1
    conversion_rate = round((validated_ops / total_ops * 100), 1) if total_ops else 0.0

    if total_ops == 0:
        divergence_diag = "No opportunities generated yet (idle or recon phase)"
    elif total_ops > 15 and validated_ops == 0:
        divergence_diag = "⚠️ Severe Over-Divergence: Excessive ideas without landing prototypes"
    elif conversion_rate >= 20.0 or validated_ops >= 1:
        divergence_diag = "✅ Healthy: Solid idea-to-validation conversion"
    else:
        divergence_diag = "⚡ Balanced: Backlog accumulating, focus on closing current mainline"

    # Branch telemetry is now explicit. Only BRANCH_SWITCHED counts as a switch.
    session_switches = sum(1 for e in events if e["event"] == "BRANCH_SWITCHED")
    lifetime_switches = sum(1 for e in all_events if e["event"] == "BRANCH_SWITCHED")
    branch_activations = sum(1 for e in events if e["event"] == "BRANCH_ACTIVATED")

    if session_switches > 4:
        conv_diag = "⚠️ Butterfly Hopping: High explicit branch switching in current session. Stricter Momentum Lock needed."
    elif session_switches >= 1:
        conv_diag = "✅ Controlled Evolution: Explicit branch switches were recorded and remained observable."
    else:
        conv_diag = "⚡ Focused Execution: No explicit branch-switch events recorded in current session."

    # Memory quality: use structured action telemetry, not generic prose lines.
    action_events = [e for e in events if e["event"] in {"CHECKPOINT", "ACTION_STARTED", "ACTION_COMPLETED"}]
    action_counts: Dict[str, int] = {}
    for event in action_events:
        norm = normalized_action(event["message"])
        if len(norm) > 10:
            action_counts[norm] = action_counts.get(norm, 0) + 1
    redundant_count = sum(count - 2 for count in action_counts.values() if count > 2)

    state_valid = bool(state.get("phase") and state.get("updated_at"))
    if not action_events:
        memory_fidelity = None
        mem_diag = "⚪ No structured action events recorded; memory fidelity is not yet measurable."
    else:
        total_actions = len(action_events)
        memory_fidelity = round(max(0.0, (1.0 - redundant_count / total_actions) * 100), 1)
        if redundant_count > 0:
            mem_diag = f"⚠️ Amnesia Alert: {redundant_count} redundant action loop(s) detected in current session."
        else:
            mem_diag = "✅ No redundant action loops detected in current session."

    confirmed_discoveries = count_markdown_items(DISCOVERIES_FILE)
    verified_bugs = load_verified_bugs()
    verified_pitfalls = load_verified_pitfalls()

    # Session-anchored deliverables (marginal yield, avoiding cumulative asset confusion)
    session_bugs = [b for b in verified_bugs if b.get("session_id") == curr_session_id]
    session_pitfalls = [p for p in verified_pitfalls if p.get("observed_in_session") == curr_session_id]
    
    session_validated_ops = 0
    for node in nodes.values():
        if node.get("status") in {"VALIDATED_SANDBOX", "VALIDATED", "PRODUCTION_CANDIDATE", "INTEGRATED"}:
            if node.get("created_session_id") == curr_session_id:
                session_validated_ops += 1
            else:
                for note in node.get("notes", []):
                    if note.get("status") in {"VALIDATED_SANDBOX", "VALIDATED", "PRODUCTION_CANDIDATE", "INTEGRATED"} and curr_session_id in str(note.get("note", "")):
                        session_validated_ops += 1
                        break

    session_yield = round(
        (session_validated_ops * 3.0)
        + (len(session_bugs) * 4.0)
        + (len(session_pitfalls) * 1.0),
        1,
    )

    cumulative_leverage = round(
        (validated_ops * 3.0)
        + (len(verified_bugs) * 4.0)
        + (confirmed_discoveries * 1.5)
        + (len(verified_pitfalls) * 1.0),
        1,
    )

    # True historical comparison against the previous recorded run.
    current_version = get_current_skill_version()
    history = load_evolution_history()
    if not history:
        evo_status = "ACTIVE_BASELINE"
        evo_verdict = f"{current_version} Initial Baseline registered. Historical deltas will benchmark subsequent sessions."
        next_rule_proposal = f"Maintain {current_version} Baseline rules until the next overnight run completes."
        delta_summary = {
            "comparison": "None (first baseline session)",
            "delta_session_switches": 0,
            "delta_memory_fidelity": 0.0,
            "delta_session_yield": 0.0,
        }
        comp_baseline = {
            "version": current_version,
            "session": "None (initial run)",
        }
    else:
        last_run = history[-1]
        last_version = last_run.get("version", current_version)
        last_session = last_run.get("session_id", "")
        last_status = last_run.get("status", "ACTIVE_BASELINE")
        last_metrics = last_run.get("metrics", {})

        baseline_run = get_active_baseline_run(history, current_version)
        if baseline_run:
            base_version = baseline_run.get("version", current_version)
            base_session = baseline_run.get("session_id", "")
            base_metrics = baseline_run.get("metrics", {})
        else:
            base_version = last_version
            base_session = last_session
            base_metrics = last_metrics

        comp_baseline = {
            "version": base_version,
            "session": base_session,
        }

        previous_switches = last_metrics.get("session_switches", 0)
        previous_fidelity = last_metrics.get("memory_fidelity")
        previous_session_yield = last_metrics.get("session_yield", 0.0)

        current_fidelity_num = memory_fidelity if memory_fidelity is not None else previous_fidelity if previous_fidelity is not None else 100.0
        delta_switches = session_switches - previous_switches
        delta_fidelity = round(current_fidelity_num - float(previous_fidelity if previous_fidelity is not None else 100.0), 1)
        delta_session_yield = round(session_yield - float(previous_session_yield), 1)
        delta_summary = {
            "compared_to_timestamp": last_run.get("timestamp"),
            "compared_to_version": base_version,
            "compared_to_session": base_session,
            "delta_session_switches": delta_switches,
            "delta_memory_fidelity": f"{delta_fidelity}%",
            "delta_session_yield": f"{delta_session_yield} pts",
        }

        is_candidate = (current_version != base_version) or (last_status == "PENDING_EVALUATION")
        is_same_session = (last_session == curr_session_id)

        # 1. Did the candidate run without a distinct operational session?
        if is_candidate and is_same_session:
            evo_status = "PENDING_EVALUATION"
            evo_verdict = (
                f"Candidate version {current_version} active in current session ({curr_session_id}). "
                f"Status: PENDING_EVALUATION. Awaiting next independent session run under {current_version} "
                f"governance to evaluate empirical delta against baseline {base_version}."
            )
            next_rule_proposal = (
                f"Operate the next overnight session under candidate {current_version} rules. "
                f"Confirmation requires verified session yield, disciplined branch switches (<=3), and zero regression."
            )
        elif is_candidate:
            # Evaluating candidate version in an independent session:
            # Check regression:
            if delta_switches > 3 or delta_fidelity < -15.0:
                evo_status = "REGRESSION"
                evo_verdict = (
                    f"⚠️ Regression detected under candidate {current_version}: "
                    f"branch switches increased by {delta_switches} or memory fidelity dropped by {delta_fidelity}%."
                )
                next_rule_proposal = f"ROLLBACK: Discard candidate {current_version} and restore baseline {base_version}."
            elif session_yield > 0 and delta_fidelity >= 0.0 and delta_switches <= 3:
                evo_status = "CONFIRMED"
                evo_verdict = (
                    f"✅ Confirmed Improvement across versions ({base_version} -> {current_version}): "
                    f"candidate achieved verified session yield (+{session_yield} pts) with disciplined execution and stable memory."
                )
                next_rule_proposal = f"PROPOSAL: Promote candidate {current_version} to new Frozen Baseline."
            else:
                evo_status = "INCONCLUSIVE"
                evo_verdict = (
                    f"⚪ Inconclusive: Data insufficient in candidate session ({current_version}) to prove superiority or degradation. "
                    f"Session yield is {session_yield} pts. Cannot determine evolution without substantive task deliverables."
                )
                next_rule_proposal = f"Maintain {current_version} in PENDING_EVALUATION status until sufficient task evidence is collected."
        else:
            # Operating on established baseline
            evo_status = "ACTIVE_BASELINE"
            evo_verdict = f"Operating normally under {current_version} baseline within expected variance."
            next_rule_proposal = f"Continue operating under {current_version} baseline."

    report = {
        "divergence_quality": {
            "total_generated_lifetime": total_ops,
            "status_breakdown": by_status,
            "validation_rate_lifetime": f"{conversion_rate}%",
            "validation_rate_num": conversion_rate,
            "diagnosis": divergence_diag,
        },
        "convergence_balance": {
            "current_session_id": curr_session_id,
            "current_session_switches": session_switches,
            "current_session_branch_activations": branch_activations,
            "lifetime_switches": lifetime_switches,
            "current_session_structured_events": len(events),
            "diagnosis": conv_diag,
        },
        "memory_quality": {
            "active_state_intact": state_valid,
            "mainline": state.get("mainline"),
            "secondary": state.get("secondary"),
            "current_action": state.get("current_action"),
            "next_action": state.get("next_action"),
            "session_redundant_loops": redundant_count,
            "memory_fidelity_score": f"{memory_fidelity}%" if memory_fidelity is not None else "N/A",
            "memory_fidelity_num": memory_fidelity,
            "diagnosis": mem_diag,
        },
        "execution_value": {
            "session_yield": f"{session_yield} pts",
            "session_prototypes": session_validated_ops,
            "session_bugs_killed": len(session_bugs),
            "session_anchored_pitfalls": len(session_pitfalls),
            "cumulative_prototypes": validated_ops,
            "cumulative_bugs_killed": len(verified_bugs),
            "cumulative_discoveries": confirmed_discoveries,
            "cumulative_anchored_pitfalls": len(verified_pitfalls),
            "cumulative_effort_avoidance_index": f"{cumulative_leverage} pts",
            "metric_transparency": "Session yield measures marginal deliverables with test/event anchors; cumulative index measures total project assets.",
            "diagnosis": f"Current session produced {session_validated_ops} prototype(s), {len(session_bugs)} bug fix(es), {len(session_pitfalls)} anchored pitfall(s)",
        },
        "evolution_check": {
            "current_version": current_version,
            "candidate_status": evo_status,
            "comparison_baseline": comp_baseline,
            "verdict": evo_verdict,
            "historical_delta": delta_summary,
            "rule_adjustment_recommendation": next_rule_proposal,
        },
        "operational_metadata": {
            "opportunity_count": total_ops,
            "action_count": len(action_events),
            "event_count": len(events),
            "note": "Operational counts recorded for future task complexity normalization.",
        },
    }

    run_record = {
        "timestamp": utc_now(),
        "session_id": curr_session_id,
        "version": current_version,
        "status": evo_status,
        "metrics": {
            "session_yield": session_yield,
            "session_switches": session_switches,
            "session_redundant_loops": redundant_count,
            "memory_fidelity": memory_fidelity,
            "validation_rate": conversion_rate,
            "cumulative_leverage": cumulative_leverage,
            "session_bugs": len(session_bugs),
            "session_prototypes": session_validated_ops,
            "session_pitfalls": len(session_pitfalls),
        },
        "operational_metadata": {
            "opportunity_count": total_ops,
            "action_count": len(action_events),
            "event_count": len(events),
        },
    }
    save_evolution_run(run_record)

    print("══════════════════════════════════════════════════════════════")
    print(f" 📊 FIVE-DIMENSIONAL EMPIRICAL AUDIT REPORT ({current_version} HARNESS)")
    print("══════════════════════════════════════════════════════════════")
    print("\n1. 自我发散质量 (Divergence Quality):")
    for key, value in report["divergence_quality"].items():
        if key != "validation_rate_num":
            print(f"   • {key}: {value}")
    print("\n2. 探索与收敛平衡 (Convergence & Momentum: Structured Telemetry):")
    for key, value in report["convergence_balance"].items():
        print(f"   • {key}: {value}")
    print("\n3. 长程记忆与恢复损耗 (Memory Quality: Session-Aware):")
    for key, value in report["memory_quality"].items():
        if key != "memory_fidelity_num":
            print(f"   • {key}: {value}")
    print("\n4. 自主执行与用户省力净价值 (Execution & Honest Value):")
    for key, value in report["execution_value"].items():
        print(f"   • {key}: {value}")
    print("\n5. 递归进化有效性回溯 (Meta-Evolution: True Historical Delta):")
    for key, value in report["evolution_check"].items():
        if key == "comparison_baseline" and isinstance(value, dict):
            print(f"   • comparison_baseline: version={value.get('version')}, session={value.get('session')}")
        else:
            print(f"   • {key}: {value}")
    print("\n══════════════════════════════════════════════════════════════")

    generate_evolution_proposal(report)
    return report


def generate_evolution_proposal(report: Dict[str, Any]) -> None:
    evo = report["evolution_check"]
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    exec_val = report["execution_value"]
    op_meta = report.get("operational_metadata", {})
    comp_base = evo.get("comparison_baseline", {})
    content = f"""# 🧬 SKILL RECURSIVE EVOLUTION PROPOSAL

**Evaluation Date**: {now_str}  
**Current Active Version**: `{evo.get('current_version', 'Unknown')}`  
**Candidate Status**: `{evo.get('candidate_status', 'Unknown')}`  
**Comparison Baseline**: `{comp_base.get('version', 'None')}` (Session: `{comp_base.get('session', 'None')}`)  
**Telemetry Verdict**: {evo['verdict']}

---

## 1. 📊 Empirical Telemetry Input

- **Session Marginal Yield**: `{exec_val.get('session_yield', '0.0 pts')}` (Prototypes: `{exec_val.get('session_prototypes', 0)}`, Bugs: `{exec_val.get('session_bugs_killed', 0)}`, Anchored Pitfalls: `{exec_val.get('session_anchored_pitfalls', 0)}`)
- **Cumulative Project Assets**: `{exec_val.get('cumulative_effort_avoidance_index', '0.0 pts')}` ({exec_val.get('metric_transparency', '')})
- **Divergence Validation Rate (Lifetime)**: `{report['divergence_quality'].get('validation_rate_lifetime', 'N/A')}` ({report['divergence_quality'].get('diagnosis', '')})
- **Current Session Branch Switches**: `{report['convergence_balance'].get('current_session_switches', 0)}` (Lifetime: `{report['convergence_balance'].get('lifetime_switches', 0)}`)
- **Memory Fidelity Score**: `{report['memory_quality'].get('memory_fidelity_score', 'N/A')}` (Redundant loops: `{report['memory_quality'].get('session_redundant_loops', 0)}`)
- **Operational Complexity Footprint**: `opportunities={op_meta.get('opportunity_count', 0)}, actions={op_meta.get('action_count', 0)}, events={op_meta.get('event_count', 0)}`

## 2. 📈 True Historical Delta (vs Previous Session)

```json
{json.dumps(evo.get('historical_delta', {}), indent=2, ensure_ascii=False)}
```

## 3. 🎯 Proposed Rule Delta

> {evo['rule_adjustment_recommendation']}

## 4. 🛡️ Verification & Promotion Guardrails

- **Promotion Requirement**: To transition from `PENDING_EVALUATION` to `CONFIRMED`, an independent operational session must achieve verified session yield > 0, explicit branch switches <= 3, and zero regression.
- **Inconclusive Handling**: If session yield is zero or task evidence is insufficient, system outputs `INCONCLUSIVE` rather than guessing.
- **Automatic Rollback Trigger**: If validation rate drops below 10% or redundant action loops exceed 2, immediately flag `REGRESSION` and revert the candidate rule.

---

*Generated empirically by self_eval.py*
"""
    with open(EVOLUTION_PROPOSAL_FILE, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    audit()
