#!/usr/bin/env python3
"""
stop_hook.py - Antigravity Lifecycle Stop Hook for Autonomous Expansion

Enforces continuous overnight R&D sessions by intercepting the Agent's attempt
to stop the execution loop. Returns:
  {"decision": "continue", "reason": "..."}
when the session is active, has remaining time budget, and has actionable work.
Returns:
  {"decision": "stop", "reason": "..."}
when time budget is exhausted, no-progress limit is reached, red-zone blocked,
or overnight mode is disabled.
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.environ.get("ANTIGRAVITY_MEMORY_DIR") or os.path.abspath(os.path.join(SCRIPT_DIR, "../../../memory"))
ACTIVE_STATE_FILE = os.path.join(MEMORY_DIR, "active_state.json")
GRAPH_FILE = os.path.join(MEMORY_DIR, "opportunity_graph.json")


def load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def save_json_atomic(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temp_file = path + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_file, path)


def parse_iso_time(ts_str: Optional[str]) -> Optional[datetime]:
    if not ts_str:
        return None
    try:
        clean = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(clean)
    except (ValueError, TypeError):
        return None


def evaluate_stop_condition(stdin_data: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    state = load_json(ACTIVE_STATE_FILE, {})
    graph = load_json(GRAPH_FILE, {"nodes": {}})
    nodes = graph.get("nodes", {}) if isinstance(graph, dict) else {}

    is_overnight = state.get("overnight_mode", False)
    deadline_str = state.get("budget_deadline")
    deadline_dt = parse_iso_time(deadline_str)
    now_utc = datetime.now(timezone.utc)
    budget_unexpired = (deadline_dt is not None) and (now_utc < deadline_dt)

    # 1. Non-overnight / Normal interactive check
    if not is_overnight and not budget_unexpired:
        return "stop", "Normal interactive session: overnight_mode is disabled."

    if budget_unexpired and not state.get("blocked_reason"):
        state_modified = False
        # Auto-heal tampered overnight_mode
        if not is_overnight:
            state["overnight_mode"] = True
            state_modified = True
        # Auto-heal tampered phase (prevent bypassing hook via phase=completed)
        if state.get("phase") in ("completed", "done", "complete", "stopped", "finalized"):
            state["phase"] = "rediscovery"
            state_modified = True
        if state_modified:
            save_json_atomic(ACTIVE_STATE_FILE, state)

    session_id = state.get("session_id", "session_unknown")

    # 2. Red-Zone Block Check
    blocked_reason = state.get("blocked_reason")
    if blocked_reason:
        return "stop", f"Overnight session paused: blocked on decision requiring user intervention ({blocked_reason})."

    # 3. Runtime Budget Check (Wall-Clock Deadline Model)
    # LIMITATION NOTE (Passive Evaluation vs Active Watchdog):
    # Python scripts in this environment are passive functions called at agent cycle
    # boundaries. If the host platform is paused, sleeping, or idle, the Python process
    # is not executing and cannot spontaneously wake up or kill the host process at 240.0m.
    # Therefore, we evaluate wall-clock elapsed time strictly upon invocation:
    # Any checkpoint, resume, stop hook, or report entry point checking after deadline
    # guarantees an immediate, deterministic 'stop' decision.
    max_runtime_minutes = float(state.get("max_runtime_minutes", 240.0))
    started_at_str = state.get("started_at")
    started_dt = parse_iso_time(started_at_str)

    now_utc = datetime.now(timezone.utc)
    elapsed_minutes = 0.0
    if started_dt:
        elapsed_minutes = max(0.0, (now_utc - started_dt).total_seconds() / 60.0)

    deadline_str = state.get("budget_deadline")
    deadline_dt = parse_iso_time(deadline_str)
    if not deadline_dt and started_dt:
        deadline_dt = started_dt + timedelta(minutes=max_runtime_minutes)

    deadline_passed = (deadline_dt is not None) and (now_utc >= deadline_dt)
    if deadline_passed or elapsed_minutes >= max_runtime_minutes:
        return "stop", (
            f"Overnight session '{session_id}' budget exhausted: "
            f"elapsed {elapsed_minutes:.1f}m >= max {max_runtime_minutes:.1f}m."
        )

    # 4. Consecutive No-Progress Anti-Loop Check
    no_progress_limit = int(state.get("no_progress_limit", 3))
    consecutive_no_progress = int(state.get("consecutive_no_progress_count", 0))

    if consecutive_no_progress >= no_progress_limit:
        return "stop", (
            f"Overnight session '{session_id}' stopped: reached consecutive no-progress limit "
            f"({consecutive_no_progress}/{no_progress_limit} cycles without new deliverables)."
        )

    # 5. Opportunity Graph Backlog Analysis
    active_nodes = [n for n in nodes.values() if n.get("status") in {"ACTIVATED", "EXPLORING"}]
    pending_nodes = [n for n in nodes.values() if n.get("status") in {"PARKED", "PROPOSED"}]
    validated_nodes = [
        n for n in nodes.values()
        if n.get("status") in {"VALIDATED_SANDBOX", "VALIDATED", "PRODUCTION_CANDIDATE", "INTEGRATED"}
    ]

    if active_nodes:
        cur_node = active_nodes[0]
        return "continue", (
            f"Overnight session '{session_id}' active ({elapsed_minutes:.1f}m / {max_runtime_minutes:.1f}m). "
            f"Branch [{cur_node.get('id')}] '{cur_node.get('title')}' is currently {cur_node.get('status')}. "
            "Continue implementation, benchmark verification, or external research."
        )

    if pending_nodes:
        next_node = pending_nodes[0]
        return "continue", (
            f"Overnight session '{session_id}' active ({elapsed_minutes:.1f}m / {max_runtime_minutes:.1f}m). "
            f"Previous branch validated. Backlog has {len(pending_nodes)} pending opportunity/opportunities. "
            f"Next recommended target: [{next_node.get('id')}] '{next_node.get('title')}'. "
            "Activate this branch or perform External Research Scan before prototyping."
        )

    return "continue", (
        f"Overnight session '{session_id}' active ({elapsed_minutes:.1f}m / {max_runtime_minutes:.1f}m). "
        f"All {len(validated_nodes)} existing opportunities are validated. "
        "Do NOT stop early: enter the next Observation & Discovery phase. Inspect codebase, benchmark end-to-end "
        "integration, profile bottlenecks, or perform External Research on GitHub/arXiv to discover new opportunities."
    )


def main():
    stdin_data = {}
    if not sys.stdin.isatty():
        try:
            raw_input = sys.stdin.read()
            if raw_input.strip():
                stdin_data = json.loads(raw_input)
        except Exception:
            stdin_data = {}

    decision, reason = evaluate_stop_condition(stdin_data)
    output = {
        "decision": decision,
        "reason": reason,
    }
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
