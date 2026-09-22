#!/usr/bin/env python3
"""
checkpoint.py - Antigravity Active State & Context Checkpoint Manager

Guarantees:
- Atomic active_state.json writes.
- Explicit, deterministic checkpoint argument semantics.
- Structured session telemetry for downstream Self-Eval.
- Session ID persistence for context recovery and history isolation.
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.environ.get("ANTIGRAVITY_MEMORY_DIR") or os.path.abspath(os.path.join(SCRIPT_DIR, "../../../memory"))
ACTIVE_STATE_FILE = os.path.join(MEMORY_DIR, "active_state.json")
SESSION_LOG_FILE = os.path.join(MEMORY_DIR, "session_log.md")
GRAPH_FILE = os.path.join(MEMORY_DIR, "opportunity_graph.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso_time(ts_str: Optional[str]) -> Optional[datetime]:
    if not ts_str:
        return None
    try:
        clean = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(clean)
    except (ValueError, TypeError):
        return None


def get_budget_status(state: Dict[str, Any]) -> Tuple[bool, float, float]:
    """
    Returns (budget_remains, elapsed_minutes, max_runtime_minutes).

    WALL-CLOCK DEADLINE MODEL:
    - started_at + max_runtime_minutes = budget_deadline.
    - If now_utc >= budget_deadline or elapsed_minutes >= max_runtime_minutes,
      budget is strictly exhausted regardless of intervening pauses.
    - LIMITATION NOTE: This evaluation is passive; the Python runtime does not
      actively self-awaken if host execution is suspended or waiting for input.
    """
    max_runtime = float(state.get("max_runtime_minutes", 240.0))
    started_at_str = state.get("started_at")
    started_dt = parse_iso_time(started_at_str)
    if not started_dt:
        return False, 0.0, max_runtime

    deadline_str = state.get("budget_deadline")
    deadline_dt = parse_iso_time(deadline_str)
    if not deadline_dt and state.get("overnight_mode", False):
        deadline_dt = started_dt + timedelta(minutes=max_runtime)

    now_utc = datetime.now(timezone.utc)
    elapsed = max(0.0, (now_utc - started_dt).total_seconds() / 60.0)
    if deadline_dt is None:
        return False, elapsed, max_runtime
    budget_remains = (now_utc < deadline_dt) and (elapsed < max_runtime)
    return budget_remains, elapsed, max_runtime


def init_memory_dir() -> None:
    os.makedirs(MEMORY_DIR, exist_ok=True)


def atomic_write_json(path: str, data: Any) -> None:
    init_memory_dir()
    temp_file = path + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_file, path)


def get_state() -> Dict[str, Any]:
    init_memory_dir()
    if not os.path.exists(ACTIVE_STATE_FILE):
        return {
            "session_id": "session_default",
            "phase": "recon",
            "mainline": None,
            "secondary": None,
            "current_action": "initial_reconnaissance",
            "last_completed_action": None,
            "blocked_reason": None,
            "next_action": "read_project_codebase",
            "updated_at": utc_now(),
        }
    with open(ACTIVE_STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: Dict[str, Any], log_msg: Optional[str] = None, event: Optional[str] = None) -> Dict[str, Any]:
    init_memory_dir()
    state["updated_at"] = utc_now()
    atomic_write_json(ACTIVE_STATE_FILE, state)
    if log_msg is not None:
        append_event(
            event or "CHECKPOINT",
            log_msg,
            phase=state.get("phase", "recon"),
            session_id=state.get("session_id", "session_default"),
        )
    return state


def append_event(
    event: str,
    message: str,
    phase: str = "recon",
    session_id: Optional[str] = None,
) -> None:
    init_memory_dir()
    sess = session_id or get_state().get("session_id", "session_default")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{now_str}] [{sess}] [{phase.upper()}] [EVENT={event}] {message}\n"
    with open(SESSION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)


def append_log(message: str, phase: str = "recon", session_id: Optional[str] = None) -> None:
    append_event("NOTE", message, phase=phase, session_id=session_id)


def active_branches_from_graph() -> Dict[str, Optional[str]]:
    """Reconcile active branch state from the Opportunity Graph at session boundaries."""
    if not os.path.exists(GRAPH_FILE):
        return {"mainline": None, "secondary": None}
    try:
        with open(GRAPH_FILE, "r", encoding="utf-8") as f:
            graph = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"mainline": None, "secondary": None}

    result: Dict[str, Optional[str]] = {"mainline": None, "secondary": None}
    for node in graph.get("nodes", {}).values():
        if node.get("status") not in {"ACTIVATED", "EXPLORING"}:
            continue
        tier = node.get("tier")
        if tier == "main" and result["mainline"] is None:
            result["mainline"] = node.get("id")
        elif tier == "secondary" and result["secondary"] is None:
            result["secondary"] = node.get("id")
    return result


def start_session(
    session_id: str,
    overnight_mode: bool = False,
    max_runtime_minutes: int = 240,
    no_progress_limit: int = 3,
) -> None:
    if not session_id or session_id.strip() != session_id:
        raise SystemExit("❌ session_id must be non-empty and must not contain surrounding whitespace.")
    state = get_state()
    prev_session_id = state.get("session_id")

    # Guardrail: Active overnight session cannot be demoted or replaced before deadline
    deadline_str = state.get("budget_deadline")
    deadline_dt = parse_iso_time(deadline_str)
    now_utc = datetime.now(timezone.utc)
    budget_unexpired = (deadline_dt is not None) and (now_utc < deadline_dt)
    is_active_overnight = state.get("overnight_mode", False) or budget_unexpired

    if is_active_overnight and not overnight_mode:
        budget_remains, elapsed, max_rt = get_budget_status(state)
        if budget_remains and not state.get("blocked_reason"):
            err_msg = (
                f"❌ REFUSED: active overnight session '{prev_session_id}' cannot be demoted to interactive "
                f"before deadline ({elapsed:.1f}m / {max_rt:.1f}m, deadline: {deadline_str or 'none'})."
            )
            print(err_msg, file=sys.stderr)
            raise SystemExit(1)

    # Clean session boundary: do not inherit stale started_at across distinct sessions
    if prev_session_id != session_id or overnight_mode or not state.get("started_at"):
        state["started_at"] = utc_now()

    start_dt = parse_iso_time(state.get("started_at")) or datetime.now(timezone.utc)
    if overnight_mode:
        deadline_dt = start_dt + timedelta(minutes=float(max_runtime_minutes))
        state["budget_deadline"] = deadline_dt.isoformat()
    else:
        state["budget_deadline"] = None

    state["session_id"] = session_id
    state["overnight_mode"] = overnight_mode
    state["max_runtime_minutes"] = max_runtime_minutes
    state["consecutive_no_progress_count"] = 0
    state["no_progress_limit"] = no_progress_limit
    state["blocked_reason"] = None
    active = active_branches_from_graph()
    state["mainline"] = active["mainline"]
    state["secondary"] = active["secondary"]
    save_state(state)
    append_event(
        "SESSION_START",
        f"session={session_id} overnight={overnight_mode} budget={max_runtime_minutes}m deadline={state.get('budget_deadline') or 'none'} mainline={state.get('mainline') or 'none'} secondary={state.get('secondary') or 'none'}",
        phase=state.get("phase", "recon"),
        session_id=session_id,
    )
    print(f"🎬 Session started: '{session_id}' (overnight={overnight_mode}, budget={max_runtime_minutes}m, deadline={state.get('budget_deadline')}) recorded in state and log.")


def set_checkpoint(
    phase: str,
    mainline: str,
    secondary: str,
    current_action: str,
    next_action: str,
    log_msg: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    state = get_state()

    # ── Programmatic Guardrail: Mandatory Rediscovery on Empty Graph ──────
    # "All current opportunities completed" != "R&D space exhausted".
    # Intercept premature session completion if overnight mode is active, budget remains,
    # and no explicit unresolvable blocker exists.
    deadline_str = state.get("budget_deadline")
    deadline_dt = parse_iso_time(deadline_str)
    now_utc = datetime.now(timezone.utc)
    budget_unexpired = (deadline_dt is not None) and (now_utc < deadline_dt)
    is_overnight = state.get("overnight_mode", False) or budget_unexpired

    if phase.lower() in ("complete", "completed", "done", "stopped", "finalized", "closed", "finish") and is_overnight:
        budget_remains, elapsed, max_rt = get_budget_status(state)
        blocked_reason = state.get("blocked_reason")
        if budget_remains and not blocked_reason:
            intercept_msg = (
                f"⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='{phase}' while overnight budget remains "
                f"({elapsed:.1f}m / {max_rt:.1f}m). Mandating Rediscovery Phase instead of early exit."
            )
            print(intercept_msg, file=sys.stderr)
            phase = "discovery"
            current_action = "rediscover_opportunities_after_batch_completion"
            next_action = "observe_and_discover_next_opportunity"
            log_msg = f"{log_msg or ''} [{intercept_msg}]".strip()

    state["phase"] = phase
    state["mainline"] = None if mainline == "none" else mainline
    state["secondary"] = None if secondary == "none" else secondary
    if session_id is not None:
        state["session_id"] = session_id
    state["last_completed_action"] = state.get("current_action")
    state["current_action"] = current_action
    state["next_action"] = next_action

    message = log_msg or f"current_action={current_action} next_action={next_action}"
    save_state(state, message, event="CHECKPOINT")
    print(
        f"Checkpoint saved: Phase={state['phase']}, Main={state['mainline']}, "
        f"Sec={state['secondary']}, Action={state['current_action']}"
    )


def set_checkpoint_legacy(
    phase: str,
    mainline: str,
    current_action: str,
    next_action: str,
    log_msg: Optional[str] = None,
) -> None:
    """Explicit legacy command; no heuristic argument guessing is used."""
    set_checkpoint(phase, mainline, "none", current_action, next_action, log_msg)


def end_overnight(force: bool = False) -> None:
    state = get_state()
    deadline_str = state.get("budget_deadline")
    deadline_dt = parse_iso_time(deadline_str)
    now_utc = datetime.now(timezone.utc)
    budget_unexpired = (deadline_dt is not None) and (now_utc < deadline_dt)
    is_overnight = state.get("overnight_mode", False) or budget_unexpired

    if is_overnight:
        budget_remains, elapsed, max_rt = get_budget_status(state)
        blocked_reason = state.get("blocked_reason")
        if budget_remains and not blocked_reason:
            err_msg = (
                f"❌ REFUSED: autonomous overnight session cannot finalize before deadline "
                f"(elapsed: {elapsed:.1f}m / {max_rt:.1f}m, deadline: {deadline_str or 'none'}).\n"
                f"Rule: Zero-Bypass Invariance: no self-finalization allowed in overnight mode before deadline.\n"
                f"Autonomous agent must continue discovery, exploration, and benchmarking until budget deadline."
            )
            print(err_msg, file=sys.stderr)
            raise SystemExit(1)

    state["overnight_mode"] = False
    state["budget_deadline"] = None
    save_state(state, "Overnight mode ended explicitly.", event="SESSION_END")


def reconcile_active_state(state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Bi-directional synchronization: Reconcile active_state.json with opportunity_graph.json.
    Ensures active_state.json does not retain stale mainline/secondary pointers or
    conflicting phases when graph nodes are validated/parked/activated.
    """
    if state is None:
        state = get_state()
    active = active_branches_from_graph()
    state["mainline"] = active["mainline"]
    state["secondary"] = active["secondary"]

    has_active_nodes = (active["mainline"] is not None) or (active["secondary"] is not None)
    current_phase = state.get("phase", "recon")

    if has_active_nodes:
        if current_phase in ("completed", "done", "discovery", "rediscovery", "recon"):
            state["phase"] = "exploring"
    else:
        # No nodes currently active in graph
        budget_remains, elapsed, max_rt = get_budget_status(state)
        deadline_str = state.get("budget_deadline")
        deadline_dt = parse_iso_time(deadline_str)
        now_utc = datetime.now(timezone.utc)
        budget_unexpired = (deadline_dt is not None) and (now_utc < deadline_dt)
        is_overnight = state.get("overnight_mode", False) or budget_unexpired

        if is_overnight and budget_remains and not state.get("blocked_reason"):
            state["phase"] = "rediscovery"
            state["current_action"] = "rediscover_opportunities_after_batch_completion"
            state["next_action"] = "observe_and_discover_next_opportunity"
        else:
            state["phase"] = "completed"
            state["current_action"] = "all_opportunities_validated"
            state["next_action"] = "morning_report_delivered"

    save_state(state, "Reconciled active state with opportunity graph.", event="STATE_RECONCILE")
    return state


def finalize_session(force: bool = False, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Canonical Shutdown Sequence (Zero-Bypass Invariance):
    1. end_overnight(force=force) strictly verifies budget exhaustion or unresolvable blocker.
    2. Atomically clear mainline/secondary, budget_deadline, and set phase='completed'.
    3. Persist state and append SESSION_FINALIZED event.
    """
    end_overnight(force=force)
    state = get_state()
    if session_id:
        state["session_id"] = session_id
    state["phase"] = "completed"
    state["mainline"] = None
    state["secondary"] = None
    state["last_completed_action"] = state.get("current_action")
    state["current_action"] = "session_finalized"
    state["next_action"] = "none"
    state["overnight_mode"] = False
    state["budget_deadline"] = None
    save_state(state, "Session finalized via canonical shutdown sequence.", event="SESSION_FINALIZED")
    print(f"🏁 Session '{state.get('session_id')}' finalized: phase=completed, overnight=False.")
    return state


def print_usage() -> None:
    print(
        "Usage:\n"
        "  checkpoint.py get\n"
        "  checkpoint.py start_session <session_id> [--overnight] [max_runtime_minutes] [no_progress_limit]\n"
        "  checkpoint.py end_overnight\n"
        "  checkpoint.py finalize [session_id]\n"
        "  checkpoint.py reconcile\n"
        "  checkpoint.py set <phase> <mainline> <secondary> <current_action> <next_action> [log_msg] [session_id]\n"
        "  checkpoint.py set-legacy <phase> <mainline> <current_action> <next_action> [log_msg]\n"
        "  checkpoint.py log <message...>"
    )


def main() -> None:
    if len(sys.argv) < 2:
        print_usage()
        raise SystemExit(1)

    cmd = sys.argv[1]
    if cmd == "get":
        print(json.dumps(get_state(), indent=2, ensure_ascii=False))
        return

    if cmd == "start_session":
        if len(sys.argv) < 3:
            print_usage()
            raise SystemExit(1)
        sess_id = sys.argv[2]
        is_overnight = "--overnight" in sys.argv
        args = [a for a in sys.argv[3:] if a != "--overnight"]
        runtime_min = int(args[0]) if len(args) >= 1 else 240
        progress_lim = int(args[1]) if len(args) >= 2 else 3
        start_session(sess_id, overnight_mode=is_overnight, max_runtime_minutes=runtime_min, no_progress_limit=progress_lim)
        return

    if cmd == "end_overnight":
        force = "--force" in sys.argv
        end_overnight(force=force)
        return

    if cmd == "finalize":
        force = "--force" in sys.argv
        args = [a for a in sys.argv[2:] if a != "--force"]
        sess_id = args[0] if args else None
        finalize_session(force=force, session_id=sess_id)
        return

    if cmd == "reconcile":
        reconciled = reconcile_active_state()
        print(json.dumps(reconciled, indent=2, ensure_ascii=False))
        return

    if cmd == "set":
        # Exactly five required positional arguments after `set`; no content-based guessing.
        if not (7 <= len(sys.argv) <= 9):
            print_usage()
            raise SystemExit(1)
        phase, mainline, secondary, current_action, next_action = sys.argv[2:7]
        log_msg = sys.argv[7] if len(sys.argv) >= 8 else None
        sess_id = sys.argv[8] if len(sys.argv) == 9 else None
        set_checkpoint(phase, mainline, secondary, current_action, next_action, log_msg, sess_id)
        return

    if cmd == "set-legacy":
        if not (6 <= len(sys.argv) <= 7):
            print_usage()
            raise SystemExit(1)
        phase, mainline, current_action, next_action = sys.argv[2:6]
        log_msg = sys.argv[6] if len(sys.argv) == 7 else None
        set_checkpoint_legacy(phase, mainline, current_action, next_action, log_msg)
        return

    if cmd == "log":
        if len(sys.argv) < 3:
            print_usage()
            raise SystemExit(1)
        msg = " ".join(sys.argv[2:])
        state = get_state()
        append_log(msg, state.get("phase", "recon"), state.get("session_id", "session_default"))
        print(f"Logged: {msg}")
        return

    print_usage()
    raise SystemExit(1)


if __name__ == "__main__":
    main()
