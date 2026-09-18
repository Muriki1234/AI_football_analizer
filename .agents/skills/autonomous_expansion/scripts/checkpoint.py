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
from datetime import datetime, timezone
from typing import Any, Dict, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../memory"))
ACTIVE_STATE_FILE = os.path.join(MEMORY_DIR, "active_state.json")
SESSION_LOG_FILE = os.path.join(MEMORY_DIR, "session_log.md")
GRAPH_FILE = os.path.join(MEMORY_DIR, "opportunity_graph.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def start_session(session_id: str) -> None:
    if not session_id or session_id.strip() != session_id:
        raise SystemExit("❌ session_id must be non-empty and must not contain surrounding whitespace.")
    state = get_state()
    state["session_id"] = session_id
    active = active_branches_from_graph()
    state["mainline"] = active["mainline"]
    state["secondary"] = active["secondary"]
    save_state(state)
    append_event(
        "SESSION_START",
        f"session={session_id} mainline={state.get('mainline') or 'none'} secondary={state.get('secondary') or 'none'}",
        phase=state.get("phase", "recon"),
        session_id=session_id,
    )
    print(f"🎬 Session started: '{session_id}' recorded in state and log.")


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


def print_usage() -> None:
    print(
        "Usage:\n"
        "  checkpoint.py get\n"
        "  checkpoint.py start_session <session_id>\n"
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
        if len(sys.argv) != 3:
            print_usage()
            raise SystemExit(1)
        start_session(sys.argv[2])
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
