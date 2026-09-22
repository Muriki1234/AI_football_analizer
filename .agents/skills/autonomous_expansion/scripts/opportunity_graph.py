#!/usr/bin/env python3
"""
opportunity_graph.py - Opportunity Lifecycle & Branch Enforcement Engine

Core guarantees:
- Valid lifecycle transitions (including PROPOSED entry paths).
- Hard cap: 1 MAIN + 1 SECONDARY active branch.
- No ACTIVATED/EXPLORING node may use the BACKLOG tier.
- Dedup supports check mode and hard guard mode.
- Structured branch telemetry is written to session_log.md.
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../memory"))
GRAPH_FILE = os.path.join(MEMORY_DIR, "opportunity_graph.json")
STATE_FILE = os.path.join(MEMORY_DIR, "active_state.json")
SESSION_LOG_FILE = os.path.join(MEMORY_DIR, "session_log.md")

VALID_STATUSES = [
    "PARKED",
    "ACTIVATED",
    "EXPLORING",
    "VALIDATED_SANDBOX",
    "PRODUCTION_CANDIDATE",
    "INTEGRATED",
    "DISPROVED",
    "PROPOSED",
    "ARCHIVED",
]
ACTIVE_STATUSES = {"ACTIVATED", "EXPLORING"}
VALID_TIERS = {"main", "secondary"}

LEGACY_STATUS_MIGRATION = {
    "VALIDATED": "VALIDATED_SANDBOX",
}

ALLOWED_TRANSITIONS = {
    "PARKED": ["ACTIVATED", "PROPOSED", "ARCHIVED"],
    "ACTIVATED": ["EXPLORING", "PARKED", "ARCHIVED"],
    "EXPLORING": ["VALIDATED_SANDBOX", "DISPROVED", "PROPOSED", "PARKED", "ARCHIVED"],
    "VALIDATED_SANDBOX": ["PRODUCTION_CANDIDATE", "PARKED", "ARCHIVED"],
    "PRODUCTION_CANDIDATE": ["INTEGRATED", "VALIDATED_SANDBOX", "PARKED", "ARCHIVED"],
    "INTEGRATED": ["PARKED", "ARCHIVED"],
    "DISPROVED": ["ARCHIVED", "PARKED"],
    "PROPOSED": ["ACTIVATED", "PARKED", "ARCHIVED"],
    "ARCHIVED": ["PARKED"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_memory_dir() -> None:
    os.makedirs(MEMORY_DIR, exist_ok=True)


def atomic_write_json(path: str, data: Any) -> None:
    ensure_memory_dir()
    temp_file = path + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_file, path)


def load_graph() -> Dict[str, Any]:
    ensure_memory_dir()
    if not os.path.exists(GRAPH_FILE):
        return {"nodes": {}, "edges": [], "updated_at": utc_now()}
    with open(GRAPH_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("nodes", {})
    data.setdefault("edges", [])

    # Automatic Legacy Migration: Migrate legacy statuses (e.g. VALIDATED -> VALIDATED_SANDBOX)
    migrated_any = False
    for node in data.get("nodes", {}).values():
        status = node.get("status")
        if status in LEGACY_STATUS_MIGRATION:
            target = LEGACY_STATUS_MIGRATION[status]
            node["status"] = target
            node.setdefault("notes", []).append({
                "time": utc_now(),
                "status": target,
                "note": f"Legacy status '{status}' automatically migrated to '{target}'",
            })
            migrated_any = True

    if migrated_any:
        save_graph(data)

    return data


def save_graph(data: Dict[str, Any]) -> None:
    data["updated_at"] = utc_now()
    atomic_write_json(GRAPH_FILE, data)


def current_session_id() -> str:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        return state.get("session_id", "session_default")
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return "session_default"


def append_event(event: str, message: str, phase: str = "graph", session_id: Optional[str] = None) -> None:
    """Write machine-readable telemetry without relying on message text."""
    ensure_memory_dir()
    sess = session_id or current_session_id()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{sess}] [{phase.upper()}] [EVENT={event}] {message}\n"
    with open(SESSION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)


def session_branch_activations(tier: str, session_id: str) -> List[str]:
    """Return branch IDs already active or activated in this session, in chronological order.

    The session-start snapshot is included so replacing a branch that was inherited from
    a previous session is still measured as a real branch switch.
    """
    if not os.path.exists(SESSION_LOG_FILE):
        return []
    results: List[str] = []
    with open(SESSION_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if f"[{session_id}]" not in line:
                continue
            marker = f"tier={tier}"
            if marker not in line and "[EVENT=SESSION_START]" not in line:
                continue

            if "[EVENT=SESSION_START]" in line:
                key = "mainline=" if tier == "main" else "secondary="
                for token in line.split():
                    if token.startswith(key):
                        value = token.split("=", 1)[1]
                        if value != "none":
                            results.append(value)
                        break
                continue

            if "[EVENT=BRANCH_ACTIVATED]" not in line and "[EVENT=BRANCH_SWITCHED]" not in line:
                continue
            for token in line.split():
                if token.startswith("opportunity="):
                    results.append(token.split("=", 1)[1])
                    break
                if token.startswith("to="):
                    results.append(token.split("=", 1)[1])
                    break
    return results


def add_opportunity(op_id: str, title: str, desc: str = "", parent_id: Optional[str] = None) -> None:
    graph = load_graph()
    if op_id in graph["nodes"]:
        existing = graph["nodes"][op_id]
        print(f"Warning: Opportunity '{op_id}' already exists with status: {existing.get('status')}")
        return

    session_id = current_session_id()
    graph["nodes"][op_id] = {
        "id": op_id,
        "title": title,
        "description": desc,
        "status": "PARKED",
        "tier": "backlog",
        "created_at": utc_now(),
        "created_session_id": session_id,
        "notes": [],
    }

    if parent_id and parent_id in graph["nodes"]:
        graph["edges"].append({"from": parent_id, "to": op_id})

    save_graph(graph)
    append_event("OPPORTUNITY_CREATED", f"opportunity={op_id} status=PARKED")
    print(f"✅ Added Opportunity '{op_id}' ({title}) -> Status: PARKED")


def active_nodes(graph: Dict[str, Any], tier: Optional[str] = None, exclude_id: Optional[str] = None) -> List[Dict[str, Any]]:
    return [
        n
        for n in graph["nodes"].values()
        if n.get("status") in ACTIVE_STATUSES
        and (tier is None or n.get("tier") == tier)
        and n.get("id") != exclude_id
    ]


def enforce_concurrency(graph: Dict[str, Any], op_id: str, target_tier: str) -> None:
    """Single hard gate for every route entering ACTIVATED/EXPLORING."""
    normalized = str(target_tier).lower()
    if normalized not in VALID_TIERS:
        raise SystemExit(
            f"❌ Invalid active tier '{target_tier}'. ACTIVATED/EXPLORING nodes must be exactly 'main' or 'secondary'."
        )

    active = active_nodes(graph, normalized, exclude_id=op_id)
    if active:
        curr = active[0]["id"]
        label = "Mainline" if normalized == "main" else "Secondary branch"
        action = "Finish or PARK it" if normalized == "main" else "Complete or PARK it"
        raise SystemExit(
            f"❌ Branch Pruning Violation: {label} '{curr}' is already active! {action} before activating another."
        )


def activate_opportunity(op_id: str, tier: str = "main") -> None:
    transition_status(op_id, "ACTIVATED", note=f"Activated as {tier} branch", tier=tier)


def sync_active_state_on_transition(
    op_id: str,
    entering_active: bool,
    leaving_active: bool,
    tier: str,
) -> None:
    """
    Bi-directional synchronization: Ensure active_state.json reflects opportunity_graph.json
    transitions (Defect 1 Minimal Fix).
    - If node enters active: update mainline/secondary, set phase to 'exploring' if needed.
    - If node leaves active: reconcile active branches. If no active nodes remain, set phase to
      'rediscovery' (if overnight budget remains) or 'completed'.
    """
    if not os.path.exists(STATE_FILE):
        return
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return

    graph = load_graph()
    active_m = [n["id"] for n in graph.get("nodes", {}).values() if n.get("status") in ACTIVE_STATUSES and n.get("tier") == "main"]
    active_s = [n["id"] for n in graph.get("nodes", {}).values() if n.get("status") in ACTIVE_STATUSES and n.get("tier") == "secondary"]

    state["mainline"] = active_m[0] if active_m else None
    state["secondary"] = active_s[0] if active_s else None

    if entering_active:
        if state.get("phase") in ("completed", "done", "discovery", "rediscovery", "recon"):
            state["phase"] = "exploring"
        state["last_completed_action"] = state.get("current_action")
        state["current_action"] = f"exploring_{op_id}"
        state["next_action"] = f"validate_{op_id}"
    elif leaving_active:
        state["last_completed_action"] = state.get("current_action")
        if not active_m and not active_s:
            # All active nodes cleared!
            overnight = state.get("overnight_mode", False)
            blocked = state.get("blocked_reason")

            # Wall-clock deadline check
            now_utc = datetime.now(timezone.utc)
            max_rt = float(state.get("max_runtime_minutes", 240.0))
            started_at_str = state.get("started_at")
            started_dt = None
            if started_at_str:
                try:
                    clean_ts = started_at_str.replace("Z", "+00:00")
                    started_dt = datetime.fromisoformat(clean_ts)
                except Exception:
                    started_dt = None

            deadline_str = state.get("budget_deadline")
            deadline_dt = None
            if deadline_str:
                try:
                    clean_dl = deadline_str.replace("Z", "+00:00")
                    deadline_dt = datetime.fromisoformat(clean_dl)
                except Exception:
                    deadline_dt = None

            if not deadline_dt and started_dt and overnight:
                deadline_dt = started_dt + timedelta(minutes=max_rt)

            elapsed = (now_utc - started_dt).total_seconds() / 60.0 if started_dt else 0.0
            budget_remains = (deadline_dt is None or now_utc < deadline_dt) and (elapsed < max_rt)

            deadline_unexpired = (deadline_dt is not None) and (now_utc < deadline_dt) and (deadline_str is not None)
            is_overnight = overnight or deadline_unexpired

            if is_overnight and budget_remains and not blocked:
                state["phase"] = "rediscovery"
                state["current_action"] = "rediscover_opportunities_after_batch_completion"
                state["next_action"] = "observe_and_discover_next_opportunity"
            else:
                state["phase"] = "completed"
                state["current_action"] = "all_opportunities_validated"
                state["next_action"] = "morning_report_delivered"
        else:
            if state["mainline"]:
                state["current_action"] = f"exploring_{state['mainline']}"
                state["next_action"] = f"validate_{state['mainline']}"

    state["updated_at"] = utc_now()
    atomic_write_json(STATE_FILE, state)
    append_event(
        "STATE_SYNC",
        f"opportunity={op_id} mainline={state.get('mainline') or 'none'} secondary={state.get('secondary') or 'none'} phase={state.get('phase')}",
        phase=state.get("phase", "graph"),
        session_id=state.get("session_id", "session_default"),
    )


def transition_status(
    op_id: str,
    new_status: str,
    note: Optional[str] = None,
    tier: Optional[str] = None,
    user_authorized: bool = False,
) -> None:
    new_status = new_status.upper()
    if new_status in LEGACY_STATUS_MIGRATION:
        raise SystemExit(
            f"❌ Legacy Status Deprecated: '{new_status}' is legacy-only. "
            f"Use '{LEGACY_STATUS_MIGRATION[new_status]}' instead."
        )

    if new_status not in VALID_STATUSES:
        raise SystemExit(f"❌ Invalid status '{new_status}'. Must be one of: {VALID_STATUSES}")

    if new_status == "INTEGRATED":
        # Check autonomous mode from active_state.json
        is_autonomous = False
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    st = json.load(f)
                is_autonomous = bool(st.get("overnight_mode", False) or st.get("autonomous_mode", False))
            except Exception:
                pass

        if is_autonomous:
            raise SystemExit(
                "❌ Autonomous Integration Block: INTEGRATED is unconditionally forbidden during autonomous operation (overnight_mode=True).\n"
                "   Production integration may only occur through an explicitly human-initiated, non-autonomous workflow outside the overnight loop."
            )

        if not user_authorized:
            raise SystemExit(
                "❌ Unauthorized Integration Block: Transitioning to 'INTEGRATED' requires explicit human authorization (--user-authorized outside autonomous mode)."
            )

    graph = load_graph()
    if op_id not in graph["nodes"]:
        raise SystemExit(f"❌ Opportunity '{op_id}' not found.")

    node = graph["nodes"][op_id]
    old_status = node.get("status", "PARKED")
    allowed = ALLOWED_TRANSITIONS.get(old_status, [])
    if new_status not in allowed:
        raise SystemExit(
            f"❌ Illegal Lifecycle Transition: Cannot transition '{op_id}' from {old_status} ➔ {new_status}!\n"
            f"   Allowed transitions from {old_status}: {allowed}"
        )

    session_id = current_session_id()
    was_active = old_status in ACTIVE_STATUSES
    entering_active = new_status in ACTIVE_STATUSES and not was_active
    old_tier = node.get("tier", "backlog")

    if new_status in ACTIVE_STATUSES:
        effective_tier = (tier or node.get("tier", "")).lower()
        if effective_tier not in VALID_TIERS:
            raise SystemExit(
                "❌ Active branch requires an explicit valid tier: 'main' or 'secondary'. "
                "'backlog' and unknown tiers cannot be ACTIVATED/EXPLORING."
            )
        enforce_concurrency(graph, op_id, effective_tier)
        node["tier"] = effective_tier
        if "activated_at" not in node:
            node["activated_at"] = utc_now()
    else:
        node["tier"] = "backlog"

    node["status"] = new_status
    if note:
        node.setdefault("notes", []).append({"time": utc_now(), "status": new_status, "note": note})

    save_graph(graph)

    # Structured telemetry: only these branch events are used for switch counting.
    if entering_active:
        prior = session_branch_activations(node["tier"], session_id)
        if prior and prior[-1] != op_id:
            append_event(
                "BRANCH_SWITCHED",
                f"tier={node['tier']} from={prior[-1]} to={op_id}",
                session_id=session_id,
            )
        else:
            append_event(
                "BRANCH_ACTIVATED",
                f"tier={node['tier']} opportunity={op_id}",
                session_id=session_id,
            )
    elif was_active and new_status not in ACTIVE_STATUSES:
        append_event(
            "BRANCH_DEACTIVATED",
            f"tier={old_tier} opportunity={op_id} status={new_status}",
            session_id=session_id,
        )

    # Bi-directional sync with active_state.json
    sync_active_state_on_transition(
        op_id,
        entering_active=entering_active,
        leaving_active=(was_active and new_status not in ACTIVE_STATUSES),
        tier=node.get("tier", "backlog"),
    )

    append_event(
        "OPPORTUNITY_TRANSITION",
        f"opportunity={op_id} from={old_status} to={new_status}",
        session_id=session_id,
    )
    print(f"🔄 Transitioned '{op_id}': {old_status} ➔ {new_status} (Tier: {node.get('tier', 'backlog')})")


def promote_candidate(op_id: str, evidence_note: str = "Evidence review passed") -> None:
    transition_status(op_id, "PRODUCTION_CANDIDATE", note=evidence_note)


def integrate_opportunity(op_id: str, note: Optional[str] = None, user_authorized: bool = False) -> None:
    transition_status(
        op_id,
        "INTEGRATED",
        note=note or "Production integration authorized",
        user_authorized=user_authorized,
    )


def list_opportunities() -> None:
    graph = load_graph()
    nodes = graph.get("nodes", {})
    if not nodes:
        print("Opportunity Graph is currently empty.")
        return

    print("══════════════════════════════════════════════════════════")
    print(" 🗺️ OPPORTUNITY GRAPH (LIFECYCLE STATUS)")
    print("══════════════════════════════════════════════════════════")

    by_status: Dict[str, List[Dict[str, Any]]] = {}
    for node in nodes.values():
        by_status.setdefault(node.get("status", "PARKED"), []).append(node)

    for status in [
        "ACTIVATED",
        "EXPLORING",
        "PRODUCTION_CANDIDATE",
        "INTEGRATED",
        "VALIDATED_SANDBOX",
        "PARKED",
        "PROPOSED",
        "DISPROVED",
        "ARCHIVED",
    ]:
        if status not in by_status:
            continue
        print(f"\n▶ [{status}] ({len(by_status[status])})")
        for node in by_status[status]:
            tier = node.get("tier", "")
            tier_str = f" [{tier.upper()}]" if tier in VALID_TIERS else ""
            print(f"   • {node['id']}: {node['title']}{tier_str}")

    print("\n══════════════════════════════════════════════════════════")


def check_dedup(query: str, mode: str = "check") -> None:
    if mode not in {"check", "guard"}:
        raise SystemExit("❌ Dedup mode must be 'check' or 'guard'.")

    graph = load_graph()
    query_lower = query.lower()
    matches = [
        n
        for n in graph.get("nodes", {}).values()
        if query_lower in n["id"].lower()
        or query_lower in n["title"].lower()
        or query_lower in n.get("description", "").lower()
    ]

    if matches:
        print(f"⚠️ Exploration Deduplication Match! Found {len(matches)} existing match(es) for '{query}':")
        for match in matches:
            print(f"   - [{match['status']}] {match['id']}: {match['title']}")
        if mode == "guard":
            raise SystemExit(
                "❌ Dedup Guard Block: Duplicate opportunity exists. Record materially new evidence before retrying."
            )
    else:
        print(f"✅ Clean: No prior exploration found for '{query}'. Fresh direction!")


def record_research(
    op_id: str,
    decision: str,
    findings: str,
    reason: str,
    sources: Optional[List[str]] = None,
) -> None:
    valid_decisions = {"adopted", "adapted", "rejected", "hybrid", "not_applicable"}
    if decision.lower() not in valid_decisions:
        raise SystemExit(f"❌ Invalid decision '{decision}'. Must be one of: {sorted(valid_decisions)}")

    graph = load_graph()
    nodes = graph.get("nodes", {})
    if op_id not in nodes:
        raise SystemExit(f"❌ Opportunity '{op_id}' does not exist.")

    node = nodes[op_id]
    sources_list = sources or []
    node["external_research"] = {
        "searched": True if decision.lower() != "not_applicable" else False,
        "decision": decision.lower(),
        "findings": findings,
        "reason": reason,
        "sources": sources_list,
        "updated_at": utc_now(),
    }
    save_graph(graph)
    append_event(
        "EXTERNAL_RESEARCH",
        f"opportunity={op_id} decision={decision.lower()} sources={len(sources_list)}",
    )
    print(f"🔬 External Research recorded for '{op_id}': decision='{decision.lower()}' with {len(sources_list)} source(s).")


def usage() -> None:
    print(
        "Usage:\n"
        "  opportunity_graph.py list\n"
        "  opportunity_graph.py add <id> <title> [desc] [parent_id]\n"
        "  opportunity_graph.py activate <id> [main|secondary]\n"
        "  opportunity_graph.py transition <id> <status> [note] [tier] [--user-authorized]\n"
        "  opportunity_graph.py promote-candidate <id> [evidence_note]\n"
        "  opportunity_graph.py integrate <id> [--user-authorized] [note]\n"
        "  opportunity_graph.py record-research <id> <decision> <findings> <reason> [sources...]\n"
        "  opportunity_graph.py dedup <query> [--guard|--check]"
    )


def main() -> None:
    if len(sys.argv) < 2:
        usage()
        raise SystemExit(1)

    cmd = sys.argv[1]
    if cmd == "list":
        list_opportunities()
    elif cmd == "add":
        if len(sys.argv) < 4:
            usage()
            raise SystemExit(1)
        desc = sys.argv[4] if len(sys.argv) > 4 else ""
        parent = sys.argv[5] if len(sys.argv) > 5 else None
        add_opportunity(sys.argv[2], sys.argv[3], desc, parent)
    elif cmd == "activate":
        if len(sys.argv) < 3:
            usage()
            raise SystemExit(1)
        tier = sys.argv[3] if len(sys.argv) > 3 else "main"
        activate_opportunity(sys.argv[2], tier)
    elif cmd == "transition":
        if len(sys.argv) < 4:
            usage()
            raise SystemExit(1)
        user_auth = "--user-authorized" in sys.argv
        clean_args = [arg for arg in sys.argv[4:] if arg != "--user-authorized"]
        note = clean_args[0] if len(clean_args) > 0 else None
        tier = clean_args[1] if len(clean_args) > 1 else None
        transition_status(sys.argv[2], sys.argv[3], note, tier, user_authorized=user_auth)
    elif cmd == "promote-candidate":
        if len(sys.argv) < 3:
            usage()
            raise SystemExit(1)
        evidence = sys.argv[3] if len(sys.argv) > 3 else "Evidence review passed"
        promote_candidate(sys.argv[2], evidence)
    elif cmd == "integrate":
        if len(sys.argv) < 3:
            usage()
            raise SystemExit(1)
        user_auth = "--user-authorized" in sys.argv
        clean_args = [arg for arg in sys.argv[3:] if arg != "--user-authorized"]
        note = clean_args[0] if len(clean_args) > 0 else None
        integrate_opportunity(sys.argv[2], note, user_authorized=user_auth)
    elif cmd == "record-research":
        if len(sys.argv) < 6:
            usage()
            raise SystemExit(1)
        op_id = sys.argv[2]
        decision = sys.argv[3]
        findings = sys.argv[4]
        reason = sys.argv[5]
        sources = sys.argv[6:] if len(sys.argv) > 6 else []
        record_research(op_id, decision, findings, reason, sources)
    elif cmd == "dedup":
        if len(sys.argv) < 3:
            usage()
            raise SystemExit(1)
        mode = "guard" if "--guard" in sys.argv else "check"
        query_terms = [arg for arg in sys.argv[2:] if not arg.startswith("--")]
        check_dedup(" ".join(query_terms), mode)
    else:
        usage()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
