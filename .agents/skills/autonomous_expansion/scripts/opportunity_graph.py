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
from datetime import datetime, timezone
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
    "VALIDATED",
    "DISPROVED",
    "PROPOSED",
    "ARCHIVED",
]
ACTIVE_STATUSES = {"ACTIVATED", "EXPLORING"}
VALID_TIERS = {"main", "secondary"}

ALLOWED_TRANSITIONS = {
    "PARKED": ["ACTIVATED", "PROPOSED", "ARCHIVED"],
    "ACTIVATED": ["EXPLORING", "PARKED", "ARCHIVED"],
    "EXPLORING": ["VALIDATED", "DISPROVED", "PROPOSED", "PARKED", "ARCHIVED"],
    "VALIDATED": ["ARCHIVED", "PARKED"],
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


def transition_status(
    op_id: str,
    new_status: str,
    note: Optional[str] = None,
    tier: Optional[str] = None,
) -> None:
    new_status = new_status.upper()
    if new_status not in VALID_STATUSES:
        raise SystemExit(f"❌ Invalid status '{new_status}'. Must be one of: {VALID_STATUSES}")

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

    append_event(
        "OPPORTUNITY_TRANSITION",
        f"opportunity={op_id} from={old_status} to={new_status}",
        session_id=session_id,
    )
    print(f"🔄 Transitioned '{op_id}': {old_status} ➔ {new_status} (Tier: {node.get('tier', 'backlog')})")


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

    for status in ["ACTIVATED", "EXPLORING", "PARKED", "PROPOSED", "VALIDATED", "DISPROVED", "ARCHIVED"]:
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


def usage() -> None:
    print(
        "Usage:\n"
        "  opportunity_graph.py list\n"
        "  opportunity_graph.py add <id> <title> [desc] [parent_id]\n"
        "  opportunity_graph.py activate <id> [main|secondary]\n"
        "  opportunity_graph.py transition <id> <status> [note] [tier]\n"
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
        note = sys.argv[4] if len(sys.argv) > 4 else None
        tier = sys.argv[5] if len(sys.argv) > 5 else None
        transition_status(sys.argv[2], sys.argv[3], note, tier)
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
