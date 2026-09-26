#!/usr/bin/env python3
"""
test_autonomous_expansion.py - Unit tests for Autonomous Expansion v5.3 Infrastructure:
- Antigravity Stop Hook (stop_hook.py)
- Continuous Goal Session Controller (checkpoint.py)
- External Research Gate (opportunity_graph.py)
"""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import datetime, timezone, timedelta

import stop_hook
import opportunity_graph
import checkpoint
import overnight_report
import self_eval


class TestStopHook(unittest.TestCase):
    def setUp(self):
        self.orig_active_file = stop_hook.ACTIVE_STATE_FILE
        self.orig_graph_file = stop_hook.GRAPH_FILE

        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_active_file = os.path.join(self.temp_dir.name, "active_state.json")
        self.test_graph_file = os.path.join(self.temp_dir.name, "opportunity_graph.json")

        stop_hook.ACTIVE_STATE_FILE = self.test_active_file
        stop_hook.GRAPH_FILE = self.test_graph_file

    def tearDown(self):
        stop_hook.ACTIVE_STATE_FILE = self.orig_active_file
        stop_hook.GRAPH_FILE = self.orig_graph_file
        self.temp_dir.cleanup()

    def _write_state(self, state):
        with open(self.test_active_file, "w", encoding="utf-8") as f:
            json.dump(state, f)

    def _write_graph(self, graph):
        with open(self.test_graph_file, "w", encoding="utf-8") as f:
            json.dump(graph, f)

    def test_interactive_session_allows_stop(self):
        self._write_state({"overnight_mode": False})
        decision, reason = stop_hook.evaluate_stop_condition()
        self.assertEqual(decision, "stop")
        self.assertIn("overnight_mode is disabled", reason)

    def test_overnight_session_with_budget_and_active_work_continues(self):
        now_utc = datetime.now(timezone.utc).isoformat()
        self._write_state({
            "overnight_mode": True,
            "started_at": now_utc,
            "max_runtime_minutes": 240,
            "consecutive_no_progress_count": 0,
            "no_progress_limit": 3,
        })
        self._write_graph({
            "nodes": {
                "op1": {"id": "op1", "title": "Test Op", "status": "EXPLORING"}
            }
        })
        decision, reason = stop_hook.evaluate_stop_condition()
        self.assertEqual(decision, "continue")
        self.assertIn("op1", reason)

    def test_overnight_session_budget_exhausted_stops(self):
        past_time = (datetime.now(timezone.utc) - timedelta(minutes=250)).isoformat()
        self._write_state({
            "overnight_mode": True,
            "started_at": past_time,
            "max_runtime_minutes": 240,
            "consecutive_no_progress_count": 0,
        })
        self._write_graph({"nodes": {}})
        decision, reason = stop_hook.evaluate_stop_condition()
        self.assertEqual(decision, "stop")
        self.assertIn("budget exhausted", reason)

    def test_overnight_session_consecutive_no_progress_stops(self):
        now_utc = datetime.now(timezone.utc).isoformat()
        self._write_state({
            "overnight_mode": True,
            "started_at": now_utc,
            "max_runtime_minutes": 240,
            "consecutive_no_progress_count": 3,
            "no_progress_limit": 3,
        })
        self._write_graph({"nodes": {}})
        decision, reason = stop_hook.evaluate_stop_condition()
        self.assertEqual(decision, "stop")
        self.assertIn("reached consecutive no-progress limit", reason)

    def test_overnight_session_blocked_reason_stops(self):
        now_utc = datetime.now(timezone.utc).isoformat()
        self._write_state({
            "overnight_mode": True,
            "started_at": now_utc,
            "max_runtime_minutes": 240,
            "blocked_reason": "USER_DECISION_REQUIRED",
        })
        self._write_graph({"nodes": {}})
        decision, reason = stop_hook.evaluate_stop_condition()
        self.assertEqual(decision, "stop")
        self.assertIn("USER_DECISION_REQUIRED", reason)

    def test_stop_hook_catches_tampered_overnight_mode_with_unexpired_deadline(self):
        now_utc = datetime.now(timezone.utc)
        unexpired_deadline = (now_utc + timedelta(minutes=60)).isoformat()
        self._write_state({
            "session_id": "test_tamper_sess",
            "overnight_mode": False,
            "budget_deadline": unexpired_deadline,
            "started_at": (now_utc - timedelta(minutes=10)).isoformat(),
            "max_runtime_minutes": 240,
            "consecutive_no_progress_count": 0,
        })
        self._write_graph({"nodes": {}})
        decision, reason = stop_hook.evaluate_stop_condition()
        self.assertEqual(decision, "continue")
        # Ensure stop_hook auto-healed overnight_mode back to True
        with open(self.test_active_file, "r", encoding="utf-8") as f:
            repaired_state = json.load(f)
        self.assertTrue(repaired_state["overnight_mode"])

    def test_stop_hook_catches_tampered_phase_completed_with_unexpired_deadline(self):
        now_utc = datetime.now(timezone.utc)
        unexpired_deadline = (now_utc + timedelta(minutes=60)).isoformat()
        self._write_state({
            "session_id": "test_tamper_phase_sess",
            "overnight_mode": False,
            "phase": "completed",
            "budget_deadline": unexpired_deadline,
            "started_at": (now_utc - timedelta(minutes=10)).isoformat(),
            "max_runtime_minutes": 240,
            "consecutive_no_progress_count": 0,
        })
        self._write_graph({"nodes": {}})
        decision, reason = stop_hook.evaluate_stop_condition()
        self.assertEqual(decision, "continue")
        with open(self.test_active_file, "r", encoding="utf-8") as f:
            repaired_state = json.load(f)
        self.assertTrue(repaired_state["overnight_mode"])
        self.assertEqual(repaired_state["phase"], "rediscovery")


class TestExternalResearchGate(unittest.TestCase):
    def setUp(self):
        self.orig_graph_file = opportunity_graph.GRAPH_FILE
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_graph_file = os.path.join(self.temp_dir.name, "opportunity_graph.json")
        opportunity_graph.GRAPH_FILE = self.test_graph_file

        with open(self.test_graph_file, "w", encoding="utf-8") as f:
            json.dump({"nodes": {"sample_op": {"id": "sample_op", "title": "Sample", "status": "PARKED"}}}, f)

    def tearDown(self):
        opportunity_graph.GRAPH_FILE = self.orig_graph_file
        self.temp_dir.cleanup()

    def test_record_research_success(self):
        opportunity_graph.record_research(
            "sample_op",
            "hybrid",
            "Found external CVPR SOTA and GitHub repo",
            "Adopted lightweight components and rejected heavy dependencies",
            ["arXiv:2501.12345", "github.com/foo/bar"]
        )
        with open(self.test_graph_file, "r", encoding="utf-8") as f:
            graph = json.load(f)

        node = graph["nodes"]["sample_op"]
        self.assertIn("external_research", node)
        res = node["external_research"]
        self.assertTrue(res["searched"])
        self.assertEqual(res["decision"], "hybrid")
        self.assertEqual(len(res["sources"]), 2)


class TestCheckpointGuardrails(unittest.TestCase):
    def setUp(self):
        self.orig_active_file = checkpoint.ACTIVE_STATE_FILE
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_active_file = os.path.join(self.temp_dir.name, "active_state.json")
        checkpoint.ACTIVE_STATE_FILE = self.test_active_file

    def tearDown(self):
        checkpoint.ACTIVE_STATE_FILE = self.orig_active_file
        self.temp_dir.cleanup()

    def _write_state(self, state):
        with open(self.test_active_file, "w", encoding="utf-8") as f:
            json.dump(state, f)

    def test_set_checkpoint_intercepts_premature_completion(self):
        now_utc = datetime.now(timezone.utc).isoformat()
        self._write_state({
            "session_id": "test_sess",
            "overnight_mode": True,
            "started_at": now_utc,
            "max_runtime_minutes": 240,
            "blocked_reason": None,
        })
        checkpoint.set_checkpoint(
            phase="complete",
            mainline="none",
            secondary="none",
            current_action="all_opportunities_validated",
            next_action="morning_report_delivered"
        )
        state = checkpoint.get_state()
        self.assertEqual(state["phase"], "discovery")
        self.assertEqual(state["next_action"], "observe_and_discover_next_opportunity")

    def test_set_checkpoint_allows_completion_when_budget_exhausted(self):
        past_time = (datetime.now(timezone.utc) - timedelta(minutes=300)).isoformat()
        self._write_state({
            "session_id": "test_sess",
            "overnight_mode": True,
            "started_at": past_time,
            "max_runtime_minutes": 240,
            "blocked_reason": None,
        })
        checkpoint.set_checkpoint(
            phase="complete",
            mainline="none",
            secondary="none",
            current_action="all_opportunities_validated",
            next_action="morning_report_delivered"
        )
        state = checkpoint.get_state()
        self.assertEqual(state["phase"], "complete")

    def test_end_overnight_refuses_without_force_when_budget_remains(self):
        now_utc = datetime.now(timezone.utc).isoformat()
        self._write_state({
            "session_id": "test_sess",
            "overnight_mode": True,
            "started_at": now_utc,
            "max_runtime_minutes": 240,
            "blocked_reason": None,
        })
        os.environ.pop("HUMAN_OVERRIDE_AUTH", None)
        f = io.StringIO()
        with redirect_stderr(f):
            with self.assertRaises(SystemExit) as cm:
                checkpoint.end_overnight(force=False)
        self.assertEqual(cm.exception.code, 1)
        err = f.getvalue()
        self.assertIn("REFUSED: autonomous overnight session cannot finalize before deadline", err)
        self.assertNotIn("or pass --force to override", err)
        state = checkpoint.get_state()
        self.assertTrue(state["overnight_mode"])

    def test_end_overnight_refuses_force_when_budget_remains(self):
        now_utc = datetime.now(timezone.utc).isoformat()
        self._write_state({
            "session_id": "test_sess",
            "overnight_mode": True,
            "started_at": now_utc,
            "max_runtime_minutes": 240,
            "blocked_reason": None,
        })
        f = io.StringIO()
        with redirect_stderr(f):
            with self.assertRaises(SystemExit) as cm:
                checkpoint.end_overnight(force=True)
        self.assertEqual(cm.exception.code, 1)
        err = f.getvalue()
        self.assertIn("REFUSED: autonomous overnight session cannot finalize before deadline", err)
        self.assertNotIn("or pass --force to override", err)
        state = checkpoint.get_state()
        self.assertTrue(state["overnight_mode"])

    def test_end_overnight_refuses_even_with_human_auth_env(self):
        now_utc = datetime.now(timezone.utc).isoformat()
        self._write_state({
            "session_id": "test_sess",
            "overnight_mode": True,
            "started_at": now_utc,
            "max_runtime_minutes": 240,
            "blocked_reason": None,
        })
        os.environ["HUMAN_OVERRIDE_AUTH"] = "1"
        try:
            f = io.StringIO()
            with redirect_stderr(f):
                with self.assertRaises(SystemExit) as cm:
                    checkpoint.end_overnight(force=True)
            self.assertEqual(cm.exception.code, 1)
            err = f.getvalue()
            self.assertIn("REFUSED: autonomous overnight session cannot finalize before deadline", err)
            state = checkpoint.get_state()
            self.assertTrue(state["overnight_mode"])
        finally:
            os.environ.pop("HUMAN_OVERRIDE_AUTH", None)

    def test_end_overnight_succeeds_when_budget_exhausted(self):
        past_time = (datetime.now(timezone.utc) - timedelta(minutes=300)).isoformat()
        self._write_state({
            "session_id": "test_sess",
            "overnight_mode": True,
            "started_at": past_time,
            "max_runtime_minutes": 240,
            "blocked_reason": None,
        })
        checkpoint.end_overnight(force=False)
        state = checkpoint.get_state()
        self.assertFalse(state["overnight_mode"])


class TestWallClockDeadlineModel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_state_file = os.path.join(self.temp_dir.name, "active_state.json")
        self.test_graph_file = os.path.join(self.temp_dir.name, "opportunity_graph.json")

        self.orig_cp_state = checkpoint.ACTIVE_STATE_FILE
        self.orig_sh_state = stop_hook.ACTIVE_STATE_FILE
        self.orig_sh_graph = stop_hook.GRAPH_FILE

        checkpoint.ACTIVE_STATE_FILE = self.test_state_file
        stop_hook.ACTIVE_STATE_FILE = self.test_state_file
        stop_hook.GRAPH_FILE = self.test_graph_file

        with open(self.test_graph_file, "w", encoding="utf-8") as f:
            json.dump({"nodes": {}}, f)

    def tearDown(self):
        checkpoint.ACTIVE_STATE_FILE = self.orig_cp_state
        stop_hook.ACTIVE_STATE_FILE = self.orig_sh_state
        stop_hook.GRAPH_FILE = self.orig_sh_graph
        self.temp_dir.cleanup()

    def test_start_session_records_budget_deadline(self):
        checkpoint.start_session("test_sess_deadline", overnight_mode=True, max_runtime_minutes=180)
        state = checkpoint.get_state()
        self.assertIn("budget_deadline", state)
        self.assertIsNotNone(state["budget_deadline"])
        start_dt = checkpoint.parse_iso_time(state["started_at"])
        deadline_dt = checkpoint.parse_iso_time(state["budget_deadline"])
        self.assertIsNotNone(start_dt)
        self.assertIsNotNone(deadline_dt)
        diff_minutes = (deadline_dt - start_dt).total_seconds() / 60.0
        self.assertAlmostEqual(diff_minutes, 180.0, places=2)

    def test_stop_hook_stops_when_wall_clock_deadline_reached(self):
        now_utc = datetime.now(timezone.utc)
        past_deadline = (now_utc - timedelta(minutes=5)).isoformat()
        past_start = (now_utc - timedelta(minutes=185)).isoformat()

        with open(self.test_state_file, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": "test_sess_expired",
                "overnight_mode": True,
                "started_at": past_start,
                "budget_deadline": past_deadline,
                "max_runtime_minutes": 180,
            }, f)

        decision, reason = stop_hook.evaluate_stop_condition()
        self.assertEqual(decision, "stop")
        self.assertIn("budget exhausted", reason)

    def test_stop_hook_continues_when_within_budget_deadline(self):
        now_utc = datetime.now(timezone.utc)
        future_deadline = (now_utc + timedelta(minutes=60)).isoformat()
        start_time = (now_utc - timedelta(minutes=120)).isoformat()

        with open(self.test_state_file, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": "test_sess_active",
                "overnight_mode": True,
                "started_at": start_time,
                "budget_deadline": future_deadline,
                "max_runtime_minutes": 180,
                "consecutive_no_progress_count": 0,
            }, f)

        with open(self.test_graph_file, "w", encoding="utf-8") as f:
            json.dump({
                "nodes": {
                    "op_active": {"id": "op_active", "title": "Active Feature", "status": "EXPLORING", "tier": "main"}
                }
            }, f)

        decision, reason = stop_hook.evaluate_stop_condition()
        self.assertEqual(decision, "continue")
        self.assertIn("op_active", reason)


class TestStateGraphBidirectionalSync(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_graph_file = os.path.join(self.temp_dir.name, "opportunity_graph.json")
        self.test_state_file = os.path.join(self.temp_dir.name, "active_state.json")
        self.test_log_file = os.path.join(self.temp_dir.name, "session_log.md")

        self.orig_op_graph = opportunity_graph.GRAPH_FILE
        self.orig_op_state = opportunity_graph.STATE_FILE
        self.orig_op_log = opportunity_graph.SESSION_LOG_FILE

        self.orig_cp_graph = checkpoint.GRAPH_FILE
        self.orig_cp_state = checkpoint.ACTIVE_STATE_FILE
        self.orig_cp_log = checkpoint.SESSION_LOG_FILE

        opportunity_graph.GRAPH_FILE = self.test_graph_file
        opportunity_graph.STATE_FILE = self.test_state_file
        opportunity_graph.SESSION_LOG_FILE = self.test_log_file

        checkpoint.GRAPH_FILE = self.test_graph_file
        checkpoint.ACTIVE_STATE_FILE = self.test_state_file
        checkpoint.SESSION_LOG_FILE = self.test_log_file

        # Init graph and state
        with open(self.test_graph_file, "w", encoding="utf-8") as f:
            json.dump({
                "nodes": {
                    "test_op": {
                        "id": "test_op",
                        "title": "Test Opportunity",
                        "status": "PARKED",
                        "tier": "backlog",
                        "notes": []
                    }
                },
                "edges": []
            }, f)

        with open(self.test_state_file, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": "test_sync_sess",
                "phase": "recon",
                "mainline": None,
                "secondary": None,
                "current_action": "init",
                "next_action": "start",
                "overnight_mode": False,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "max_runtime_minutes": 240
            }, f)

    def tearDown(self):
        opportunity_graph.GRAPH_FILE = self.orig_op_graph
        opportunity_graph.STATE_FILE = self.orig_op_state
        opportunity_graph.SESSION_LOG_FILE = self.orig_op_log

        checkpoint.GRAPH_FILE = self.orig_cp_graph
        checkpoint.ACTIVE_STATE_FILE = self.orig_cp_state
        checkpoint.SESSION_LOG_FILE = self.orig_cp_log
        self.temp_dir.cleanup()

    def test_transition_to_active_syncs_state_mainline_and_phase(self):
        opportunity_graph.transition_status("test_op", "ACTIVATED", tier="main")
        opportunity_graph.transition_status("test_op", "EXPLORING")

        state = checkpoint.get_state()
        self.assertEqual(state["mainline"], "test_op")
        self.assertEqual(state["phase"], "exploring")
        self.assertEqual(state["current_action"], "exploring_test_op")

    def test_transition_to_validated_clears_mainline_and_syncs_completed(self):
        opportunity_graph.transition_status("test_op", "ACTIVATED", tier="main")
        opportunity_graph.transition_status("test_op", "EXPLORING")
        opportunity_graph.transition_status("test_op", "VALIDATED_SANDBOX")

        state = checkpoint.get_state()
        self.assertIsNone(state["mainline"])
        self.assertEqual(state["phase"], "completed")
        self.assertEqual(state["current_action"], "all_opportunities_validated")


class TestLifecycleResolverAndReport(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_graph_file = os.path.join(self.temp_dir.name, "opportunity_graph.json")
        self.test_state_file = os.path.join(self.temp_dir.name, "active_state.json")
        self.test_log_file = os.path.join(self.temp_dir.name, "session_log.md")
        self.test_bugs_file = os.path.join(self.temp_dir.name, "bugs_resolved.json")
        self.test_reports_dir = os.path.join(self.temp_dir.name, "reports")

        self.orig_rep_state = overnight_report.STATE_FILE
        self.orig_rep_graph = overnight_report.GRAPH_FILE
        self.orig_rep_log = overnight_report.SESSION_LOG_FILE
        self.orig_rep_bugs = overnight_report.BUGS_FILE
        self.orig_rep_dir = overnight_report.REPORTS_DIR

        overnight_report.STATE_FILE = self.test_state_file
        overnight_report.GRAPH_FILE = self.test_graph_file
        overnight_report.SESSION_LOG_FILE = self.test_log_file
        overnight_report.BUGS_FILE = self.test_bugs_file
        overnight_report.REPORTS_DIR = self.test_reports_dir

    def tearDown(self):
        overnight_report.STATE_FILE = self.orig_rep_state
        overnight_report.GRAPH_FILE = self.orig_rep_graph
        overnight_report.SESSION_LOG_FILE = self.orig_rep_log
        overnight_report.BUGS_FILE = self.orig_rep_bugs
        overnight_report.REPORTS_DIR = self.orig_rep_dir
        self.temp_dir.cleanup()

    def test_resolve_lifecycle_header_reconciles_stale_exploring_state(self):
        # Stale state claims 'exploring' with a mainline, but graph has all VALIDATED nodes
        stale_state = {
            "session_id": "sess_test_recon",
            "phase": "exploring",
            "mainline": "turnover_transition_spotter",
            "secondary": None,
            "overnight_mode": True,
            "started_at": (datetime.now(timezone.utc) - timedelta(minutes=300)).isoformat(),
            "max_runtime_minutes": 240,
        }
        all_validated_graph = {
            "nodes": {
                "turnover_transition_spotter": {
                    "id": "turnover_transition_spotter",
                    "status": "VALIDATED",
                    "tier": "backlog",
                }
            }
        }
        header = overnight_report.resolve_lifecycle_header(stale_state, all_validated_graph)
        self.assertEqual(header["phase"], "completed (budget_exhausted)")
        self.assertEqual(header["mainline"], "None")
        self.assertTrue(header["budget_exhausted"])

    def test_report_generation_separates_session_delta_and_cumulative_assets(self):
        now_str = datetime.now(timezone.utc).isoformat()
        state = {
            "session_id": "current_session_123",
            "phase": "completed",
            "mainline": None,
            "secondary": None,
            "overnight_mode": False,
            "started_at": now_str,
            "max_runtime_minutes": 240,
        }
        graph = {
            "nodes": {
                "op_old": {
                    "id": "op_old",
                    "title": "Old Op",
                    "status": "VALIDATED",
                    "created_session_id": "prior_session",
                },
                "op_new": {
                    "id": "op_new",
                    "title": "New Op",
                    "status": "VALIDATED",
                    "created_session_id": "current_session_123",
                },
            }
        }
        bugs = {
            "resolved_bugs": [
                {
                    "bug_id": "BUG_OLD",
                    "title": "Old Bug",
                    "session_id": "prior_session",
                    "tests_passed": True,
                    "rollback": False,
                },
                {
                    "bug_id": "BUG_NEW",
                    "title": "New Bug",
                    "session_id": "current_session_123",
                    "tests_passed": True,
                    "rollback": False,
                },
            ]
        }

        with open(self.test_state_file, "w", encoding="utf-8") as f:
            json.dump(state, f)
        with open(self.test_graph_file, "w", encoding="utf-8") as f:
            json.dump(graph, f)
        with open(self.test_bugs_file, "w", encoding="utf-8") as f:
            json.dump(bugs, f)
        with open(self.test_log_file, "w", encoding="utf-8") as f:
            f.write(
                f"[2026-09-20 10:00:00] [current_session_123] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=op_new from=EXPLORING to=VALIDATED\n"
            )

        report_path = overnight_report.generate()
        self.assertTrue(os.path.exists(report_path))

        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check section titles
        self.assertIn("NORTH STAR DELTA: 本会话新增交付 (Current Session Delta: `current_session_123`)", content)
        self.assertIn("CUMULATIVE PROJECT ASSETS: 项目历史累计总资产", content)

        # Delta section checks
        delta_part = content.split("## 🏛️ CUMULATIVE PROJECT ASSETS")[0]
        self.assertIn("BUG_NEW", delta_part)
        self.assertNotIn("BUG_OLD", delta_part)
        self.assertIn("op_new", delta_part)
        self.assertNotIn("op_old", delta_part)

        # Cumulative section checks
        cumulative_part = content.split("## 🏛️ CUMULATIVE PROJECT ASSETS")[1]
        self.assertIn("BUG_OLD", cumulative_part)
        self.assertIn("BUG_NEW", cumulative_part)
        self.assertIn("op_old", cumulative_part)
        self.assertIn("op_new", cumulative_part)


class TestCanonicalShutdownSequence(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_state_file = os.path.join(self.temp_dir.name, "active_state.json")
        self.orig_cp_state = checkpoint.ACTIVE_STATE_FILE
        checkpoint.ACTIVE_STATE_FILE = self.test_state_file

        with open(self.test_state_file, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": "test_finalize_sess",
                "phase": "exploring",
                "mainline": "some_node",
                "secondary": None,
                "current_action": "running",
                "next_action": "validate",
                "overnight_mode": True,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "max_runtime_minutes": 240,
            }, f)

    def tearDown(self):
        checkpoint.ACTIVE_STATE_FILE = self.orig_cp_state
        self.temp_dir.cleanup()
        os.environ.pop("HUMAN_OVERRIDE_AUTH", None)

    def test_finalize_session_refuses_force_when_budget_remains(self):
        f = io.StringIO()
        with redirect_stderr(f):
            with self.assertRaises(SystemExit) as cm:
                checkpoint.finalize_session(force=True)
        self.assertEqual(cm.exception.code, 1)
        err = f.getvalue()
        self.assertIn("REFUSED: autonomous overnight session cannot finalize before deadline", err)
        state = checkpoint.get_state()
        self.assertTrue(state["overnight_mode"])
        self.assertEqual(state["phase"], "exploring")

    def test_finalize_session_refuses_without_force_when_budget_remains(self):
        f = io.StringIO()
        with redirect_stderr(f):
            with self.assertRaises(SystemExit) as cm:
                checkpoint.finalize_session(force=False)
        self.assertEqual(cm.exception.code, 1)
        err = f.getvalue()
        self.assertIn("REFUSED: autonomous overnight session cannot finalize before deadline", err)
        state = checkpoint.get_state()
        self.assertTrue(state["overnight_mode"])
        self.assertEqual(state["phase"], "exploring")

    def test_finalize_session_refuses_even_with_human_auth_env(self):
        os.environ["HUMAN_OVERRIDE_AUTH"] = "1"
        try:
            f = io.StringIO()
            with redirect_stderr(f):
                with self.assertRaises(SystemExit) as cm:
                    checkpoint.finalize_session(force=True)
            self.assertEqual(cm.exception.code, 1)
            err = f.getvalue()
            self.assertIn("REFUSED: autonomous overnight session cannot finalize before deadline", err)
            state = checkpoint.get_state()
            self.assertTrue(state["overnight_mode"])
            self.assertEqual(state["phase"], "exploring")
        finally:
            os.environ.pop("HUMAN_OVERRIDE_AUTH", None)

    def test_finalize_session_succeeds_when_budget_exhausted(self):
        past_time = (datetime.now(timezone.utc) - timedelta(minutes=300)).isoformat()
        with open(self.test_state_file, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": "test_finalize_sess",
                "phase": "exploring",
                "mainline": "some_node",
                "secondary": None,
                "current_action": "running",
                "next_action": "validate",
                "overnight_mode": True,
                "started_at": past_time,
                "max_runtime_minutes": 240,
            }, f)

        finalized = checkpoint.finalize_session(force=False)
        self.assertFalse(finalized["overnight_mode"])
        self.assertEqual(finalized["phase"], "completed")
        self.assertIsNone(finalized["mainline"])
        self.assertIsNone(finalized["secondary"])
        self.assertIsNone(finalized["budget_deadline"])


class TestCLIEnforcement(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_file = os.path.join(self.temp_dir.name, "active_state.json")
        now_utc = datetime.now(timezone.utc)
        deadline = (now_utc + timedelta(minutes=180)).isoformat()
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": "test_cli_sess",
                "overnight_mode": True,
                "started_at": (now_utc - timedelta(minutes=10)).isoformat(),
                "budget_deadline": deadline,
                "max_runtime_minutes": 240,
                "phase": "exploring"
            }, f)

        self.env = os.environ.copy()
        self.env["ANTIGRAVITY_MEMORY_DIR"] = self.temp_dir.name
        self.env.pop("HUMAN_OVERRIDE_AUTH", None)
        self.script_path = os.path.abspath(checkpoint.__file__)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cli_finalize_refused_when_budget_remains(self):
        import subprocess
        res = subprocess.run(
            [sys.executable, self.script_path, "finalize"],
            capture_output=True, text=True, env=self.env
        )
        self.assertEqual(res.returncode, 1)
        self.assertIn("REFUSED: autonomous overnight session cannot finalize before deadline", res.stderr)
        self.assertNotIn("or pass --force to override", res.stderr)

    def test_cli_finalize_force_refused_when_budget_remains(self):
        import subprocess
        res = subprocess.run(
            [sys.executable, self.script_path, "finalize", "--force"],
            capture_output=True, text=True, env=self.env
        )
        self.assertEqual(res.returncode, 1)
        self.assertIn("REFUSED: autonomous overnight session cannot finalize before deadline", res.stderr)
        self.assertNotIn("or pass --force to override", res.stderr)

    def test_cli_end_overnight_force_refused_when_budget_remains(self):
        import subprocess
        res = subprocess.run(
            [sys.executable, self.script_path, "end_overnight", "--force"],
            capture_output=True, text=True, env=self.env
        )
        self.assertEqual(res.returncode, 1)
        self.assertIn("REFUSED: autonomous overnight session cannot finalize before deadline", res.stderr)
        self.assertNotIn("or pass --force to override", res.stderr)

    def test_cli_finalize_force_refused_even_with_human_auth_env(self):
        import subprocess
        self.env["HUMAN_OVERRIDE_AUTH"] = "1"
        res = subprocess.run(
            [sys.executable, self.script_path, "finalize", "--force"],
            capture_output=True, text=True, env=self.env
        )
        self.assertEqual(res.returncode, 1)
        self.assertIn("REFUSED: autonomous overnight session cannot finalize before deadline", res.stderr)


class TestProductionIntegrationGate(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.graph_file = os.path.join(self.temp_dir.name, "opportunity_graph.json")
        self.state_file = os.path.join(self.temp_dir.name, "active_state.json")
        self.log_file = os.path.join(self.temp_dir.name, "session_log.md")

        self.orig_graph = opportunity_graph.GRAPH_FILE
        self.orig_state = opportunity_graph.STATE_FILE
        self.orig_log = opportunity_graph.SESSION_LOG_FILE
        self.orig_mem = opportunity_graph.MEMORY_DIR

        opportunity_graph.GRAPH_FILE = self.graph_file
        opportunity_graph.STATE_FILE = self.state_file
        opportunity_graph.SESSION_LOG_FILE = self.log_file
        opportunity_graph.MEMORY_DIR = self.temp_dir.name

    def tearDown(self):
        opportunity_graph.GRAPH_FILE = self.orig_graph
        opportunity_graph.STATE_FILE = self.orig_state
        opportunity_graph.SESSION_LOG_FILE = self.orig_log
        opportunity_graph.MEMORY_DIR = self.orig_mem
        self.temp_dir.cleanup()

    def _write_state(self, overnight_mode=False):
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": "test_gate_session",
                "overnight_mode": overnight_mode,
                "phase": "exploring"
            }, f)

    def test_1_new_feature_to_validated_sandbox(self):
        """Test 1: New Feature transitions PARKED ➔ ACTIVATED ➔ EXPLORING ➔ VALIDATED_SANDBOX"""
        self._write_state(overnight_mode=True)
        opportunity_graph.add_opportunity("feat_test", "New Feature")
        opportunity_graph.activate_opportunity("feat_test", "main")
        opportunity_graph.transition_status("feat_test", "EXPLORING")
        opportunity_graph.transition_status("feat_test", "VALIDATED_SANDBOX")
        graph = opportunity_graph.load_graph()
        self.assertEqual(graph["nodes"]["feat_test"]["status"], "VALIDATED_SANDBOX")

    def test_2_evidence_review_to_production_candidate(self):
        """Test 2: Feature with empirical evidence review transitions VALIDATED_SANDBOX ➔ PRODUCTION_CANDIDATE"""
        self._write_state(overnight_mode=True)
        opportunity_graph.add_opportunity("feat_cand", "Candidate Feature")
        opportunity_graph.activate_opportunity("feat_cand", "main")
        opportunity_graph.transition_status("feat_cand", "EXPLORING")
        opportunity_graph.transition_status("feat_cand", "VALIDATED_SANDBOX")
        opportunity_graph.promote_candidate("feat_cand", "Empirical 750-frame E2E benchmark passed")
        graph = opportunity_graph.load_graph()
        self.assertEqual(graph["nodes"]["feat_cand"]["status"], "PRODUCTION_CANDIDATE")

    def test_3_agent_integrate_without_auth_must_fail(self):
        """Test 3: Agent attempts to transition PRODUCTION_CANDIDATE ➔ INTEGRATED without --user-authorized MUST FAIL"""
        self._write_state(overnight_mode=False)
        opportunity_graph.add_opportunity("feat_noauth", "No Auth Feature")
        opportunity_graph.activate_opportunity("feat_noauth", "main")
        opportunity_graph.transition_status("feat_noauth", "EXPLORING")
        opportunity_graph.transition_status("feat_noauth", "VALIDATED_SANDBOX")
        opportunity_graph.promote_candidate("feat_noauth", "Evidence verified")
        with self.assertRaises(SystemExit) as cm:
            opportunity_graph.transition_status("feat_noauth", "INTEGRATED", user_authorized=False)
        self.assertIn("requires explicit human authorization", str(cm.exception))

    def test_4_agent_integrate_with_user_auth_during_autonomous_mode_must_fail(self):
        """Test 4: Agent attempts integrate with --user-authorized while in autonomous mode (overnight_mode=True) MUST STILL FAIL"""
        self._write_state(overnight_mode=True)
        opportunity_graph.add_opportunity("feat_auto", "Autonomous Feature")
        opportunity_graph.activate_opportunity("feat_auto", "main")
        opportunity_graph.transition_status("feat_auto", "EXPLORING")
        opportunity_graph.transition_status("feat_auto", "VALIDATED_SANDBOX")
        opportunity_graph.promote_candidate("feat_auto", "Evidence verified")
        with self.assertRaises(SystemExit) as cm:
            opportunity_graph.transition_status("feat_auto", "INTEGRATED", user_authorized=True)
        self.assertIn("unconditionally forbidden during autonomous operation", str(cm.exception))

    def test_5_human_authorized_outside_autonomous_mode_succeeds(self):
        """Test 5: Explicit exit from autonomous mode (overnight_mode=False) + human authorization SUCCEEDS to INTEGRATED"""
        self._write_state(overnight_mode=False)
        opportunity_graph.add_opportunity("feat_prod", "Approved Feature")
        opportunity_graph.activate_opportunity("feat_prod", "main")
        opportunity_graph.transition_status("feat_prod", "EXPLORING")
        opportunity_graph.transition_status("feat_prod", "VALIDATED_SANDBOX")
        opportunity_graph.promote_candidate("feat_prod", "Evidence verified")
        opportunity_graph.integrate_opportunity("feat_prod", "Human user explicitly approved", user_authorized=True)
        graph = opportunity_graph.load_graph()
        self.assertEqual(graph["nodes"]["feat_prod"]["status"], "INTEGRATED")

    def test_legacy_validated_migrates_to_validated_sandbox(self):
        """Test legacy migration: node with VALIDATED is automatically migrated to VALIDATED_SANDBOX on load_graph"""
        with open(self.graph_file, "w", encoding="utf-8") as f:
            json.dump({
                "nodes": {
                    "legacy_op": {
                        "id": "legacy_op",
                        "title": "Legacy Op",
                        "status": "VALIDATED",
                    }
                }
            }, f)
        graph = opportunity_graph.load_graph()
        self.assertEqual(graph["nodes"]["legacy_op"]["status"], "VALIDATED_SANDBOX")

    def test_exploring_cannot_jump_directly_to_integrated(self):
        """Forbidden transition: Cannot bypass evidence review and jump directly from EXPLORING to INTEGRATED"""
        self._write_state(overnight_mode=False)
        opportunity_graph.add_opportunity("feat_jump", "Jump Feature")
        opportunity_graph.activate_opportunity("feat_jump", "main")
        opportunity_graph.transition_status("feat_jump", "EXPLORING")
        with self.assertRaises(SystemExit) as cm:
            opportunity_graph.transition_status("feat_jump", "INTEGRATED", user_authorized=True)
        self.assertIn("Illegal Lifecycle Transition", str(cm.exception))

    def test_deprecated_validated_status_rejected_in_transition(self):
        """Rejection: Attempting transition to legacy VALIDATED status raises SystemExit"""
        self._write_state(overnight_mode=False)
        opportunity_graph.add_opportunity("feat_depr", "Depr Feature")
        opportunity_graph.activate_opportunity("feat_depr", "main")
        opportunity_graph.transition_status("feat_depr", "EXPLORING")
        with self.assertRaises(SystemExit) as cm:
            opportunity_graph.transition_status("feat_depr", "VALIDATED")
        self.assertIn("Legacy Status Deprecated", str(cm.exception))


class TestSelfEval(unittest.TestCase):
    def test_self_eval_audit_runs_cleanly(self):
        """Verify that self_eval.audit() executes without NameError or crash."""
        res = self_eval.audit()
        self.assertIsInstance(res, dict)
        self.assertIn("divergence_quality", res)
        self.assertIn("convergence_balance", res)
        self.assertIn("memory_quality", res)
        self.assertIn("execution_value", res)
        self.assertIn("evolution_check", res)


if __name__ == "__main__":
    unittest.main()
