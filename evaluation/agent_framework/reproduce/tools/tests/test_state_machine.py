#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import state_machine as sm
from repro_toolkit import assert_action_allowed, ensure_runtime_env_selected, set_fsm_state


class ReproStateMachineTests(unittest.TestCase):
    def test_base_transition_table_contains_expected_core_paths(self) -> None:
        self.assertIn("repro_preflight", sm.next_allowed_actions(sm.INIT))
        self.assertIn("repro_run_once", sm.next_allowed_actions(sm.PREFLIGHT))
        self.assertIn("verify_claim", sm.next_allowed_actions(sm.RUN_ONCE))
        self.assertIn("classify_failure", sm.next_allowed_actions(sm.RUN_FAILED))
        self.assertIn("render_artifacts", sm.next_allowed_actions(sm.SUCCESS))

    def test_apply_fix_forbidden_when_unrecoverable_or_budget_exhausted(self) -> None:
        no_recovery = sm.next_allowed_actions(sm.CLASSIFY_FAILURE, recoverable=False, fix_budget=3)
        self.assertNotIn("apply_fix", no_recovery)

        no_budget = sm.next_allowed_actions(sm.CLASSIFY_FAILURE, recoverable=True, fix_budget=0)
        self.assertNotIn("apply_fix", no_budget)
        self.assertIn("terminal_failed", no_budget)

    def test_run_failure_must_classify_before_any_success_path(self) -> None:
        actions = sm.next_allowed_actions(sm.RUN_FAILED)
        self.assertIn("env_fix", actions)
        self.assertIn("constraint_resolve", actions)
        self.assertIn("classify_failure", actions)
        self.assertNotIn("success", actions)
        self.assertNotIn("done", actions)
        self.assertTrue(sm.is_action_allowed(sm.RUN_FAILED, "classify_failure"))
        self.assertFalse(sm.is_action_allowed(sm.RUN_FAILED, "success"))

    def test_classify_failure_prefers_source_fix_path(self) -> None:
        actions = sm.next_allowed_actions(sm.CLASSIFY_FAILURE, recoverable=True, fix_budget=2)
        self.assertIn("runtime_trace_fix", actions)
        self.assertIn("source_path_fix", actions)
        self.assertIn("dependency_runtime_fix", actions)
        self.assertIn("source_fix", actions)
        self.assertIn("apply_fix", actions)

    def test_verify_claim_requires_tests_gate_when_enabled(self) -> None:
        actions = sm.next_allowed_actions(sm.VERIFY_CLAIM, tests_required=True)
        self.assertNotIn("success", actions)
        self.assertIn("test_plan", actions)

    def test_artifacts_rendered_is_terminal_done_only(self) -> None:
        actions = sm.next_allowed_actions(sm.ARTIFACTS_RENDERED)
        self.assertEqual(actions, ["done"])

    def test_tool_guard_blocks_invalid_action_for_current_state(self) -> None:
        state = {}
        ok, _ = assert_action_allowed(state, "repro_preflight")
        self.assertTrue(ok)

        set_fsm_state(state, sm.PREFLIGHT)
        blocked, allowed = assert_action_allowed(state, "apply_fix")
        self.assertFalse(blocked)
        self.assertIn("repro_run_once", allowed)

    def test_runtime_env_ignores_selector_candidate_for_mutating_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "temp/agent_framework/reproduce_agent/runs/example_run"
            candidate_python = run_root / "runtime_python_selector/cand_01/bin/python"
            stable_python = root / "temp/agent_framework/reproduce/runtime_venvs/example_run/bin/python"
            candidate_python.parent.mkdir(parents=True)
            stable_python.parent.mkdir(parents=True)
            candidate_python.write_text("#!/bin/sh\n", encoding="utf-8")
            stable_python.write_text("#!/bin/sh\n", encoding="utf-8")

            state = {
                "run_root": str(run_root),
                "runtime_venv_dir": str(candidate_python.parent.parent),
                "runtime_python_executable": str(candidate_python),
            }

            venv_dir, python = ensure_runtime_env_selected(
                state=state,
                repo_root=root,
                bootstrap_python_executable=sys.executable,
            )

            self.assertEqual(venv_dir, stable_python.parent.parent)
            self.assertEqual(python, stable_python)
            self.assertEqual(state["runtime_python_executable"], str(stable_python))


if __name__ == "__main__":
    unittest.main()
