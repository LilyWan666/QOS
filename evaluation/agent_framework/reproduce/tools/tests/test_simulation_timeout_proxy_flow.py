#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPRO_ROOT = Path(__file__).resolve().parents[2]
if str(REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRO_ROOT))

TOOLS_ROOT = REPRO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import run_reproduce as rr
from run_reproduce_tool_loop import choose_action_fallback
from tool_openevolve_proxy_search import (
    _build_evaluator,
    _build_initial_program,
    _build_training_data,
    _seed_transfer_payload,
)
from tool_simulation_checkpoint_analyze import build_memory


class SimulationTimeoutProxyFlowTests(unittest.TestCase):
    def _recipe(self) -> dict:
        return {
            "name": "qos_fig11_full_agent_autorun",
            "recovery": {"simulation_only": True},
            "simulation_budget": {
                "strict_timeout_seconds": 900,
                "proxy_on_timeout": True,
            },
            "evolution": {
                "proxy_search": {
                    "enabled": True,
                }
            },
        }

    def test_timeout_classifies_as_simulation_too_expensive(self) -> None:
        classification = rr.classify_failure(
            recipe=self._recipe(),
            workspace_root=Path("."),
            status="run_failed",
            preflight_payload={},
            run_payload={
                "timed_out": True,
                "duration_seconds": 900.0,
                "timeout_seconds": 900,
                "returncode": 124,
            },
            metrics_payload=None,
            run_error_signals={},
            parse_error=None,
        )

        self.assertEqual(classification["category"], "simulation_too_expensive")
        self.assertTrue(classification["recoverable"])
        self.assertEqual(classification["evidence"][0]["strict_success"], False)

    def test_recovery_plan_routes_timeout_to_openevolve_proxy(self) -> None:
        classification = {
            "category": "simulation_too_expensive",
            "recoverable": True,
            "confidence": 0.98,
            "evidence": [],
        }

        plan = rr.build_recovery_plan(self._recipe(), classification)
        actions = [item["action"] for item in plan["actions"]]

        self.assertEqual(
            actions,
            [
                "physical_qpu_env_probe",
                "openevolve_target_probe",
                "openevolve_target_semantic_select",
                "openevolve_param_probe",
                "simulation_checkpoint_analyze",
                "proxy_semantic_factor_brainstorm",
                "proxy_metric_semantic_propose",
                "proxy_feature_semantic_validate",
                "proxy_environment_predict",
                "openevolve_proxy_search",
                "openevolve_slurm_submit",
                "openevolve_slurm_collect",
                "openevolve_proxy_verify",
                "proxy_verify_physical",
            ],
        )

    def test_timeout_recovery_includes_semantic_feature_validation(self) -> None:
        classification = {
            "category": "simulation_too_expensive",
            "recoverable": True,
            "confidence": 0.98,
            "evidence": [],
        }

        plan = rr.build_recovery_plan(self._recipe(), classification)
        actions = [item["action"] for item in plan["actions"]]

        self.assertLess(
            actions.index("proxy_metric_semantic_propose"),
            actions.index("proxy_feature_semantic_validate"),
        )
        self.assertLess(
            actions.index("proxy_feature_semantic_validate"),
            actions.index("openevolve_proxy_search"),
        )

    def test_legacy_proxy_recovery_prefix_is_preserved(self) -> None:
        classification = {
            "category": "simulation_too_expensive",
            "recoverable": True,
            "confidence": 0.98,
            "evidence": [],
        }

        plan = rr.build_recovery_plan(self._recipe(), classification)
        actions = [item["action"] for item in plan["actions"]]

        self.assertEqual(
            [action for action in actions if action.startswith("openevolve_")],
            [
                "openevolve_target_probe",
                "openevolve_target_semantic_select",
                "openevolve_param_probe",
                "openevolve_proxy_search",
                "openevolve_slurm_submit",
                "openevolve_slurm_collect",
                "openevolve_proxy_verify",
            ],
        )

    def test_fallback_chooses_proxy_target_probe_after_expensive_simulation(self) -> None:
        state = {
            "fsm_state": "CLASSIFY_FAILURE",
            "last_classification": {
                "failure_category": "simulation_too_expensive",
                "suggested_recovery_actions": [
                    "openevolve_target_probe",
                    "openevolve_target_semantic_select",
                    "openevolve_param_probe",
                    "simulation_checkpoint_analyze",
                    "openevolve_proxy_search",
                    "openevolve_slurm_submit",
                    "openevolve_slurm_collect",
                ],
            },
            "last_attempt_payload": {
                "diagnosis_payload": {
                    "classification": {"category": "simulation_too_expensive"}
                }
            },
        }
        allowed = [
            "repo_path_probe",
            "build_original_runner",
            "openevolve_target_probe",
            "openevolve_target_semantic_select",
            "openevolve_param_probe",
            "simulation_checkpoint_analyze",
            "openevolve_proxy_search",
            "openevolve_slurm_submit",
            "openevolve_slurm_collect",
            "terminal_failed",
        ]

        self.assertEqual(choose_action_fallback(state, allowed), "openevolve_target_probe")

    def test_openevolve_initial_seed_normalizes_percent_utilization_by_default(self) -> None:
        target = {
            "entrypoint": "qos.multiprogrammer.multiprogrammer:Multiprogrammer.get_matching_score",
            "source_path": "qos/multiprogrammer/multiprogrammer.py",
            "source": """
            def get_matching_score(self, q1, q2, backend, weighted=False, weights=[]):
                util_eff = self.effective_utilization(q1, q2, backend)
                entanglementDiff = self.entanglementComparison(q1, q2)
                measurementDiff = self.measurementComparison(q1, q2)
                parallelismDiff = self.parallelismComparison(q1, q2)
                return (util_eff + entanglementDiff + measurementDiff + parallelismDiff) / 4
            """,
        }

        program = _build_initial_program(target)

        self.assertIn("Initial seed mode: manual_qos_normalized", program)
        self.assertIn("_qos_agent_normalize_utilization(self.effective_utilization", program)

    def test_openevolve_evaluator_does_not_gate_score_by_front_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            training_path = work / "training_data.json"
            records = []
            for index in range(10):
                rank_value = 10 - index
                records.append(
                    {
                        "pair_label": f"pair_{index}",
                        "effective_utilization": float(rank_value),
                        "proxy_fidelity_label": float(rank_value),
                        "features": {
                            "selected_qubits": 8,
                            "left_qubits": 4,
                            "right_qubits": 4,
                            "depth_1": 100.0 if index == 1 else float(10 - index),
                            "depth_2": 1.0,
                        },
                    }
                )
            training_path.write_text(
                json.dumps(
                    {
                        "label_metric": "proxy_estimated_fidelity",
                        "source_kind": "physical_qpu",
                        "split": {"mode": "none"},
                        "thresholds": {"util30": records},
                    }
                ),
                encoding="utf-8",
            )
            evaluator_path = work / "evaluator.py"
            evaluator_path.write_text(_build_evaluator(training_path), encoding="utf-8")
            program_path = work / "candidate.py"
            program_path.write_text(
                """
def get_matching_score(self, q1, q2, backend, weighted=False, weights=[]):
    return float(q1.get_metadata("depth"))
""",
                encoding="utf-8",
            )

            spec = importlib.util.spec_from_file_location("test_evaluator", evaluator_path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            result = module.evaluate(program_path)
            metrics = result.metrics if hasattr(result, "metrics") else result

        self.assertAlmostEqual(metrics["inv_avg_pareto_rank"], 0.5)
        self.assertEqual(metrics["top_rank_overlap"], 0.0)
        self.assertAlmostEqual(metrics["validation_score"], 0.5)
        self.assertEqual(metrics["objective_aggregation"], "selected_topk_average_pareto_rank")
        self.assertEqual(metrics["front_overlap_affects_validation_score"], 0.0)

    def test_checkpoint_analyze_builds_scale_memory_from_partial_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            attempt = run_root / "attempts" / "attempt_0001_run_once"
            attempt.mkdir(parents=True)
            (attempt / "partial_simulation_checkpoint.json").write_text(
                """{
                  "schema": "qos_agent.partial_simulation_checkpoint.v1",
                  "checkpoint_kind": "application_threshold_metrics",
                  "thresholds": [0.3, 0.6, 0.88],
                  "simulation_shots": 8192,
                  "qpu_qubits": 27,
                  "no_mp_target_size_records": {
                    "0.3": {"fidelity": 0.96, "seconds": 3.0},
                    "0.6": {"fidelity": 0.24, "seconds": 188.0}
                  },
                  "selected_pair_simulation_records_by_threshold": {}
                }""",
                encoding="utf-8",
            )

            memory = build_memory(Path("."), run_root)

        self.assertTrue(memory["success"])
        self.assertTrue(memory["usable_as_scale_memory"])
        self.assertFalse(memory["usable_for_proxy_seed"])
        self.assertEqual(memory["completed_no_mp_thresholds"], ["0.3", "0.6"])
        self.assertEqual(memory["timed_out_or_missing_thresholds"], ["0.88"])
        self.assertTrue(memory["seed_transfer"]["uses_simulation_grounded_sources"])
        self.assertFalse(memory["seed_transfer"]["uses_pair_level_ground_truth"])

    def test_small_to_large_seed_transfer_uses_completed_smaller_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            metrics_path.write_text(
                """{
                  "simulation_shots": 8192,
                  "selected_pair_simulation_records_by_threshold": {
                    "0.3": {
                      "selected_qubits": 8,
                      "qos": {"records": [
                        {
                          "pair_label": "a+b",
                          "left_application": "a",
                          "right_application": "b",
                          "joint_qubits": 8,
                          "left_qubits": 4,
                          "right_qubits": 4,
                          "relative_fidelity": 0.91,
                          "matching_score": 0.8
                        }
                      ]}
                    },
                    "0.6": {
                      "selected_qubits": 16,
                      "qos": {"records": [
                        {
                          "pair_label": "c+d",
                          "left_application": "c",
                          "right_application": "d",
                          "joint_qubits": 16,
                          "left_qubits": 8,
                          "right_qubits": 8,
                          "relative_fidelity": 0.73,
                          "matching_score": 0.7
                        }
                      ]}
                    }
                  }
                }""",
                encoding="utf-8",
            )

            training = _build_training_data(metrics_path)
            transfer = _seed_transfer_payload(
                training,
                {
                    "OE_EVAL_UTILS": "30:60:88",
                    "OE_SEED_TRANSFER": "small_to_large",
                    "OE_SEED_SOURCE_POLICY": "completed_smaller_thresholds",
                    "OE_SEED_TARGET_POLICY": "timed_out_larger_thresholds",
                },
            )

        self.assertEqual(training["status"], "pair_record_ground_truth")
        self.assertEqual(training["record_count"], 2)
        self.assertEqual(transfer["source_thresholds"], ["0.3", "0.6"])
        self.assertEqual(transfer["target_thresholds"], ["88"])
        self.assertTrue(transfer["uses_simulation_grounded_sources"])
        self.assertFalse(transfer["strict_fig11_success"])


if __name__ == "__main__":
    unittest.main()
