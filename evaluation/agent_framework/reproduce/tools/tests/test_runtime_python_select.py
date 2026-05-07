#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from tool_runtime_python_select import _candidate_pythons, _runtime_python_select_failures


class RuntimePythonSelectPolicyTests(unittest.TestCase):
    def test_candidate_pythons_skip_hints_on_first_try(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            hint_dir = repo_root / "temp/agent_framework/reproduce/python_hints"
            hint_dir.mkdir(parents=True, exist_ok=True)
            hinted_python = "/tmp/fake-hint-python"
            (hint_dir / "runtime_python").write_text(hinted_python, encoding="utf-8")

            candidates = _candidate_pythons(
                repo_root, bootstrap_python="python3", include_hints=False
            )
            self.assertNotIn(hinted_python, candidates)

    def test_candidate_pythons_include_hints_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            hint_dir = repo_root / "temp/agent_framework/reproduce/python_hints"
            hint_dir.mkdir(parents=True, exist_ok=True)
            hinted_python = "/tmp/fake-hint-python"
            (hint_dir / "runtime_python").write_text(hinted_python, encoding="utf-8")

            candidates = _candidate_pythons(
                repo_root, bootstrap_python="python3", include_hints=True
            )
            self.assertIn(hinted_python, candidates)

    def test_runtime_python_select_failures_counts_only_failed_attempts(self) -> None:
        state = {
            "history": [
                {"step": "repro_runtime_python_select", "payload": {"status": "failed"}},
                {"step": "repro_runtime_python_select", "payload": {"status": "success"}},
                {"step": "repro_runtime_env_select", "payload": {"status": "failed"}},
                {"step": "repro_runtime_python_select", "payload": {"status": "failed"}},
            ]
        }
        self.assertEqual(_runtime_python_select_failures(state), 2)


if __name__ == "__main__":
    unittest.main()
