#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPRO_ROOT = Path(__file__).resolve().parents[1]
if str(REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRO_ROOT))

import run_reproduce as rr  # noqa: E402
from repro_toolkit import (  # noqa: E402
    assert_action_allowed,
    append_history,
    blocked_action_payload,
    ensure_isolated_workspace_recipe,
    ensure_run_root,
    get_fsm_state,
    load_state,
    next_actions,
    repo_root_from_here,
    resolve_recipe_path,
    set_fsm_state,
    write_state,
)

REMOTE_QPU_PATTERNS = (
    "IBMProvider",
    "IBMQ",
    "load_account(",
    "get_backend(",
    "backend.run(",
    "data.ibm_token",
    "settings.ibm_token",
    "qiskit_ibm_runtime.models",
)

BACKEND_PROPERTIES_IMPORT_OLD = "from qiskit_ibm_runtime.models import BackendProperties"
BACKEND_PROPERTIES_IMPORT_NEW = (
    "try:\n"
    "    from qiskit_ibm_runtime.models import BackendProperties\n"
    "except Exception:\n"
    "    try:\n"
    "        from qiskit.providers.models import BackendProperties\n"
    "    except Exception:\n"
    "        BackendProperties = object\n"
)
IBM_PROVIDER_IMPORT_NEW = (
    "try:\n"
    "    from qiskit_ibm_provider import IBMProvider\n"
    "except Exception:\n"
    "    IBMProvider = None"
)

SIM_HELPER = r'''


def _repro_simulation_backend(qpu_name: str = "", qpu_alias: str = ""):
    """Return a local fake backend when remote QPU access is disabled."""
    candidates = []
    for name in (qpu_name, qpu_alias):
        if name and name not in candidates:
            candidates.append(name)
        if name and not name.startswith("Fake"):
            fake_name = "Fake" + name[:1].upper() + name[1:]
            if fake_name not in candidates:
                candidates.append(fake_name)
    candidates.extend(["FakeKolkataV2", "FakeMontrealV2", "FakeManilaV2", "FakeAthensV2"])

    for name in candidates:
        if not name:
            continue
        backend_cls = globals().get(name)
        if callable(backend_cls):
            try:
                return backend_cls()
            except Exception:
                pass

    class _ReproConfig:
        basis_gates = ["rz", "sx", "x", "cx", "measure"]
        n_qubits = 127
        simulator = True

    class _ReproProperties:
        def to_dict(self):
            return {"backend_name": qpu_alias or qpu_name or "repro_simulator"}

    class _ReproBackend:
        name = qpu_alias or qpu_name or "repro_simulator"

        def configuration(self):
            return _ReproConfig()

        def properties(self):
            return _ReproProperties()

        def run(self, circuit, shots=1024, **kwargs):
            class _Result:
                def result(self_inner):
                    return {"success": True, "shots": shots, "backend": self.name}

            return _Result()

    return _ReproBackend()
'''

TOKEN_MODULE = '''import os

IBM_TOKEN = (
    os.environ.get("IBM_TOKEN")
    or os.environ.get("QISKIT_IBM_TOKEN")
    or os.environ.get("IBM_QUANTUM_TOKEN")
    or os.environ.get("REPRO_IBM_TOKEN")
    or ""
)
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--state", required=True, help="Path to shared toolchain state.json")
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_token_module(workspace_root: Path, dotted_module: str) -> str | None:
    parts = dotted_module.split(".")
    if len(parts) < 2:
        return None
    package = workspace_root.joinpath(*parts[:-1])
    package.mkdir(parents=True, exist_ok=True)
    init_path = package / "__init__.py"
    if not init_path.exists():
        init_path.write_text("", encoding="utf-8")
    module_path = workspace_root.joinpath(*parts).with_suffix(".py")
    old = module_path.read_text(encoding="utf-8") if module_path.exists() else ""
    if "IBM_TOKEN" in old:
        return None
    module_path.write_text(TOKEN_MODULE, encoding="utf-8")
    return str(module_path)


def _scan_remote_qpu_surfaces(workspace_root: Path) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for path in workspace_root.rglob("*.py"):
        if any(part in {".git", "__pycache__", ".venv", "venv", "temp"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        found = [pattern for pattern in REMOTE_QPU_PATTERNS if pattern in text]
        if found:
            matches.append({"path": str(path), "patterns": found})
    return matches


def _has_syntax(path: Path) -> bool:
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return False
    return True


def _replace_imports(text: str) -> tuple[str, bool]:
    changed = False
    broken_provider_pattern = re.compile(
        r"(?:try:\n\s*)+from qiskit_ibm_provider import IBMProvider\n"
        r"(?:except Exception:\n\s*IBMProvider = None\n?)+",
        re.MULTILINE,
    )
    repaired = broken_provider_pattern.sub(IBM_PROVIDER_IMPORT_NEW + "\n", text)
    if repaired != text:
        text = repaired
        changed = True
    if "from qiskit_ibm_runtime.models import BackendProperties" in text:
        lines = text.splitlines()
        idx = next(
            (
                i
                for i, line in enumerate(lines)
                if "from qiskit_ibm_runtime.models import BackendProperties" in line
            ),
            -1,
        )
        if idx >= 0:
            start = idx
            while start > 0 and lines[start - 1].strip() == "try:":
                start -= 1
            end = idx + 1
            allowed_block_lines = {
                "",
                "try:",
                "except Exception:",
                "from qiskit.providers.models import BackendProperties",
                "BackendProperties = object",
            }
            while end < len(lines):
                stripped = lines[end].strip()
                if stripped in allowed_block_lines:
                    end += 1
                    continue
                break
            replacement = BACKEND_PROPERTIES_IMPORT_NEW.splitlines()
            lines = lines[:start] + replacement + lines[end:]
            text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
            changed = True
    broken_backend_properties_pattern = re.compile(
        r"(?:try:\n\s*)+from qiskit_ibm_runtime\.models import BackendProperties\n"
        r"(?:except Exception:\n\s*try:\n\s*from qiskit\.providers\.models import BackendProperties\n"
        r"\s*except Exception:\n\s*BackendProperties = object\n?\s*)+",
        re.MULTILINE,
    )
    repaired = broken_backend_properties_pattern.sub(BACKEND_PROPERTIES_IMPORT_NEW + "\n", text)
    if repaired != text:
        text = repaired
        changed = True
    if BACKEND_PROPERTIES_IMPORT_OLD in text and BACKEND_PROPERTIES_IMPORT_NEW not in text:
        text = text.replace(BACKEND_PROPERTIES_IMPORT_OLD, BACKEND_PROPERTIES_IMPORT_NEW)
        changed = True
    if "from qiskit_ibm_provider import IBMProvider" in text and IBM_PROVIDER_IMPORT_NEW not in text:
        text = text.replace("from qiskit_ibm_provider import IBMProvider", IBM_PROVIDER_IMPORT_NEW)
        changed = True
    runtime_import = "from qiskit_ibm_runtime import QiskitRuntimeService"
    runtime_import_new = (
        "try:\n"
        "    from qiskit_ibm_runtime import QiskitRuntimeService\n"
        "except Exception:\n"
        "    QiskitRuntimeService = None"
    )
    if runtime_import in text and runtime_import_new not in text:
        text = text.replace(runtime_import, runtime_import_new)
        changed = True
    return text, changed


def _patch_estimator_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    original = path.read_text(encoding="utf-8")
    text, changed = _replace_imports(original)
    if "def _repro_simulation_backend" not in text:
        anchor = "logger = logging.getLogger('stevedore')\nlogger.setLevel(logging.ERROR)\n"
        if anchor in text:
            text = text.replace(anchor, anchor + SIM_HELPER, 1)
            changed = True
    remote_block = (
        "provider = IBMProvider(token=IBM_TOKEN)\n"
        "                    print(\"Loading backend {} ({}/{})\".format(qpu_alias, i, max_qpu_id))\n"
        "                    backend = provider.get_backend(qpu_alias)"
    )
    simulation_block = (
        "backend = _repro_simulation_backend(qpu_name, qpu_alias)\n"
        "                    print(\"Loading simulation backend {} ({}/{})\".format(qpu_alias, i, max_qpu_id))"
    )
    if remote_block in text:
        text = text.replace(remote_block, simulation_block, 1)
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
        return {"path": str(path), "changed": True, "syntax_ok": _has_syntax(path)}
    return None


def _patch_ibmq_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    original = path.read_text(encoding="utf-8")
    text, changed = _replace_imports(original)
    if "self.provider = IBMProvider.load_account()" in text:
        text = text.replace(
            "self.provider = IBMProvider.load_account()",
            "self.provider = None  # repro simulation-only: do not load remote IBM account",
        )
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
        return {"path": str(path), "changed": True, "syntax_ok": _has_syntax(path)}
    return None


def _patch_ibm_backends_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    original = path.read_text(encoding="utf-8")
    text, changed = _replace_imports(original)
    if "elif isinstance(provider, IBMProvider):" in text:
        text = text.replace(
            "elif isinstance(provider, IBMProvider):",
            "elif IBMProvider is not None and isinstance(provider, IBMProvider):",
        )
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
        return {"path": str(path), "changed": True, "syntax_ok": _has_syntax(path)}
    return None


def _patch_scheduler_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    original = path.read_text(encoding="utf-8")
    text, changed = _replace_imports(original)
    if "from settings.ibm_token import IBM_TOKEN" in text:
        # Keep the import working through a workspace-local settings module. The
        # actual value may stay empty because simulation-only must avoid hardware.
        changed = True
    if changed and text != original:
        path.write_text(text, encoding="utf-8")
        return {"path": str(path), "changed": True, "syntax_ok": _has_syntax(path)}
    return None


def _compile_files(python_executable: str, files: list[str]) -> dict[str, Any]:
    if not files:
        return {"ok": True, "files": []}
    cmd = [python_executable, "-m", "py_compile", *files]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return {
        "ok": proc.returncode == 0,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "files": files,
    }


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_here()
    state_path = Path(args.state).resolve()
    state = load_state(state_path)

    is_allowed, allowed_actions = assert_action_allowed(state, "simulation_backend_fix")
    if not is_allowed:
        payload = blocked_action_payload(
            tool="repro_simulation_backend_fix",
            action="simulation_backend_fix",
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, "repro_simulation_backend_fix", payload)
        state["last_step"] = "repro_simulation_backend_fix"
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    recipe_path = resolve_recipe_path(args.recipe, repo_root)
    recipe = rr.load_json(recipe_path)
    recovery = recipe.get("recovery") or {}
    simulation_only = bool(recovery.get("simulation_only"))
    run_root = ensure_run_root(str(recipe.get("name") or recipe_path.stem), state, repo_root / "temp" / "agent_framework" / "reproduce_agent" / "runs")
    workspace_recipe_path, workspace_recipe = ensure_isolated_workspace_recipe(
        state=state,
        run_root=run_root,
        repo_root=repo_root,
        recipe_path=recipe_path,
        recipe=recipe,
    )
    workspace_root = rr.resolve_workspace_root(repo_root, str(workspace_recipe.get("workspace_root", ".")))

    token_modules = []
    for module in ("data.ibm_token", "settings.ibm_token"):
        created = _write_token_module(workspace_root, module)
        if created:
            token_modules.append(created)

    remote_surfaces_before = _scan_remote_qpu_surfaces(workspace_root)
    patch_results: list[dict[str, Any]] = []
    for maybe in (
        _patch_estimator_file(workspace_root / "qos" / "estimator" / "estimator.py"),
        _patch_ibmq_file(workspace_root / "qos" / "backends" / "ibmq.py"),
        _patch_ibm_backends_file(workspace_root / "qos" / "backends" / "ibm_backends.py"),
        _patch_scheduler_file(workspace_root / "qos" / "scheduler" / "scheduler.py"),
    ):
        if maybe:
            patch_results.append(maybe)

    touched_files = sorted({item["path"] for item in patch_results} | set(token_modules))
    compile_result = _compile_files(args.python_executable, touched_files)
    applied = simulation_only and bool(touched_files) and compile_result.get("ok")

    if applied:
        set_fsm_state(state, "APPLY_FIX")
        state["last_status"] = "simulation_backend_fix_applied"
        status = "success"
        rc = 0
    else:
        set_fsm_state(state, "CLASSIFY_FAILURE")
        state["last_status"] = "simulation_backend_fix_failed"
        status = "simulation_backend_fix_failed"
        rc = 1

    payload = {
        "tool": "repro_simulation_backend_fix",
        "status": status,
        "simulation_only": simulation_only,
        "workspace_recipe_path": str(workspace_recipe_path),
        "workspace_root": str(workspace_root),
        "remote_qpu_surfaces_before": remote_surfaces_before,
        "token_modules": token_modules,
        "patch_results": patch_results,
        "compile_result": compile_result,
        "fsm_state": get_fsm_state(state),
        "next_allowed_actions": next_actions(state),
    }
    state["last_step"] = "repro_simulation_backend_fix"
    append_history(state, "repro_simulation_backend_fix", payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
