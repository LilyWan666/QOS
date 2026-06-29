#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any, Callable


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact_rel(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return str(path)


def resolve_existing_path(raw: Any, repo_root: Path) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path if path.exists() else None


def qubits_from_name(name: str) -> int:
    match = re.search(r"-(\d+)$", str(name or ""))
    return int(match.group(1)) if match else 0


def app_from_name(name: str) -> str:
    text = str(name or "")
    match = re.match(r"(.+)-\d+$", text)
    return match.group(1) if match else text


_BENCHMARK_META_CACHE: dict[tuple[str, str], dict[str, Any] | None] = {}


def _benchmark_dir_name(app: str) -> str:
    text = str(app or "").strip().lower()
    explicit = {
        "bv": "bv",
        "ghz": "ghz",
        "qsvm": "qsvm",
        "w-state": "wstate",
        "wstate": "wstate",
        "hs-1": "hamsim_1",
        "hs-2": "hamsim_2",
        "hs-3": "hamsim_3",
        "tl-1": "twolocal_1",
        "tl-2": "twolocal_2",
        "tl-3": "twolocal_3",
        "vqe-1": "vqe_1",
        "vqe-2": "vqe_2",
        "vqe-3": "vqe_3",
        "qaoa-r2": "qaoa_r2",
        "qaoa-r3": "qaoa_r3",
        "qaoa-r4": "qaoa_r4",
        "qaoa-p1": "qaoa_pl1",
        "qaoa-p3": "qaoa_pl3",
        "qaoa-b": "qaoa_b",
    }
    if text in explicit:
        return explicit[text]
    text = text.replace("-", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
    return text


def _qasm_op_qubits(line: str) -> list[str]:
    return [f"{name}[{idx}]" for name, idx in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]", line)]


def _qasm_metadata(path: Path) -> dict[str, Any]:
    qubit_count = 0
    clbit_count = 0
    depth_by_qubit: dict[str, int] = {}
    depth = 0
    instr = 0
    measure = 0
    cnot = 0
    nonlocal_count = 0
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split("//", 1)[0].strip().rstrip(";")
        if not line or line.startswith(("OPENQASM", "include", "barrier")):
            continue
        qreg = re.match(r"qreg\s+([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]", line)
        if qreg:
            count = int(qreg.group(2))
            qubit_count += count
            for idx in range(count):
                depth_by_qubit[f"{qreg.group(1)}[{idx}]"] = 0
            continue
        creg = re.match(r"creg\s+[A-Za-z_][A-Za-z0-9_]*\[(\d+)\]", line)
        if creg:
            clbit_count += int(creg.group(1))
            continue
        qubits = _qasm_op_qubits(line)
        if not qubits:
            continue
        op = line.split(None, 1)[0].split("(", 1)[0].lower()
        instr += 1
        unique_qubits = sorted(set(qubits))
        if op == "measure":
            measure += 1
        if op in {"cx", "cnot"}:
            cnot += 1
        if len(unique_qubits) >= 2:
            nonlocal_count += 1
        next_depth = max(depth_by_qubit.get(qubit, 0) for qubit in unique_qubits) + 1
        for qubit in unique_qubits:
            depth_by_qubit[qubit] = next_depth
        depth = max(depth, next_depth)
    return {
        "depth": float(depth),
        "num_qubits": float(qubit_count),
        "num_clbits": float(clbit_count),
        "number_instructions": float(instr),
        "num_measurements": float(measure),
        "num_cnot_gates": float(cnot),
        "num_nonlocal_gates": float(nonlocal_count),
        "critical_depth": float(depth),
    }


def benchmark_metadata_for_name(repo_root: Path | None, name: str) -> dict[str, Any] | None:
    if repo_root is None:
        return None
    app = app_from_name(name)
    qubits = qubits_from_name(name)
    if not app or not qubits:
        return None
    key = (str(repo_root.resolve()), str(name))
    if key in _BENCHMARK_META_CACHE:
        return _BENCHMARK_META_CACHE[key]
    qasm = repo_root / "evaluation" / "benchmarks" / _benchmark_dir_name(app) / f"{qubits}.qasm"
    if not qasm.exists():
        _BENCHMARK_META_CACHE[key] = None
        return None
    meta = _qasm_metadata(qasm)
    meta["benchmark_name"] = name
    meta["benchmark_app"] = app
    meta["benchmark_qasm"] = artifact_rel(repo_root, qasm)
    _BENCHMARK_META_CACHE[key] = meta
    return meta


def _ratio(left: Any, right: Any) -> float | None:
    try:
        a = float(left)
        b = float(right)
    except Exception:
        return None
    if a <= 0 and b <= 0:
        return None
    return min(a, b) / max(a, b, 1.0)


def _sum(left: Any, right: Any) -> float | None:
    try:
        return float(left) + float(right)
    except Exception:
        return None


def _diff(left: Any, right: Any) -> float | None:
    try:
        return abs(float(left) - float(right))
    except Exception:
        return None


def _density(left: Any, right: Any, left_qubits: float, right_qubits: float) -> float | None:
    total = _sum(left, right)
    if total is None:
        return None
    return total / max(left_qubits + right_qubits, 1.0)


def _bounded01(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
    except Exception:
        val = default
    if not math.isfinite(val):
        val = default
    return max(0.0, min(1.0, val))


def _derived_side_metrics(meta: dict[str, Any] | None) -> dict[str, float]:
    if not meta:
        return {
            "entanglement_ratio": 0.0,
            "measurement": 0.0,
            "parallelism": 0.0,
        }
    depth = float(meta.get("depth") or 0.0)
    instr = float(meta.get("number_instructions") or 0.0)
    nonlocal_gates = float(meta.get("num_nonlocal_gates") or 0.0)
    measurements = float(meta.get("num_measurements") or 0.0)
    return {
        "entanglement_ratio": _bounded01(nonlocal_gates / max(instr, 1.0)),
        "measurement": _bounded01(measurements / max(depth, 1.0)),
        "parallelism": _bounded01(1.0 - (depth / max(instr, 1.0))),
    }


def physical_record_features(record: dict[str, Any], repo_root: Path | None = None) -> dict[str, Any]:
    left_qubits = float(record.get("left_qubits") or qubits_from_name(str(record.get("name_1") or "")) or 0.0)
    right_qubits = float(record.get("right_qubits") or qubits_from_name(str(record.get("name_2") or "")) or 0.0)
    joint_qubits = float(record.get("joint_qubits") or (left_qubits + right_qubits) or 1.0)
    effective_util = float(record.get("effective_utilization") or 0.0)
    backend_qubits = joint_qubits / max(effective_util, 1e-9) if effective_util > 0 else joint_qubits
    left_meta = benchmark_metadata_for_name(repo_root, str(record.get("name_1") or ""))
    right_meta = benchmark_metadata_for_name(repo_root, str(record.get("name_2") or ""))
    feature_source = "benchmark_qasm_metadata" if left_meta and right_meta else "physical_record_only"
    features = {
        "matching_score": 0.0,
        "selection_score": 0.0,
        "effective_utilization": effective_util,
        "effective_utilization_percent": effective_util * 100.0,
        "fidelity": float(record.get("hellinger_mean") or 0.0),
        "solo_fidelity": 1.0,
        "entanglement": float(record.get("entanglement", 0.0) or 0.0),
        "measurement": float(record.get("measurement", 0.0) or 0.0),
        "parallelism": float(record.get("parallelism", 0.0) or 0.0),
        "entanglement_ratio": float(record.get("entanglement_ratio", 0.0) or 0.0),
        "joint_qubits": joint_qubits,
        "selected_qubits": backend_qubits,
        "scale_ratio": joint_qubits / max(backend_qubits, 1.0),
        "source_scale_hint": joint_qubits / max(backend_qubits, 1.0),
        "utilization_pressure": max(0.0, joint_qubits / max(backend_qubits, 1.0) - 1.0),
        "depth_ratio": None,
        "left_qubits": left_qubits,
        "right_qubits": right_qubits,
        "qubit_imbalance": abs(left_qubits - right_qubits) / max(left_qubits + right_qubits, 1.0),
        "layout_span": joint_qubits,
        "feature_source": feature_source,
    }
    if left_meta and right_meta:
        left_side = _derived_side_metrics(left_meta)
        right_side = _derived_side_metrics(right_meta)
        pairs = {
            "depth": "depth",
            "instr": "number_instructions",
            "cnot": "num_cnot_gates",
            "nonlocal": "num_nonlocal_gates",
            "measure": "num_measurements",
            "critical_depth": "critical_depth",
        }
        for prefix, key in pairs.items():
            left_value = left_meta.get(key)
            right_value = right_meta.get(key)
            features[f"{prefix}_1"] = left_value
            features[f"{prefix}_2"] = right_value
            features[f"{prefix}_ratio"] = _ratio(left_value, right_value)
            features[f"{prefix}_sum"] = _sum(left_value, right_value)
            features[f"{prefix}_diff"] = _diff(left_value, right_value)
            features[f"{prefix}_density"] = _density(left_value, right_value, left_qubits, right_qubits)
        for key in ("entanglement_ratio", "measurement", "parallelism"):
            features[f"{key}_1"] = left_side[key]
            features[f"{key}_2"] = right_side[key]
        features["entanglement_ratio"] = (left_side["entanglement_ratio"] + right_side["entanglement_ratio"]) / 2.0
        features["measurement"] = (1.0 - left_side["measurement"]) * (1.0 - right_side["measurement"])
        features["parallelism"] = (1.0 - left_side["parallelism"]) * (1.0 - right_side["parallelism"])
        features["entanglement"] = (
            (1.0 - left_side["entanglement_ratio"]) * (1.0 - right_side["entanglement_ratio"])
        )
        features["depth_similarity"] = math.exp(-0.05 * float(features.get("depth_diff") or 0.0))
        features["metadata_left_qasm"] = left_meta.get("benchmark_qasm")
        features["metadata_right_qasm"] = right_meta.get("benchmark_qasm")
        features["metadata_materialized"] = True
    else:
        features["metadata_materialized"] = False
        features["metadata_missing_reason"] = "missing_benchmark_qasm_for_pair"
    return features


class Backend:
    def __init__(self, num_qubits: Any):
        self.num_qubits = max(float(num_qubits or 1.0), 1.0)


class Qernel:
    def __init__(self, features: dict[str, Any], side: str):
        self.features = features
        self.side = side

    def get_metadata(self) -> dict[str, float]:
        left_qubits = float(self.features.get("left_qubits", 0.0) or 0.0)
        right_qubits = float(self.features.get("right_qubits", 0.0) or 0.0)
        qubits = left_qubits if self.side == "left" else right_qubits
        selected = max(float(self.features.get("selected_qubits", 1.0) or 1.0), 1.0)
        suffix = "1" if self.side == "left" else "2"
        depth = float(self.features.get(f"depth_{suffix}") or max(10.0 * qubits, 1.0))
        return {
            "depth": depth,
            "num_qubits": qubits,
            "num_clbits": qubits,
            "num_nonlocal_gates": float(self.features.get(f"nonlocal_{suffix}") or 0.0),
            "num_connected_components": 1.0,
            "number_instructions": float(self.features.get(f"instr_{suffix}") or depth),
            "num_measurements": float(self.features.get(f"measure_{suffix}") or qubits),
            "num_cnot_gates": float(self.features.get(f"cnot_{suffix}") or 0.0),
            "program_communication": 0.0,
            "liveness": float(self.features.get("utilization_pressure", 0.0) or 0.0),
            "parallelism": _bounded01(
                self.features.get(f"parallelism_{suffix}"),
                1.0 - (float(self.features.get(f"depth_{suffix}") or depth) / max(float(self.features.get(f"instr_{suffix}") or depth), 1.0)),
            ),
            "measurement": _bounded01(self.features.get(f"measurement_{suffix}", 0.0)),
            "entanglement_ratio": _bounded01(
                self.features.get(f"entanglement_ratio_{suffix}"),
                float(self.features.get(f"nonlocal_{suffix}") or 0.0)
                / max(float(self.features.get(f"instr_{suffix}") or depth), 1.0),
            ),
            "critical_depth": float(self.features.get(f"critical_depth_{suffix}") or depth),
            "selected_qubits": selected,
        }


class CandidateSelf:
    def effective_utilization(self, q1, q2, backend):
        return float(q1.features.get("effective_utilization_percent", 0.0) or 0.0)

    def entanglementComparison(self, q1, q2):
        meta1 = q1.get_metadata()
        meta2 = q2.get_metadata()
        return _bounded01((1.0 - meta1["entanglement_ratio"]) * (1.0 - meta2["entanglement_ratio"]))

    def measurementComparison(self, q1, q2):
        meta1 = q1.get_metadata()
        meta2 = q2.get_metadata()
        return _bounded01((1.0 - meta1["measurement"]) * (1.0 - meta2["measurement"]))

    def parallelismComparison(self, q1, q2):
        meta1 = q1.get_metadata()
        meta2 = q2.get_metadata()
        return _bounded01((1.0 - meta1["parallelism"]) * (1.0 - meta2["parallelism"]))

    def depthComparison(self, q1, q2):
        meta1 = q1.get_metadata()
        meta2 = q2.get_metadata()
        return _bounded01(math.exp(-0.05 * abs(float(meta1["depth"]) - float(meta2["depth"]))))

    def fidelityComparison(self, q1, q2):
        return max(0.0, min(1.0, float(q1.features.get("fidelity", 0.0) or 0.0)))


def load_score_function(program_path: Path) -> Callable[[dict[str, Any]], float]:
    spec = importlib.util.spec_from_file_location("qos_agent_proxy_program", program_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import proxy program: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    score_pair_features = getattr(module, "score_pair_features", None)
    if callable(score_pair_features):
        return lambda features: float(score_pair_features(features))
    get_matching_score = getattr(module, "get_matching_score", None)
    if not callable(get_matching_score):
        raise ValueError(f"proxy program does not define score_pair_features or get_matching_score: {program_path}")

    def score_from_features(features: dict[str, Any]) -> float:
        q1 = Qernel(features, "left")
        q2 = Qernel(features, "right")
        backend = Backend(features.get("selected_qubits", features.get("joint_qubits", 1.0)))
        return float(get_matching_score(CandidateSelf(), q1, q2, backend, False, []))

    return score_from_features


def rank(values: list[tuple[str, float]]) -> dict[str, int]:
    ordered = sorted(values, key=lambda item: (-float(item[1]), item[0]))
    return {label: index + 1 for index, (label, _score) in enumerate(ordered)}


def spearman(left: dict[str, int], right: dict[str, int]) -> float | None:
    labels = sorted(set(left) & set(right))
    n = len(labels)
    if n < 2:
        return None
    d2 = sum((left[label] - right[label]) ** 2 for label in labels)
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def summarize_physical_records(
    records: list[dict[str, Any]],
    score_fn: Callable[[dict[str, Any]], float] | None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = f"{record.get('backend')}:{record.get('util_label')}"
        groups.setdefault(key, []).append(record)
    group_summaries: list[dict[str, Any]] = []
    correlations: list[float] = []
    for key, group in sorted(groups.items()):
        physical_values: list[tuple[str, float]] = []
        proxy_values: list[tuple[str, float]] = []
        for idx, record in enumerate(group):
            label = str(record.get("pair_label") or f"{key}:{idx}")
            physical_values.append((label, float(record.get("hellinger_mean") or 0.0)))
            if score_fn is not None:
                try:
                    proxy_score = float(score_fn(physical_record_features(record, repo_root=repo_root)))
                    if math.isnan(proxy_score) or math.isinf(proxy_score):
                        proxy_score = -1e9
                except Exception:
                    proxy_score = -1e9
                proxy_values.append((label, proxy_score))
        corr = spearman(rank(physical_values), rank(proxy_values)) if proxy_values else None
        if corr is not None and not math.isnan(corr):
            correlations.append(corr)
        values = [float(record.get("hellinger_mean") or 0.0) for record in group]
        group_summaries.append(
            {
                "group": key,
                "backend": group[0].get("backend"),
                "util_label": group[0].get("util_label"),
                "pair_count": len(group),
                "hellinger_mean_avg": sum(values) / len(values) if values else None,
                "hellinger_mean_min": min(values) if values else None,
                "hellinger_mean_max": max(values) if values else None,
                "spearman_proxy_vs_physical": corr,
            }
        )
    return {
        "groups": group_summaries,
        "mean_spearman_proxy_vs_physical": sum(correlations) / len(correlations) if correlations else None,
        "correlation_group_count": len(correlations),
    }
