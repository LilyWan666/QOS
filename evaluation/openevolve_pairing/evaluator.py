"""Evaluator for openevolve pairing search."""

import os
import sys
import random
import re
import csv
import io
import importlib.util
import types
from typing import Dict, List, Tuple, Any
from collections import Counter

import numpy as np
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm optional
    def tqdm(x, **_kwargs):
        return x

from qos.multiprogrammer.multiprogrammer import Multiprogrammer
from qos.error_mitigator.analyser import SupermarqFeaturesAnalysisPass, BasicAnalysisPass
from qos.types.types import Qernel

import config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_REPRO_PATH = os.path.join(PROJECT_ROOT, "test", "reproduce_fig11.py")
_repro_spec = importlib.util.spec_from_file_location("repro_fig11", _REPRO_PATH)
repro = importlib.util.module_from_spec(_repro_spec)
_repro_spec.loader.exec_module(repro)

_INIT_DONE = False
_BENCHMARKS = None
_CANDIDATES = None
_FEATURES = None
_QERNEL_PAIRS = None
_MP = None
_PAIR_METRICS: Dict[str, Tuple[float, float]] = {}
_PAIR_METADATA: Dict[str, Dict[str, str]] = {}
_PAIR_METADATA_COLUMNS: List[str] = []
_PAIR_RANKS = None
_PAIR_PROXY = None
_SIM = None

_PROXY_DEFAULT_FEATURES = (
    "depth_ratio",
    "cnot_ratio",
    "nonlocal_ratio",
    "measure_ratio",
    "instr_ratio",
)

_PROXY_FEATURE_ALIASES = {
    "depth": "depth_max",
    "max_depth": "depth_max",
    "depth_max": "depth_max",
    "depthsum": "depth_sum",
    "number_instructions_ratio": "instr_ratio",
    "instruction_ratio": "instr_ratio",
    "instructions_ratio": "instr_ratio",
    "instr_ratio": "instr_ratio",
    "num_nonlocal_gates_ratio": "nonlocal_ratio",
    "nonlocal_gates_ratio": "nonlocal_ratio",
    "nonlocal_ratio": "nonlocal_ratio",
    "num_measurements_ratio": "measure_ratio",
    "measurement_ratio": "measure_ratio",
    "measure_ratio": "measure_ratio",
    "meas_ratio": "measure_ratio",
    "num_cnot_gates_ratio": "cnot_ratio",
    "cnot_gates_ratio": "cnot_ratio",
    "cnot_ratio": "cnot_ratio",
    "-cnot_sum": "cnot_sum",
    "neg_cnot_sum": "cnot_sum",
    "-nonlocal_sum": "nonlocal_sum",
    "neg_nonlocal_sum": "nonlocal_sum",
    "-instr_sum": "instr_sum",
    "neg_instr_sum": "instr_sum",
    "critical_depth_avg": "critical_depth_avg",
    "critical_depth_2": "critical_depth_2",
    "criticaldepthavg": "critical_depth_avg",
    "criticaldepth2": "critical_depth_2",
    "critical_depth2": "critical_depth_2",
}

_PROXY_BENEFIT_FEATURES = {
    "depth_ratio",
    "cnot_ratio",
    "nonlocal_ratio",
    "measure_ratio",
    "instr_ratio",
    "qubit_ratio",
    "entanglement",
    "measurement",
    "parallelism",
    "depth_sim",
    "critical_depth_avg",
    "critical_depth_2",
}

_PROXY_COST_FEATURES = {
    "depth_sum",
    "depth_diff",
    "depth_max",
    "cnot_sum",
    "cnot_diff",
    "cnot_max",
    "nonlocal_sum",
    "nonlocal_diff",
    "nonlocal_max",
    "measure_sum",
    "measure_diff",
    "measure_max",
    "instr_sum",
    "instr_diff",
    "instr_max",
    "qubits_sum",
    "qubits_diff",
    "qubits_max",
}

_PROXY_SUPPORTED_FEATURES = _PROXY_BENEFIT_FEATURES | _PROXY_COST_FEATURES


def _reset_runtime_state():
    global _INIT_DONE, _BENCHMARKS, _CANDIDATES, _FEATURES, _QERNEL_PAIRS, _MP
    global _PAIR_METRICS, _PAIR_METADATA, _PAIR_METADATA_COLUMNS, _PAIR_RANKS, _PAIR_PROXY, _SIM
    _INIT_DONE = False
    _BENCHMARKS = None
    _CANDIDATES = None
    _FEATURES = None
    _QERNEL_PAIRS = None
    _MP = None
    _PAIR_METRICS = {}
    _PAIR_METADATA = {}
    _PAIR_METADATA_COLUMNS = []
    _PAIR_RANKS = None
    _PAIR_PROXY = None
    _SIM = None


def _strip_code_fences(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _extract_get_matching_score_block(text: str) -> str:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("def get_matching_score"):
            start = i
            break
    if start is None:
        return ""
    base_indent = len(lines[start]) - len(lines[start].lstrip())
    block_lines = []
    for j in range(start, len(lines)):
        line = lines[j]
        stripped = line.strip()
        if j > start and stripped and (len(line) - len(line.lstrip())) <= base_indent:
            break
        block_lines.append(line)
    return "\n".join(block_lines).rstrip() + "\n"


def _normalize_candidate_signature(text: str) -> str:
    # Fix common LLM stub: def get_matching_score(...):
    pattern = r"def\s+get_matching_score\s*\(\s*\.\.\.\s*\)\s*:"
    replacement = (
        "def get_matching_score(self, q1, q2, backend, weighted=False, weights=[]):"
    )
    return re.sub(pattern, replacement, text)


def _load_score_fn(path):
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    source = _strip_code_fences(source)

    module = types.ModuleType("candidate")
    try:
        compiled = compile(source, path, "exec")
        exec(compiled, module.__dict__)
    except SyntaxError:
        sanitized = _extract_get_matching_score_block(source)
        if not sanitized:
            raise
        sanitized = _normalize_candidate_signature(sanitized)
        compiled = compile(sanitized, f"{path} (sanitized)", "exec")
        exec(compiled, module.__dict__)

    if hasattr(module, "get_matching_score"):
        return "method", module.get_matching_score
    raise ValueError("Candidate program must define get_matching_score.")


def _pareto_frontier(points: List[Tuple[float, float]]):
    # Strict non-dominated frontier for maximization.
    points = sorted(points, key=lambda p: (p[0], p[1]), reverse=True)
    frontier = []
    best_y = float("-inf")
    for x, y in points:
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    return sorted(frontier, key=lambda p: p[0])


def _hypervolume(points: List[Tuple[float, float]], ref: Tuple[float, float]):
    ref_x, ref_y = ref
    filtered = [(x, y) for x, y in points if x >= ref_x and y >= ref_y]
    if not filtered:
        return 0.0
    frontier = _pareto_frontier(filtered)
    hv = 0.0
    prev_x = ref_x
    for x, y in frontier:
        hv += max(0.0, x - prev_x) * max(0.0, y - ref_y)
        prev_x = x
    return hv


def _canonical_proxy_feature_name(name: str) -> str:
    key = str(name or "").strip().lower()
    if not key:
        return ""
    return _PROXY_FEATURE_ALIASES.get(key, key)


def _resolve_proxy_feature_name() -> str:
    configured = list(getattr(config, "PROXY_FEATURES", []) or [])
    candidates = []
    for raw in configured:
        canonical = _canonical_proxy_feature_name(raw)
        if canonical and canonical not in candidates:
            candidates.append(canonical)
    if not candidates:
        candidates = [x for x in _PROXY_DEFAULT_FEATURES]

    selected_raw = str(getattr(config, "PROXY_FEATURE", "") or "").strip().lower()
    if selected_raw in ("", "auto", "default", "first", "from_list", "list"):
        selected = candidates[0]
    elif selected_raw.isdigit():
        idx = int(selected_raw) - 1
        if idx < 0 or idx >= len(candidates):
            raise ValueError(
                f"Invalid OE_PROXY_FEATURE index: {selected_raw}; valid range is 1..{len(candidates)}."
            )
        selected = candidates[idx]
    else:
        selected = _canonical_proxy_feature_name(selected_raw)

    if selected not in _PROXY_SUPPORTED_FEATURES:
        supported = ", ".join(sorted(_PROXY_SUPPORTED_FEATURES))
        raise ValueError(
            f"Unsupported OE_PROXY_FEATURE='{selected}'. Supported features: {supported}"
        )
    return selected


def _resolve_pareto_second_metric() -> str:
    raw = str(getattr(config, "PARETO_SECOND_METRIC", "fidelity") or "fidelity").strip().lower()
    if raw in ("fidelity", "fid", "sim", "simulation"):
        return "fidelity"
    if raw in ("proxy", "feature", "proxy_feature"):
        return "proxy"
    raise ValueError(
        f"Unsupported OE_PARETO_SECOND_METRIC='{raw}'. Use 'fidelity' or 'proxy'."
    )


def _pareto_second_metric_desc() -> str:
    mode = _resolve_pareto_second_metric()
    if mode == "proxy":
        return f"proxy[{_resolve_proxy_feature_name()}]"
    return "fidelity"


def _fast_fidelity_proxy(q1: Qernel, q2: Qernel) -> float:
    meta1 = q1.get_metadata()
    meta2 = q2.get_metadata()
    instr1 = meta1.get("number_instructions", 0)
    instr2 = meta2.get("number_instructions", 0)
    instr_diff = abs(instr1 - instr2)
    instr_sum = instr1 + instr2
    instr_diff_norm = instr_diff / max(instr_sum, 1)
    return 1.0 - instr_diff_norm


def _get_pair_proxy(idx: int) -> float:
    global _PAIR_PROXY
    if _PAIR_PROXY is None:
        _PAIR_PROXY = {}
    cache_key = ("legacy", int(idx))
    if cache_key in _PAIR_PROXY:
        return _PAIR_PROXY[cache_key]
    q1, q2 = _QERNEL_PAIRS[idx]
    val = _fast_fidelity_proxy(q1, q2)
    _PAIR_PROXY[cache_key] = val
    return val


def _extract_proxy_feature_raw(idx: int, feature_name: str) -> float:
    f = _FEATURES[idx]
    if feature_name in f:
        return float(f.get(feature_name, 0.0))
    if feature_name == "depth_max":
        return float(max(f.get("depth_1", 0.0), f.get("depth_2", 0.0)))
    if feature_name == "cnot_max":
        return float(max(f.get("cnot_1", 0.0), f.get("cnot_2", 0.0)))
    if feature_name == "nonlocal_max":
        return float(max(f.get("nonlocal_1", 0.0), f.get("nonlocal_2", 0.0)))
    if feature_name == "measure_max":
        return float(max(f.get("measure_1", 0.0), f.get("measure_2", 0.0)))
    if feature_name == "instr_max":
        return float(max(f.get("instr_1", 0.0), f.get("instr_2", 0.0)))
    if feature_name == "qubits_max":
        return float(max(f.get("num_qubits_1", 0.0), f.get("num_qubits_2", 0.0)))
    raise KeyError(f"Feature '{feature_name}' is not available in _FEATURES.")


def _transform_proxy_feature_value(feature_name: str, raw_value: float) -> float:
    v = float(raw_value)
    if not np.isfinite(v):
        return 0.0
    if feature_name in _PROXY_COST_FEATURES:
        return 1.0 / (1.0 + max(v, 0.0))
    return v


def _get_pair_feature_proxy(idx: int) -> float:
    global _PAIR_PROXY
    if _PAIR_PROXY is None:
        _PAIR_PROXY = {}
    feature_name = _resolve_proxy_feature_name()
    cache_key = ("feature", feature_name, int(idx))
    if cache_key in _PAIR_PROXY:
        return _PAIR_PROXY[cache_key]
    raw_value = _extract_proxy_feature_raw(idx, feature_name)
    val = _transform_proxy_feature_value(feature_name, raw_value)
    _PAIR_PROXY[cache_key] = val
    return val


def _dominates(p: Tuple[float, float], q: Tuple[float, float]) -> bool:
    return (p[0] >= q[0] and p[1] >= q[1]) and (p[0] > q[0] or p[1] > q[1])


def _nondominated_ranks(points: List[Tuple[float, float]]) -> List[int]:
    n = len(points)
    dominates = [set() for _ in range(n)]
    dominated_count = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(points[i], points[j]):
                dominates[i].add(j)
                dominated_count[j] += 1
            elif _dominates(points[j], points[i]):
                dominates[j].add(i)
                dominated_count[i] += 1

    ranks = [0] * n
    front = [i for i in range(n) if dominated_count[i] == 0]
    rank = 1
    while front:
        next_front = []
        for i in front:
            ranks[i] = rank
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        front = next_front
        rank += 1
    return ranks


def _ensure_pair_ranks():
    global _PAIR_RANKS
    if _PAIR_RANKS is not None:
        return
    points = []
    desc = f"Building Pareto ranks ({_pareto_second_metric_desc()})"
    for i in tqdm(range(len(_CANDIDATES)), desc=desc):
        points.append(_get_pair_metrics(i))
    _PAIR_RANKS = _nondominated_ranks(points)


def _pearson_corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("Correlation inputs must have the same length.")
    if len(xs) < 2:
        return 0.0
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 2:
        return 0.0
    x = x[mask]
    y = y[mask]
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    xd = x - x_mean
    yd = y - y_mean
    denom = float(np.sqrt(np.sum(xd * xd) * np.sum(yd * yd)))
    if denom <= 1e-15:
        return 0.0
    val = float(np.sum(xd * yd) / denom)
    if not np.isfinite(val):
        return 0.0
    return val


def _average_ranks(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and arr[order[j]] == arr[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman_corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("Correlation inputs must have the same length.")
    if len(xs) < 2:
        return 0.0
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 2:
        return 0.0
    x = x[mask]
    y = y[mask]
    rx = _average_ranks(list(x))
    ry = _average_ranks(list(y))
    return _pearson_corr(list(rx), list(ry))


def _rank_targets(ranks: List[int], target_mode: str) -> List[float]:
    mode = (target_mode or "inv_rank").strip().lower()
    if mode == "inv_rank":
        return [1.0 / max(float(r), 1.0) for r in ranks]
    if mode == "neg_rank":
        return [-float(r) for r in ranks]
    raise ValueError(f"Unsupported CORR_TARGET: {target_mode}")


def _score_rank_correlation(scores: List[float], ranks: List[int]) -> float:
    method = (getattr(config, "CORR_METHOD", "spearman") or "spearman").strip().lower()
    targets = _rank_targets(ranks, getattr(config, "CORR_TARGET", "inv_rank"))
    if method == "pearson":
        return _pearson_corr(scores, targets)
    if method == "spearman":
        return _spearman_corr(scores, targets)
    raise ValueError(f"Unsupported CORR_METHOD: {getattr(config, 'CORR_METHOD', '')}")


def _objective_from_stats(avg_rank: float, rank_score_corr: float) -> Tuple[float, float]:
    mode = (getattr(config, "EVAL_OBJECTIVE", "avg_rank") or "avg_rank").strip().lower()
    inv_avg_rank = (1.0 / avg_rank) if np.isfinite(avg_rank) and avg_rank > 0.0 else 0.0
    if mode == "avg_rank":
        score = -avg_rank if np.isfinite(avg_rank) else -100.0
        return score, inv_avg_rank
    if mode == "corr":
        return float(rank_score_corr), inv_avg_rank
    if mode == "combined":
        w_corr = max(float(getattr(config, "CORR_WEIGHT", 0.7)), 0.0)
        w_rank = max(float(getattr(config, "AVG_RANK_WEIGHT", 0.3)), 0.0)
        total = w_corr + w_rank
        if total <= 1e-15:
            w_corr = 0.7
            w_rank = 0.3
            total = 1.0
        score = (w_corr / total) * float(rank_score_corr) + (w_rank / total) * float(inv_avg_rank)
        return score, inv_avg_rank
    raise ValueError(f"Unsupported EVAL_OBJECTIVE: {getattr(config, 'EVAL_OBJECTIVE', '')}")


def _init():
    global _INIT_DONE, _BENCHMARKS, _CANDIDATES, _FEATURES, _SIM, _QERNEL_PAIRS, _MP
    if _INIT_DONE:
        return

    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    repro.SHOTS = config.SHOTS
    repro.BASELINE_USE_SCHEDULER = config.BASELINE_USE_SCHEDULER

    target_qubits = repro.UTIL_TO_QUBITS[config.TARGET_UTIL]
    _BENCHMARKS = repro.load_benchmarks()
    _CANDIDATES = repro.generate_candidate_pairs(_BENCHMARKS, target_qubits)
    random.shuffle(_CANDIDATES)

    _SIM = repro.MultiprogrammingSimulator()
    _MP = Multiprogrammer()
    _load_pair_metrics_from_csv()
    if getattr(config, "EVAL_RESTRICT_TO_CSV", False):
        before = len(_CANDIDATES)
        _CANDIDATES = [
            item for item in _CANDIDATES
            if f"{item[2]}+{item[3]}" in _PAIR_METRICS
        ]
        after = len(_CANDIDATES)
        if after == 0:
            raise ValueError(
                f"No candidates matched CSV keys for util={config.TARGET_UTIL}, shots={config.SHOTS}."
            )
        if after != before:
            print(
                f"[Evaluator] restricted candidate pool to CSV pairs: {after}/{before} "
                f"(util={int(config.TARGET_UTIL)} shots={int(config.SHOTS)})",
                flush=True,
            )
    _FEATURES, _QERNEL_PAIRS = _build_features(_CANDIDATES, _SIM.backend)
    _INIT_DONE = True


def _load_pair_metrics_from_csv():
    global _PAIR_METRICS, _PAIR_METADATA, _PAIR_METADATA_COLUMNS
    if _PAIR_METRICS:
        return
    csv_path = os.path.join(
        PROJECT_ROOT,
        "evaluation",
        "openevolve_pairing",
        "pairing_metadata",
        f"pair_metrics_util{config.TARGET_UTIL}_shots{config.SHOTS}.csv",
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Pair metrics CSV not found: {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _PAIR_METADATA_COLUMNS = reader.fieldnames or []
        for row in reader:
            key = f"{row['name_1']}+{row['name_2']}"
            _PAIR_METADATA[key] = row
            try:
                eff = float(row["effective_utilization"])
            except Exception:
                continue
            fid = float("nan")
            raw_fid = row.get("fidelity", "")
            if raw_fid is not None and str(raw_fid).strip() != "":
                try:
                    fid = float(raw_fid)
                except Exception:
                    fid = float("nan")
            _PAIR_METRICS[key] = (eff, fid)


def _build_features(candidates, backend):
    mp = Multiprogrammer()
    analyser = SupermarqFeaturesAnalysisPass()
    basic = BasicAnalysisPass()
    qernel_cache: Dict[str, Qernel] = {}

    def get_qernel(circ, label):
        if label in qernel_cache:
            return qernel_cache[label]
        q = Qernel(circ)
        basic.run(q)
        analyser.run(q)
        qernel_cache[label] = q
        return q

    features = []
    qernel_pairs = []
    for circ1, circ2, name1, name2 in candidates:
        q1 = get_qernel(circ1, name1)
        q2 = get_qernel(circ2, name2)
        meta1 = q1.get_metadata()
        meta2 = q2.get_metadata()
        eff_raw = mp.effective_utilization(q1, q2, backend)
        depth1 = meta1.get("depth", 0)
        depth2 = meta2.get("depth", 0)
        q1_qubits = meta1.get("num_qubits", circ1.num_qubits)
        q2_qubits = meta2.get("num_qubits", circ2.num_qubits)
        nonlocal1 = meta1.get("num_nonlocal_gates", 0)
        nonlocal2 = meta2.get("num_nonlocal_gates", 0)
        cnot1 = meta1.get("num_cnot_gates", 0)
        cnot2 = meta2.get("num_cnot_gates", 0)
        meas1 = meta1.get("num_measurements", 0)
        meas2 = meta2.get("num_measurements", 0)
        instr1 = meta1.get("number_instructions", 0)
        instr2 = meta2.get("number_instructions", 0)
        cc1 = meta1.get("num_connected_components", 0)
        cc2 = meta2.get("num_connected_components", 0)
        liveness1 = meta1.get("liveness", 0.0)
        liveness2 = meta2.get("liveness", 0.0)
        prog_comm1 = meta1.get("program_communication", 0.0)
        prog_comm2 = meta2.get("program_communication", 0.0)
        crit1 = meta1.get("critical_depth", 0.0)
        crit2 = meta2.get("critical_depth", 0.0)
        depth_ratio = min(depth1, depth2) / max(depth1, depth2, 1)
        qubit_ratio = min(q1_qubits, q2_qubits) / max(q1_qubits, q2_qubits, 1)
        nonlocal_ratio = min(nonlocal1, nonlocal2) / max(nonlocal1, nonlocal2, 1)
        cnot_ratio = min(cnot1, cnot2) / max(cnot1, cnot2, 1)
        meas_ratio = min(meas1, meas2) / max(meas1, meas2, 1)
        instr_ratio = min(instr1, instr2) / max(instr1, instr2, 1)
        features.append({
            "eff_util": eff_raw / 100.0,
            "eff_util_raw": eff_raw,
            "entanglement": mp.entanglementComparison(q1, q2),
            "measurement": mp.measurementComparison(q1, q2),
            "parallelism": mp.parallelismComparison(q1, q2),
            "depth_sim": mp.depthComparison(q1, q2),
            "depth_1": depth1,
            "depth_2": depth2,
            "depth_sum": depth1 + depth2,
            "depth_diff": abs(depth1 - depth2),
            "depth_ratio": depth_ratio,
            "num_qubits_1": q1_qubits,
            "num_qubits_2": q2_qubits,
            "qubits_sum": q1_qubits + q2_qubits,
            "qubits_diff": abs(q1_qubits - q2_qubits),
            "qubit_ratio": qubit_ratio,
            "nonlocal_1": nonlocal1,
            "nonlocal_2": nonlocal2,
            "nonlocal_sum": nonlocal1 + nonlocal2,
            "nonlocal_diff": abs(nonlocal1 - nonlocal2),
            "nonlocal_ratio": nonlocal_ratio,
            "cnot_1": cnot1,
            "cnot_2": cnot2,
            "cnot_sum": cnot1 + cnot2,
            "cnot_diff": abs(cnot1 - cnot2),
            "cnot_ratio": cnot_ratio,
            "measure_1": meas1,
            "measure_2": meas2,
            "measure_sum": meas1 + meas2,
            "measure_diff": abs(meas1 - meas2),
            "measure_ratio": meas_ratio,
            "instr_1": instr1,
            "instr_2": instr2,
            "instr_sum": instr1 + instr2,
            "instr_diff": abs(instr1 - instr2),
            "instr_ratio": instr_ratio,
            "connected_components_1": cc1,
            "connected_components_2": cc2,
            "connected_components_sum": cc1 + cc2,
            "liveness_1": liveness1,
            "liveness_2": liveness2,
            "liveness_avg": (liveness1 + liveness2) / 2.0,
            "program_comm_1": prog_comm1,
            "program_comm_2": prog_comm2,
            "program_comm_avg": (prog_comm1 + prog_comm2) / 2.0,
            "critical_depth_1": crit1,
            "critical_depth_2": crit2,
            "critical_depth_avg": (crit1 + crit2) / 2.0,
            "qubits_total": (circ1.num_qubits + circ2.num_qubits) / backend.num_qubits,
        })
        qernel_pairs.append((q1, q2))
    return features, qernel_pairs


def _score_baseline(f):
    # Match original MP behavior (eff_util dominates due to scale).
    return (f["eff_util_raw"] + f["entanglement"] + f["measurement"] + f["parallelism"]) / 4.0


def _get_pair_metrics(idx: int) -> Tuple[float, float]:
    _circ1, _circ2, name1, name2 = _CANDIDATES[idx]
    key = f"{name1}+{name2}"
    if key in _PAIR_METRICS:
        eff, fid = _PAIR_METRICS[key]
    else:
        if getattr(config, "EVAL_RESTRICT_TO_CSV", False):
            raise KeyError(
                f"Missing pair metrics in CSV for key={key} (util={config.TARGET_UTIL}, shots={config.SHOTS})"
            )
        _PAIR_METRICS[key] = (0.0, 0.0)
        eff, fid = _PAIR_METRICS[key]

    second_metric_mode = _resolve_pareto_second_metric()
    if second_metric_mode == "proxy":
        return float(eff), float(_get_pair_feature_proxy(idx))
    if not np.isfinite(float(fid)):
        raise ValueError(
            f"Missing fidelity for key={key} while OE_PARETO_SECOND_METRIC=fidelity. "
            "Use OE_PARETO_SECOND_METRIC=proxy or provide fidelity in pair_metrics CSV."
        )
    return float(eff), float(fid)

def _avg_bin_fidelity(points: List[Tuple[float, float]],
                      bin_edges: List[float]) -> float:
    if not points:
        return float("nan")
    bins = [(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]
    means = []
    for lo, hi in bins:
        fids = [fid for (util, fid) in points if lo <= util < hi]
        if fids:
            means.append(float(np.mean(fids)))
    if not means:
        return float("nan")
    return float(np.mean(means))


def _build_evaluation_result(payload):
    try:
        from openevolve.evaluation_result import EvaluationResult
    except Exception:
        return payload
    return EvaluationResult(**payload)


def _resolve_effective_top_k(n_items: int) -> Tuple[int, float]:
    if n_items <= 0:
        return 0, 0.0
    ratio_raw = float(getattr(config, "TOP_K_RATIO", 0.0) or 0.0)
    ratio = ratio_raw
    # Accept either fraction (0.1) or percentage (10).
    if ratio > 1.0 and ratio <= 100.0:
        ratio = ratio / 100.0
    if ratio > 0.0:
        k = int(np.ceil(float(n_items) * ratio))
        k = max(1, min(n_items, k))
        return k, ratio
    k = int(getattr(config, "TOP_K", 1) or 1)
    k = max(1, min(n_items, k))
    return k, 0.0


def _evaluate_active_task(candidate_kind: str | None,
                          candidate_score: Any,
                          log_prefix: str = "") -> Dict[str, Any]:
    def score_fn(idx, f):
        if config.EVAL_MODE == "original":
            return _score_baseline(f)
        if candidate_kind == "method":
            q1, q2 = _QERNEL_PAIRS[idx]
            return float(candidate_score(_MP, q1, q2, _SIM.backend, weighted=False, weights=[]))
        raise ValueError("Candidate scoring function not loaded.")

    scores = []
    for idx, f in enumerate(_FEATURES):
        try:
            scores.append(score_fn(idx, f))
        except Exception:
            scores.append(-1e9)

    effective_top_k, effective_top_k_ratio = _resolve_effective_top_k(len(scores))
    top_idx = np.argsort(scores)[-effective_top_k:]
    top_ranks = [_PAIR_RANKS[i] for i in top_idx]
    avg_rank = float(np.mean(top_ranks)) if top_ranks else float("inf")
    rank_score_corr = _score_rank_correlation(scores, _PAIR_RANKS)
    score, inv_avg_rank = _objective_from_stats(avg_rank, rank_score_corr)
    pareto_second_desc = _pareto_second_metric_desc()

    def _fmt4(val: float) -> str:
        return f"{val:.4f}".rstrip("0").rstrip(".")

    def _fmt_cell(val: str) -> str:
        if val is None:
            return ""
        try:
            return _fmt4(float(val))
        except Exception:
            return val

    total_counts = Counter(_PAIR_RANKS)
    selected_counts = Counter(_PAIR_RANKS[i] for i in top_idx)
    rank_lines = ["rank,total_pairs,selected_pairs,selected_pct"]
    for rank in sorted(total_counts.keys()):
        total = total_counts.get(rank, 0)
        selected = selected_counts.get(rank, 0)
        if selected == 0:
            continue
        pct = (selected / total) if total else 0.0
        rank_lines.append(f"{rank},{total},{selected},{_fmt4(pct)}")
    rank_distribution_csv = "\n".join(rank_lines)

    if config.EVAL_MODE == "both":
        base_scores = [_score_baseline(f) for f in _FEATURES]
        base_idx = np.argsort(base_scores)[-effective_top_k:]
        base_ranks = [_PAIR_RANKS[i] for i in base_idx]
        base_avg_rank = float(np.mean(base_ranks)) if base_ranks else float("inf")
        base_corr = _score_rank_correlation(base_scores, _PAIR_RANKS)
        base_obj, _base_inv = _objective_from_stats(base_avg_rank, base_corr)
        print(
            f"[Evaluator] {log_prefix}objective={config.EVAL_OBJECTIVE} baseline "
            f"util={int(config.TARGET_UTIL)} shots={int(config.SHOTS)} "
            f"top-k={int(effective_top_k)}"
            f"{f' ratio={effective_top_k_ratio:.4f}' if effective_top_k_ratio > 0.0 else ''} "
            f"avg-rank={base_avg_rank:.4f} corr={base_corr:.4f} score={base_obj:.4f} "
            f"pareto-second={pareto_second_desc}",
            flush=True,
        )

    pareto_mode = _resolve_pareto_second_metric()
    include_fidelity_in_artifacts = pareto_mode == "fidelity"
    if include_fidelity_in_artifacts:
        top_pairs_lines = [
            "name_1,name_2,eff_util,fidelity,pareto_second_metric,pareto_second_value,pareto_rank,score"
        ]
    else:
        top_pairs_lines = [
            "name_1,name_2,eff_util,pareto_second_metric,pareto_second_value,pareto_rank,score"
        ]
    for i in top_idx:
        _c1, _c2, n1, n2 = _CANDIDATES[i]
        eff, second_val = _get_pair_metrics(i)
        rnk = _PAIR_RANKS[i]
        if include_fidelity_in_artifacts:
            _eff_raw, raw_fid = _PAIR_METRICS.get(f"{n1}+{n2}", (eff, float("nan")))
            top_pairs_lines.append(
                f"{n1},{n2},{_fmt4(eff)},{_fmt4(raw_fid)},{pareto_second_desc},"
                f"{_fmt4(second_val)},{rnk},{_fmt4(scores[i])}"
            )
        else:
            top_pairs_lines.append(
                f"{n1},{n2},{_fmt4(eff)},{pareto_second_desc},{_fmt4(second_val)},{rnk},{_fmt4(scores[i])}"
            )
    top_pairs_csv = "\n".join(top_pairs_lines)

    top_rank_pairs_lines = []
    if _PAIR_METADATA_COLUMNS:
        base_cols = ("name_1", "name_2", "effective_utilization")
        if include_fidelity_in_artifacts:
            base_cols = base_cols + ("fidelity",)
        kept_cols = [
            col for col in _PAIR_METADATA_COLUMNS
            if col in base_cols or col.startswith("m1_") or col.startswith("m2_")
        ]
        header = kept_cols + ["pareto_rank", "score"]
        top_rank_pairs_lines.append(",".join(header))
        rank_cutoff = effective_top_k
        for i, rnk in enumerate(_PAIR_RANKS):
            if rnk > rank_cutoff:
                continue
            _c1, _c2, n1, n2 = _CANDIDATES[i]
            key = f"{n1}+{n2}"
            row = _PAIR_METADATA.get(key)
            if not row:
                continue
            values = [_fmt_cell(row.get(col, "")) for col in kept_cols]
            values += [str(rnk), _fmt4(scores[i])]
            top_rank_pairs_lines.append(",".join(values))
    top_rank_pairs_csv = "\n".join(top_rank_pairs_lines)

    objective_summary_csv = "\n".join([
        "objective,corr_method,corr_target,top_k,top_k_ratio,avg_rank,inv_avg_rank,rank_score_corr,score",
        f"{config.EVAL_OBJECTIVE},{config.CORR_METHOD},{config.CORR_TARGET},{int(effective_top_k)},"
        f"{effective_top_k_ratio:.6f},"
        f"{_fmt4(avg_rank if np.isfinite(avg_rank) else float('nan'))},"
        f"{_fmt4(inv_avg_rank)},{_fmt4(rank_score_corr)},{_fmt4(score)}",
    ])
    proxy_feature = _resolve_proxy_feature_name() if pareto_mode == "proxy" else ""
    pareto_metric_config_csv = "\n".join([
        "pareto_second_metric,proxy_feature",
        f"{pareto_mode},{proxy_feature}",
    ])

    print(
        f"[Evaluator] {log_prefix}objective={config.EVAL_OBJECTIVE} "
        f"util={int(config.TARGET_UTIL)} shots={int(config.SHOTS)} "
        f"top-k={int(effective_top_k)}"
        f"{f' ratio={effective_top_k_ratio:.4f}' if effective_top_k_ratio > 0.0 else ''} "
        f"avg-rank={avg_rank:.4f} corr={rank_score_corr:.4f} score={score:.4f} "
        f"pareto-second={pareto_second_desc}",
        flush=True,
    )

    return {
        "metrics": {
            "score": score,
            "combined_score": score,
            "avg_rank": avg_rank,
            "inv_avg_rank": inv_avg_rank,
            "rank_score_corr": rank_score_corr,
            "top_k": int(effective_top_k),
            "top_k_ratio": float(effective_top_k_ratio),
            "pareto_second_metric": pareto_second_desc,
        },
        "artifacts": {
            "rank_distribution_csv": rank_distribution_csv,
            "top_pairs_metrics_csv": top_pairs_csv,
            "top_rank_pairs_all_columns_csv": top_rank_pairs_csv,
            "objective_summary_csv": objective_summary_csv,
            "pareto_metric_config_csv": pareto_metric_config_csv,
        },
    }


def _baseline_metrics_for_top_k(effective_top_k: int) -> Dict[str, float]:
    base_scores = [_score_baseline(f) for f in _FEATURES]
    if not base_scores:
        return {
            "score": -100.0,
            "avg_rank": float("inf"),
            "inv_avg_rank": 0.0,
            "rank_score_corr": 0.0,
            "top_k": 0,
        }
    k = int(effective_top_k or 0)
    k = max(1, min(len(base_scores), k))
    base_idx = np.argsort(base_scores)[-k:]
    base_ranks = [_PAIR_RANKS[i] for i in base_idx]
    base_avg_rank = float(np.mean(base_ranks)) if base_ranks else float("inf")
    base_corr = _score_rank_correlation(base_scores, _PAIR_RANKS)
    base_obj, base_inv = _objective_from_stats(base_avg_rank, base_corr)
    return {
        "score": float(base_obj),
        "avg_rank": float(base_avg_rank),
        "inv_avg_rank": float(base_inv),
        "rank_score_corr": float(base_corr),
        "top_k": int(k),
    }


def _resolve_multi_util_agg_mode() -> str:
    raw = (getattr(config, "MULTI_UTIL_AGG", "mean") or "mean").strip().lower()
    if raw in ("", "mean", "avg", "average", "raw_mean"):
        return "mean"
    if raw in ("norm_to_baseline", "normalized", "normalized_mean", "baseline_norm", "norm"):
        return "norm_to_baseline"
    raise ValueError(f"Unsupported MULTI_UTIL_AGG: {raw}")


def _merge_artifact_csv_by_util(
    task_rows: List[Tuple[int, Dict[str, Any]]],
    artifact_key: str,
    shots: int,
) -> str:
    merged_payloads: List[Tuple[int, List[Dict[str, str]], List[str]]] = []
    all_headers: List[str] = []

    for util, row in task_rows:
        blob = str(row.get("artifacts", {}).get(artifact_key, "") or "").strip()
        if not blob:
            continue
        reader = csv.DictReader(blob.splitlines())
        fieldnames = [str(c) for c in (reader.fieldnames or []) if c]
        if not fieldnames:
            continue
        rows = [{k: (v if v is not None else "") for k, v in r.items()} for r in reader]
        merged_payloads.append((int(util), rows, fieldnames))
        for col in fieldnames:
            if col not in all_headers:
                all_headers.append(col)

    if not merged_payloads or not all_headers:
        return ""

    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=["util", "shots"] + all_headers)
    writer.writeheader()

    for util, rows, _fieldnames in merged_payloads:
        for row in rows:
            out_row = {"util": str(int(util)), "shots": str(int(shots))}
            for col in all_headers:
                out_row[col] = row.get(col, "")
            writer.writerow(out_row)

    return out.getvalue().strip("\r\n")


def evaluate(path):
    candidate_kind = None
    candidate_score = None
    if config.EVAL_MODE in ("candidate", "both"):
        candidate_kind, candidate_score = _load_score_fn(path)

    eval_utils = list(getattr(config, "EVAL_UTILS", [int(config.TARGET_UTIL)]) or [int(config.TARGET_UTIL)])
    eval_utils = [int(u) for u in eval_utils]
    eval_shots = int(getattr(config, "EVAL_SHOTS", int(config.SHOTS)))
    agg_mode = _resolve_multi_util_agg_mode()

    if len(eval_utils) == 1 and eval_utils[0] == int(config.TARGET_UTIL) and eval_shots == int(config.SHOTS):
        _init()
        _ensure_pair_ranks()
        return _build_evaluation_result(_evaluate_active_task(candidate_kind, candidate_score))

    orig_target_util = int(config.TARGET_UTIL)
    orig_shots = int(config.SHOTS)
    task_rows: List[Tuple[int, Dict[str, Any]]] = []

    for util in eval_utils:
        config.TARGET_UTIL = int(util)
        config.SHOTS = int(eval_shots)
        _reset_runtime_state()
        _init()
        _ensure_pair_ranks()
        task_row = _evaluate_active_task(candidate_kind, candidate_score, log_prefix=f"[multi {util}] ")
        if agg_mode == "norm_to_baseline":
            top_k = int(task_row["metrics"].get("top_k", 0))
            task_row["baseline_metrics"] = _baseline_metrics_for_top_k(top_k)
        task_rows.append((util, task_row))

    # Restore caller-visible config.
    config.TARGET_UTIL = orig_target_util
    config.SHOTS = orig_shots
    _reset_runtime_state()

    avg_ranks = [row["metrics"]["avg_rank"] for _util, row in task_rows]
    corrs = [row["metrics"]["rank_score_corr"] for _util, row in task_rows]
    raw_avg_rank_mean = float(np.mean(avg_ranks)) if avg_ranks else float("inf")
    raw_avg_rank_std = float(np.std(avg_ranks)) if len(avg_ranks) > 1 else 0.0

    if agg_mode == "norm_to_baseline":
        norm_avg_ranks = []
        for _util, row in task_rows:
            m = row["metrics"]
            base = row.get("baseline_metrics", {})
            cand_avg = float(m.get("avg_rank", float("nan")))
            base_avg = float(base.get("avg_rank", float("nan")))
            if np.isfinite(cand_avg) and np.isfinite(base_avg) and base_avg > 0.0:
                ratio = cand_avg / base_avg
            else:
                ratio = float("nan")
            m["baseline_avg_rank"] = base_avg
            m["avg_rank_over_baseline"] = ratio
            norm_avg_ranks.append(ratio)
        valid_norm = [v for v in norm_avg_ranks if np.isfinite(v)]
        if valid_norm:
            avg_rank = float(np.mean(valid_norm))
            avg_rank_std = float(np.std(valid_norm)) if len(valid_norm) > 1 else 0.0
        else:
            avg_rank = raw_avg_rank_mean
            avg_rank_std = raw_avg_rank_std
        rank_agg_metric = "mean(avg_rank/baseline_avg_rank)"
    else:
        avg_rank = raw_avg_rank_mean
        avg_rank_std = raw_avg_rank_std
        rank_agg_metric = "mean(avg_rank)"

    rank_score_corr = float(np.mean(corrs)) if corrs else 0.0
    score, inv_avg_rank = _objective_from_stats(avg_rank, rank_score_corr)
    corr_std = float(np.std(corrs)) if len(corrs) > 1 else 0.0
    pareto_second_desc = _pareto_second_metric_desc()

    merged_rank_distribution_csv = _merge_artifact_csv_by_util(
        task_rows, "rank_distribution_csv", eval_shots
    )
    merged_top_pairs_csv = _merge_artifact_csv_by_util(
        task_rows, "top_pairs_metrics_csv", eval_shots
    )
    merged_top_rank_pairs_csv = _merge_artifact_csv_by_util(
        task_rows, "top_rank_pairs_all_columns_csv", eval_shots
    )
    details_lines = [
        "util,shots,objective,corr_method,corr_target,top_k,top_k_ratio,avg_rank,inv_avg_rank,rank_score_corr,score"
    ]
    details_extended_lines = [
        "util,shots,objective,corr_method,corr_target,top_k,top_k_ratio,avg_rank,inv_avg_rank,rank_score_corr,score,baseline_avg_rank,avg_rank_over_baseline"
    ]
    for util, row in task_rows:
        m = row["metrics"]
        details_lines.append(
            f"{int(util)},{int(eval_shots)},{config.EVAL_OBJECTIVE},{config.CORR_METHOD},"
            f"{config.CORR_TARGET},{int(m.get('top_k', 0))},{float(m.get('top_k_ratio', 0.0)):.6f},"
            f"{m['avg_rank']:.6f},{m['inv_avg_rank']:.6f},"
            f"{m['rank_score_corr']:.6f},{m['score']:.6f}"
        )
        base_avg_rank = m.get("baseline_avg_rank", float("nan"))
        rank_over_base = m.get("avg_rank_over_baseline", float("nan"))
        details_extended_lines.append(
            f"{int(util)},{int(eval_shots)},{config.EVAL_OBJECTIVE},{config.CORR_METHOD},"
            f"{config.CORR_TARGET},{int(m.get('top_k', 0))},{float(m.get('top_k_ratio', 0.0)):.6f},"
            f"{m['avg_rank']:.6f},{m['inv_avg_rank']:.6f},"
            f"{m['rank_score_corr']:.6f},{m['score']:.6f},"
            f"{float(base_avg_rank):.6f},{float(rank_over_base):.6f}"
        )
    top_k_values = [int(row["metrics"].get("top_k", 0)) for _util, row in task_rows]
    top_k_ratio_values = [float(row["metrics"].get("top_k_ratio", 0.0)) for _util, row in task_rows]
    top_k_cell = str(top_k_values[0]) if top_k_values and len(set(top_k_values)) == 1 else "mixed"
    top_k_ratio_cell = (
        f"{top_k_ratio_values[0]:.6f}"
        if top_k_ratio_values and len(set(round(v, 10) for v in top_k_ratio_values)) == 1
        else "mixed"
    )
    objective_summary_csv = "\n".join([
        "objective,corr_method,corr_target,top_k,top_k_ratio,avg_rank,inv_avg_rank,rank_score_corr,score",
        f"{config.EVAL_OBJECTIVE},{config.CORR_METHOD},{config.CORR_TARGET},{top_k_cell},{top_k_ratio_cell},"
        f"{avg_rank:.6f},{inv_avg_rank:.6f},{rank_score_corr:.6f},{score:.6f}",
    ])
    objective_agg_summary_csv = "\n".join([
        "multi_util_agg,rank_agg_metric,avg_rank_used,avg_rank_used_std,avg_rank_raw_mean,avg_rank_raw_std,rank_score_corr_mean,rank_score_corr_std",
        f"{agg_mode},{rank_agg_metric},{avg_rank:.6f},{avg_rank_std:.6f},{raw_avg_rank_mean:.6f},{raw_avg_rank_std:.6f},{rank_score_corr:.6f},{corr_std:.6f}",
    ])

    for util, row in task_rows:
        m = row["metrics"]
        if agg_mode == "norm_to_baseline":
            print(
                f"[Evaluator] [multi-agg] util={int(util)} shots={int(eval_shots)} "
                f"avg-rank={float(m['avg_rank']):.4f} base-avg-rank={float(m.get('baseline_avg_rank', float('nan'))):.4f} "
                f"avg-rank/base={float(m.get('avg_rank_over_baseline', float('nan'))):.4f} "
                f"corr={float(m['rank_score_corr']):.4f} score={float(m['score']):.4f} "
                f"pareto-second={pareto_second_desc}",
                flush=True,
            )
        else:
            print(
                f"[Evaluator] [multi-agg] util={int(util)} shots={int(eval_shots)} "
                f"avg-rank={float(m['avg_rank']):.4f} corr={float(m['rank_score_corr']):.4f} "
                f"score={float(m['score']):.4f} pareto-second={pareto_second_desc}",
                flush=True,
            )

    print(
        f"[Evaluator] [multi-agg] objective={config.EVAL_OBJECTIVE} "
        f"utils={','.join(str(u) for u in eval_utils)} shots={int(eval_shots)} "
        f"top-k={top_k_cell}"
        f"{f' ratio={top_k_ratio_cell}' if top_k_ratio_cell != 'mixed' else ' ratio=mixed'} "
        f"aggregation={rank_agg_metric} "
        f"avg-rank={avg_rank:.4f}±{avg_rank_std:.4f} "
        f"(raw-mean={raw_avg_rank_mean:.4f}±{raw_avg_rank_std:.4f}) "
        f"corr={rank_score_corr:.4f}±{corr_std:.4f} score={score:.4f} mode={agg_mode} "
        f"pareto-second={pareto_second_desc}",
        flush=True,
    )

    return _build_evaluation_result({
        "metrics": {
            "score": score,
            "combined_score": score,
            "avg_rank": avg_rank,
            "inv_avg_rank": inv_avg_rank,
            "rank_score_corr": rank_score_corr,
            "avg_rank_std": avg_rank_std,
            "rank_score_corr_std": corr_std,
            "avg_rank_raw_mean": raw_avg_rank_mean,
            "avg_rank_raw_std": raw_avg_rank_std,
            "multi_util_agg_mode": agg_mode,
            "multi_util_rank_agg_metric": rank_agg_metric,
            "num_utils": len(eval_utils),
            "pareto_second_metric": pareto_second_desc,
        },
        "artifacts": {
            "rank_distribution_csv": merged_rank_distribution_csv,
            "top_pairs_metrics_csv": merged_top_pairs_csv,
            "top_rank_pairs_all_columns_csv": merged_top_rank_pairs_csv,
            # "objective_summary_csv": objective_summary_csv,
            # "objective_summary_by_util_csv": "\n".join(details_lines),
            # "objective_summary_by_util_extended_csv": "\n".join(details_extended_lines),
            # "objective_agg_summary_csv": objective_agg_summary_csv,
        },
    })
