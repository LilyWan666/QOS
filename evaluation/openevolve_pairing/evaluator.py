"""Evaluator for openevolve pairing search."""

import os
import sys
import random
import re
import csv
import importlib.util
import types
from typing import Dict, List, Tuple
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
    if idx in _PAIR_PROXY:
        return _PAIR_PROXY[idx]
    q1, q2 = _QERNEL_PAIRS[idx]
    val = _fast_fidelity_proxy(q1, q2)
    _PAIR_PROXY[idx] = val
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
    for i in tqdm(range(len(_CANDIDATES)), desc="Simulating pairs"):
        points.append(_get_pair_metrics(i))
    _PAIR_RANKS = _nondominated_ranks(points)


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
    _FEATURES, _QERNEL_PAIRS = _build_features(_CANDIDATES, _SIM.backend)
    _load_pair_metrics_from_csv()
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
                fid = float(row["fidelity"])
            except Exception:
                continue
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
    circ1, circ2, name1, name2 = _CANDIDATES[idx]
    key = f"{name1}+{name2}"
    if key in _PAIR_METRICS:
        return _PAIR_METRICS[key]
    _PAIR_METRICS[key] = (0.0, 0.0)
    return _PAIR_METRICS[key]

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


def evaluate(path):
    _init()
    _ensure_pair_ranks()

    candidate_kind = None
    candidate_score = None
    if config.EVAL_MODE in ("candidate", "both"):
        candidate_kind, candidate_score = _load_score_fn(path)

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

    top_idx = np.argsort(scores)[-config.TOP_K:]
    top_ranks = [_PAIR_RANKS[i] for i in top_idx]
    avg_rank = float(np.mean(top_ranks)) if top_ranks else float("inf")
    score = -avg_rank if avg_rank != float("inf") else -100.0

    def _fmt4(val: float) -> str:
        return f"{val:.4f}".rstrip("0").rstrip(".")

    def _fmt_cell(val: str) -> str:
        if val is None:
            return ""
        try:
            return _fmt4(float(val))
        except Exception:
            return val

    # Rank (order) distribution: total pairs per rank vs selected Top-K pairs per rank.
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
        base_idx = np.argsort(base_scores)[-config.TOP_K:]
        base_ranks = [_PAIR_RANKS[i] for i in base_idx]
        base_avg_rank = float(np.mean(base_ranks)) if base_ranks else float("inf")
        print(
            f"[Evaluator] baseline avg-rank: {base_avg_rank:.4f}",
            flush=True,
        )

    top_pairs_lines = ["name_1,name_2,eff_util,fidelity,pareto_rank,score"]
    for i in top_idx:
        _c1, _c2, n1, n2 = _CANDIDATES[i]
        eff, fid = _get_pair_metrics(i)
        rnk = _PAIR_RANKS[i]
        top_pairs_lines.append(f"{n1},{n2},{_fmt4(eff)},{_fmt4(fid)},{rnk},{_fmt4(scores[i])}")
    top_pairs_csv = "\n".join(top_pairs_lines)

    top_rank_pairs_lines = []
    if _PAIR_METADATA_COLUMNS:
        kept_cols = [
            col for col in _PAIR_METADATA_COLUMNS
            if col in ("name_1", "name_2", "effective_utilization", "fidelity")
            or col.startswith("m1_")
            or col.startswith("m2_")
        ]
        header = kept_cols + ["pareto_rank", "score"]
        top_rank_pairs_lines.append(",".join(header))
        rank_cutoff = config.TOP_K
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

    return _build_evaluation_result({
        "metrics": {"score": score, "combined_score": score},
        "artifacts": {
            "rank_distribution_csv": rank_distribution_csv,
            "top_pairs_metrics_csv": top_pairs_csv,
            "top_rank_pairs_all_columns_csv": top_rank_pairs_csv,
        },
    })
