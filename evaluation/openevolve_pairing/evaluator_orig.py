"""Evaluator for openevolve pairing search."""

import os
import sys
import random
import importlib.util
from typing import Dict, List, Tuple

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
_PAIR_RANKS = None
_PAIR_PROXY = None
_SIM = None


def _load_score_fn(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
    depth = max(meta1.get("depth", 0), meta2.get("depth", 0))
    cnot = meta1.get("num_cnot_gates", 0) + meta2.get("num_cnot_gates", 0)
    nonlocal_gates = meta1.get("num_nonlocal_gates", 0) + meta2.get("num_nonlocal_gates", 0)
    meas = meta1.get("num_measurements", 0) + meta2.get("num_measurements", 0)
    comm = (meta1.get("program_communication", 0.0) + meta2.get("program_communication", 0.0)) / 2.0
    liveness = (meta1.get("liveness", 0.0) + meta2.get("liveness", 0.0)) / 2.0
    ent_ratio = (meta1.get("entanglement_ratio", 0.0) + meta2.get("entanglement_ratio", 0.0)) / 2.0

    depth_factor = 1.0 / (1.0 + 0.01 * depth)
    cnot_factor = 1.0 / (1.0 + 0.02 * cnot)
    nonlocal_factor = 1.0 / (1.0 + 0.02 * nonlocal_gates)
    meas_factor = 1.0 / (1.0 + 0.02 * meas)
    comm_factor = 1.0 - min(comm, 1.0) * 0.3
    live_factor = 1.0 - min(liveness, 1.0) * 0.2
    ent_factor = 1.0 - min(ent_ratio, 1.0) * 0.2

    proxy = (
        depth_factor +
        cnot_factor +
        nonlocal_factor +
        meas_factor +
        comm_factor +
        live_factor +
        ent_factor
    ) / 7.0
    return max(0.0, min(1.0, proxy))


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
    _INIT_DONE = True


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
    if config.LAYOUT_MODE == "qos":
        f1, f2, eff, _l1, _l2 = _SIM.run_qos_mp(circ1, circ2)
    else:
        f1, f2, eff, _l1, _l2 = _SIM.run_baseline_mp(circ1, circ2)
    _PAIR_METRICS[key] = (eff / 100.0, (f1 + f2) / 2)
    return _PAIR_METRICS[key]


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
    top_fids = [_get_pair_metrics(i)[1] for i in top_idx]
    avg_fid = float(np.mean(top_fids)) if top_fids else float("nan")
    base = (1.0 / avg_rank) if avg_rank > 0.0 else 0.0
    score = base + config.FID_WEIGHT * (avg_fid if not np.isnan(avg_fid) else 0.0)

    if config.EVAL_MODE == "both":
        base_scores = [_score_baseline(f) for f in _FEATURES]
        base_idx = np.argsort(base_scores)[-config.TOP_K:]
        base_ranks = [_PAIR_RANKS[i] for i in base_idx]
        base_avg_rank = float(np.mean(base_ranks)) if base_ranks else float("inf")
        base_fids = [_get_pair_metrics(i)[1] for i in base_idx]
        base_avg_fid = float(np.mean(base_fids)) if base_fids else float("nan")
        print(
            f"[Evaluator] baseline avg-rank: {base_avg_rank:.4f} "
            f"avg-fid: {base_avg_fid:.4f}",
            flush=True,
        )

    return {"score": score}
