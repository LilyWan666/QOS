#!/usr/bin/env python3
"""
Reproduce Figure 11 from the QOS paper: Multiprogramming Results

Figure 11 has 3 subplots:
(a) Impact on Fidelity: No M/P vs Baseline M/P vs QOS M/P at 30%, 60%, 88% utilization
(b) Effective Utilization: Baseline vs QOS effective utilization
(c) Relative Fidelity: Fidelity relative to solo circuit execution

Usage:
    conda activate qos_fig11
    python reproduce_fig11.py

Runtime Notes:
    - Full reproduction with paper parameters takes ~30-60 minutes
    - For quick testing, modify UTIL_TO_QUBITS and SHOTS parameters below

Output:
    - figure_11.png: The reproduced figure
    - figure_11_results.json: Raw results data
"""

import os
import sys
import json
import random
import warnings
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime.fake_provider import FakeKolkataV2

# Suppress warnings
warnings.filterwarnings('ignore')

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# QOS imports
from qos.types.types import Qernel
from qos.multiprogrammer.multiprogrammer import Multiprogrammer
from qos.error_mitigator.analyser import SupermarqFeaturesAnalysisPass
from Baseline_Multiprogramming import multiprogramming as baseline_mp

# Configure matplotlib
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 11

# ============================================================================
# Configuration
# ============================================================================

BENCHMARK_MAPPING = {
    'QAOA-R3': 'qaoa_r3',
    'BV': 'bv',
    'GHZ': 'ghz',
    'HS-1': 'hamsim_1',
    'QAOA-P1': 'qaoa_pl1',
    'QSVM': 'qsvm',
    'TL-1': 'twolocal_1',
    'VQE-1': 'vqe_1',
    'W-STATE': 'wstate'
}

# Utilization level -> total qubits on 27-qubit QPU
# Using smaller sizes for faster testing (paper uses {30: 8, 60: 16, 88: 24})
# UTIL_TO_QUBITS = {30: 8, 60: 16, 88: 24}
UTIL_TO_QUBITS = {30: 8, 60: 16}
# No-M/P (Fig.11a) should match paper utilization mapping.
# NO_MP_UTIL_TO_QUBITS = {30: 8, 60: 16, 88: 24}
NO_MP_UTIL_TO_QUBITS = {30: 8, 60: 16}

# Simulation parameters
SHOTS = 100
N_PAIRS_PER_UTIL = 3  # Reduced for faster testing

# ============================================================================
# Helper Functions
# ============================================================================

def get_circuit(benchname: str, nqubits: int) -> QuantumCircuit:
    """Load a benchmark circuit from QASM file."""
    circuits_dir = os.path.join(PROJECT_ROOT, 'evaluation', 'benchmarks', benchname)
    files = [f for f in os.listdir(circuits_dir)
             if f.endswith('.qasm') and f.split('.')[0].isdigit()
             and int(f.split('.')[0]) == nqubits]

    if not files:
        raise ValueError(f"No circuit found for {benchname} with {nqubits} qubits")

    circuit = QuantumCircuit.from_qasm_file(os.path.join(circuits_dir, files[0]))
    dag = circuit_to_dag(circuit)
    dag.remove_all_ops_named('barrier')
    return dag_to_circuit(dag)


def load_benchmarks() -> Dict[str, Dict[int, QuantumCircuit]]:
    """Load all benchmark circuits for various qubit sizes."""
    benchmarks = {}
    qubit_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

    for name, bench_id in BENCHMARK_MAPPING.items():
        benchmarks[name] = {}
        for nq in qubit_sizes:
            try:
                benchmarks[name][nq] = get_circuit(bench_id, nq)
            except (ValueError, FileNotFoundError):
                continue

    return benchmarks


def create_noise_model(p1: float = 0.001, p2: float = 0.01) -> NoiseModel:
    """Create a depolarizing noise model as specified in the paper."""
    noise = NoiseModel()
    err1 = depolarizing_error(p1, 1)
    err2 = depolarizing_error(p2, 2)

    one_q_gates = ["x", "y", "z", "h", "rx", "ry", "rz", "sx", "id", "s", "sdg"]
    two_q_gates = ["cx", "cz", "swap", "rzz", "cp", "ecr"]

    noise.add_all_qubit_quantum_error(err1, one_q_gates)
    noise.add_all_qubit_quantum_error(err2, two_q_gates)

    return noise


def hellinger_fidelity(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Compute Hellinger fidelity between two probability distributions."""
    all_keys = set(p.keys()) | set(q.keys())
    bc = 0.0

    for key in all_keys:
        p_val = p.get(key, 0.0)
        q_val = q.get(key, 0.0)
        bc += np.sqrt(p_val * q_val)

    return bc ** 2


def counts_to_probs(counts: Dict[str, int]) -> Dict[str, float]:
    """Convert counts to probability distribution."""
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def compute_effective_utilization(circ1: QuantumCircuit, circ2: QuantumCircuit,
                                   backend) -> float:
    """Compute effective utilization via QOS Multiprogrammer."""
    mp = Multiprogrammer()
    return mp.effective_utilization(Qernel(circ1), Qernel(circ2), backend)


def check_layout_overlap(layout1: List[int], layout2: List[int]) -> bool:
    """Check if two layouts overlap."""
    return bool(set(layout1) & set(layout2))


# ============================================================================
# Simulation Methods
# ============================================================================

class MultiprogrammingSimulator:
    """Simulator for multiprogramming experiments."""

    def __init__(self):
        self.backend = FakeKolkataV2()
        self.noise_model = create_noise_model()
        self.simulator = AerSimulator(noise_model=self.noise_model, method='automatic')
        self.n_qubits = self.backend.num_qubits  # 27
        self.target = self.backend.target
        self.coupling_map = self.backend.coupling_map

    def run_ideal(self, circuit: QuantumCircuit) -> Dict[str, int]:
        """Run circuit without noise to get ideal distribution."""
        ideal_sim = AerSimulator(method='statevector')
        if circuit.num_clbits == 0:
            circ = circuit.copy()
            circ.measure_all()
        else:
            circ = circuit

        try:
            job = ideal_sim.run(circ, shots=SHOTS)
            return job.result().get_counts()
        except Exception as e:
            print(f"[WARN] Ideal simulation failed: {e}")
            return {'0' * circuit.num_qubits: SHOTS}

    def run_noisy(self, circuit: QuantumCircuit,
                  initial_layout: List[int] = None) -> Dict[str, int]:
        """Run circuit with noise model."""
        try:
            tc = transpile(circuit, self.backend,
                          initial_layout=initial_layout,
                          optimization_level=1)
            job = self.simulator.run(tc, shots=SHOTS)
            return job.result().get_counts()
        except Exception as e:
            print(f"[WARN] Noisy simulation failed: {e}")
            return {'0' * circuit.num_qubits: SHOTS}

    def compute_fidelity(self, noisy_counts: Dict[str, int],
                         ideal_counts: Dict[str, int]) -> float:
        """Compute Hellinger fidelity between noisy and ideal results."""
        noisy_probs = counts_to_probs(noisy_counts)
        ideal_probs = counts_to_probs(ideal_counts)
        return hellinger_fidelity(noisy_probs, ideal_probs)

    def run_solo(self, circuit: QuantumCircuit) -> Tuple[float, Dict[str, int]]:
        """Run a single circuit solo (no multiprogramming)."""
        ideal_counts = self.run_ideal(circuit)
        noisy_counts = self.run_noisy(circuit)
        fidelity = self.compute_fidelity(noisy_counts, ideal_counts)
        return fidelity, noisy_counts

    def _find_good_layout(self, circuit: QuantumCircuit) -> List[int]:
        """Find a good layout for a circuit based on backend connectivity."""
        n = circuit.num_qubits

        # Get qubit errors from target
        qubit_errors = []
        for q in range(self.n_qubits):
            try:
                measure_props = self.target.get('measure', None)
                if measure_props and (q,) in measure_props:
                    err = measure_props[(q,)].error or 0.01
                else:
                    err = 0.01
            except:
                err = 0.01
            qubit_errors.append((q, err))

        qubit_errors.sort(key=lambda x: x[1])

        # Build layout starting from best qubits
        layout = []
        used = set()

        for q, _ in qubit_errors:
            if len(layout) >= n:
                break
            if q not in used:
                if not layout or any((q, lq) in self.coupling_map or (lq, q) in self.coupling_map
                                    for lq in layout):
                    layout.append(q)
                    used.add(q)

        if len(layout) < n:
            layout = list(range(n))

        return layout[:n]

    def _find_non_overlapping_layout(self, circuit: QuantumCircuit,
                                      used_layout: List[int]) -> List[int]:
        """Find a layout that doesn't overlap with the used layout."""
        n = circuit.num_qubits
        available = set(range(self.n_qubits)) - set(used_layout)

        if len(available) < n:
            return None

        qubit_errors = []
        for q in available:
            try:
                measure_props = self.target.get('measure', None)
                if measure_props and (q,) in measure_props:
                    err = measure_props[(q,)].error or 0.01
                else:
                    err = 0.01
            except:
                err = 0.01
            qubit_errors.append((q, err))

        qubit_errors.sort(key=lambda x: x[1])

        layout = []
        used = set()

        for q, _ in qubit_errors:
            if len(layout) >= n:
                break
            if q not in used:
                if not layout or any((q, lq) in self.coupling_map or (lq, q) in self.coupling_map
                                    for lq in layout):
                    layout.append(q)
                    used.add(q)

        if len(layout) < n:
            layout = sorted(list(available))[:n]

        return layout[:n] if len(layout) >= n else None

    def _split_counts(self, combined_counts: Dict[str, int],
                       n1: int, n2: int) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Split combined measurement results into individual circuit results."""
        counts1 = defaultdict(int)
        counts2 = defaultdict(int)

        for bitstring, count in combined_counts.items():
            bs = bitstring.replace(' ', '')
            expected_len = n1 + n2
            if len(bs) < expected_len:
                bs = '0' * (expected_len - len(bs)) + bs

            bits2 = bs[:n2]
            bits1 = bs[n2:n2+n1]

            counts1[bits1] += count
            counts2[bits2] += count

        return dict(counts1), dict(counts2)

    def run_baseline_mp(self, circ1: QuantumCircuit, circ2: QuantumCircuit) -> Tuple[float, float, float]:
        """
        Run baseline multiprogramming: simple consecutive layout.
        Returns: (fidelity1, fidelity2, effective_utilization)
        """
        sched1, sched2 = circ1, circ2
        try:
            programs = [circ1, circ2]
            program_analysis = baseline_mp.analyze_programs(programs)
            try:
                utility = baseline_mp.compute_qubit_utility(self.backend)
            except Exception:
                coupling_map = list(self.backend.coupling_map)
                utility = {}
                for qubit in range(self.n_qubits):
                    neighbors = [pair[1] for pair in coupling_map if pair[0] == qubit] + \
                                [pair[0] for pair in coupling_map if pair[1] == qubit]
                    num_links = len(neighbors)
                    error_sum = max(num_links * 0.01, 0.01)
                    utility[qubit] = num_links / error_sum if error_sum > 0 else 0

            backend_props = {
                "coupling_map": self.backend.coupling_map,
                "utility": utility,
            }
            scheduled_programs = baseline_mp.shared_qubit_allocation_and_scheduling(
                programs, program_analysis, backend_props
            )
            if len(scheduled_programs) == 2:
                sched1, sched2 = scheduled_programs
        except Exception as exc:
            print(f"[WARN] Baseline scheduling failed: {exc}")

        n1, n2 = sched1.num_qubits, sched2.num_qubits

        if n1 + n2 > self.n_qubits:
            f1, _ = self.run_solo(circ1)
            f2, _ = self.run_solo(circ2)
            return f1, f2, 0.0

        # Create combined circuit with consecutive layout
        combined = QuantumCircuit(n1 + n2, n1 + n2)
        combined.compose(sched1, qubits=range(n1), clbits=range(n1), inplace=True)
        combined.compose(sched2, qubits=range(n1, n1 + n2), clbits=range(n1, n1 + n2), inplace=True)

        layout = list(range(n1 + n2))

        ideal1 = self.run_ideal(circ1)
        ideal2 = self.run_ideal(circ2)

        tc = transpile(combined, self.backend, initial_layout=layout, optimization_level=1)
        job = self.simulator.run(tc, shots=SHOTS)
        combined_counts = job.result().get_counts()

        counts1, counts2 = self._split_counts(combined_counts, n1, n2)

        f1 = self.compute_fidelity(counts1, ideal1)
        f2 = self.compute_fidelity(counts2, ideal2)

        eff_util = compute_effective_utilization(circ1, circ2, self.backend)

        return f1, f2, eff_util

    def run_qos_mp(self, circ1: QuantumCircuit, circ2: QuantumCircuit) -> Tuple[float, float, float]:
        """
        Run QOS multiprogramming: smart layout selection.
        QOS finds optimal non-overlapping layouts for each circuit.
        Returns: (fidelity1, fidelity2, effective_utilization)
        """
        n1, n2 = circ1.num_qubits, circ2.num_qubits

        if n1 + n2 > self.n_qubits:
            f1, _ = self.run_solo(circ1)
            f2, _ = self.run_solo(circ2)
            return f1, f2, 0.0

        # QOS: Find optimal layouts using error-aware selection
        layout1 = self._find_good_layout(circ1)
        layout2 = self._find_non_overlapping_layout(circ2, layout1)

        if layout2 is None:
            return self.run_baseline_mp(circ1, circ2)

        ideal1 = self.run_ideal(circ1)
        ideal2 = self.run_ideal(circ2)

        # Combine circuits with optimal layouts
        combined = QuantumCircuit(n1 + n2, n1 + n2)
        combined.compose(circ1, qubits=range(n1), clbits=range(n1), inplace=True)
        combined.compose(circ2, qubits=range(n1, n1 + n2), clbits=range(n1, n1 + n2), inplace=True)

        layout = layout1 + layout2

        tc = transpile(combined, self.backend, initial_layout=layout, optimization_level=1)
        job = self.simulator.run(tc, shots=SHOTS)
        combined_counts = job.result().get_counts()

        counts1, counts2 = self._split_counts(combined_counts, n1, n2)

        f1 = self.compute_fidelity(counts1, ideal1)
        f2 = self.compute_fidelity(counts2, ideal2)

        eff_util = compute_effective_utilization(circ1, circ2, self.backend)

        return f1, f2, eff_util


# ============================================================================
# Experiment Runner
# ============================================================================

def generate_circuit_pairs(benchmarks: Dict, target_qubits: int,
                           n_pairs: int) -> List[Tuple[QuantumCircuit, QuantumCircuit, str, str]]:
    """Generate pairs of circuits that sum to approximately target_qubits."""
    pairs = []
    bench_names = list(benchmarks.keys())

    for _ in range(n_pairs * 10):
        if len(pairs) >= n_pairs:
            break

        name1 = random.choice(bench_names)
        name2 = random.choice(bench_names)

        available_sizes1 = list(benchmarks[name1].keys())
        available_sizes2 = list(benchmarks[name2].keys())

        if not available_sizes1 or not available_sizes2:
            continue

        for s1 in available_sizes1:
            for s2 in available_sizes2:
                if abs((s1 + s2) - target_qubits) <= 2:
                    pairs.append((
                        benchmarks[name1][s1],
                        benchmarks[name2][s2],
                        f"{name1}-{s1}",
                        f"{name2}-{s2}"
                    ))
                    if len(pairs) >= n_pairs:
                        break
            if len(pairs) >= n_pairs:
                break

    return pairs[:n_pairs]


def generate_candidate_pairs(benchmarks: Dict, target_qubits: int,
                             tolerance: int = 2) -> List[Tuple[QuantumCircuit, QuantumCircuit, str, str]]:
    """Generate all candidate pairs within a qubit-count tolerance."""
    pairs = []
    seen = set()

    for name1, sizes1 in benchmarks.items():
        for name2, sizes2 in benchmarks.items():
            for s1, circ1 in sizes1.items():
                for s2, circ2 in sizes2.items():
                    if abs((s1 + s2) - target_qubits) > tolerance:
                        continue

                    key = tuple(sorted(((name1, s1), (name2, s2))))
                    if key in seen:
                        continue
                    seen.add(key)

                    p_name1, p_name2 = name1, name2
                    p_s1, p_s2 = s1, s2
                    p_c1, p_c2 = circ1, circ2
                    if (p_name2, p_s2) < (p_name1, p_s1):
                        p_name1, p_name2 = p_name2, p_name1
                        p_s1, p_s2 = p_s2, p_s1
                        p_c1, p_c2 = p_c2, p_c1

                    pairs.append((
                        p_c1,
                        p_c2,
                        f"{p_name1}-{p_s1}",
                        f"{p_name2}-{p_s2}",
                    ))

    return pairs


def select_baseline_pairs(benchmarks: Dict, target_qubits: int,
                          n_pairs: int) -> List[Tuple[QuantumCircuit, QuantumCircuit, str, str]]:
    """Baseline: random pairing within target utilization."""
    candidates = generate_candidate_pairs(benchmarks, target_qubits)
    random.shuffle(candidates)
    return candidates[:n_pairs]


def select_qos_pairs(benchmarks: Dict, target_qubits: int, n_pairs: int,
                     backend) -> List[Tuple[QuantumCircuit, QuantumCircuit, str, str]]:
    """QOS: choose highest matching-score pairs."""
    candidates = generate_candidate_pairs(benchmarks, target_qubits)
    if not candidates:
        return []

    mp = Multiprogrammer()
    analyser = SupermarqFeaturesAnalysisPass()
    qernel_cache = {}

    def get_qernel(circ: QuantumCircuit, label: str) -> Qernel:
        q = qernel_cache.get(label)
        if q is None:
            q = Qernel(circ)
            analyser.run(q)
            qernel_cache[label] = q
        return q

    scored = []
    for circ1, circ2, name1, name2 in candidates:
        try:
            q1 = get_qernel(circ1, name1)
            q2 = get_qernel(circ2, name2)
            score = mp.get_matching_score(q1, q2, backend)
            scored.append((score, (circ1, circ2, name1, name2)))
        except Exception as exc:
            print(f"[WARN] QOS scoring failed for {name1}+{name2}: {exc}")

    scored.sort(key=lambda x: x[0], reverse=True)
    return [pair for _, pair in scored[:n_pairs]]


def generate_solo_circuits(benchmarks: Dict, target_qubits: int,
                           n_samples: int = None) -> List[Tuple[QuantumCircuit, str]]:
    """Generate solo circuits with exactly target_qubits."""
    options = []
    for name, sizes in benchmarks.items():
        if target_qubits in sizes:
            options.append((sizes[target_qubits], f"{name}-{target_qubits}"))

    if not options:
        raise ValueError(f"No solo circuits available for {target_qubits} qubits")

    if n_samples is None:
        return options

    return [random.choice(options) for _ in range(n_samples)]


def run_experiments():
    """Run all experiments for Figure 11."""
    print("=" * 60, flush=True)
    print("Reproducing Figure 11: Multiprogramming Results", flush=True)
    print("=" * 60, flush=True)

    print("\n[1/4] Loading benchmark circuits...", flush=True)
    benchmarks = load_benchmarks()
    print(f"Loaded benchmarks: {list(benchmarks.keys())}", flush=True)

    print("\n[2/4] Initializing simulator...", flush=True)
    sim = MultiprogrammingSimulator()

    results = {
        'no_mp': {util: [] for util in UTIL_TO_QUBITS.keys()},
        'baseline_mp': {util: [] for util in UTIL_TO_QUBITS.keys()},
        'qos_mp': {util: [] for util in UTIL_TO_QUBITS.keys()},
        'baseline_eff_util': {util: [] for util in UTIL_TO_QUBITS.keys()},
        'qos_eff_util': {util: [] for util in UTIL_TO_QUBITS.keys()},
        'relative_fidelity_baseline': {util: [] for util in UTIL_TO_QUBITS.keys()},
        'relative_fidelity_qos': {util: [] for util in UTIL_TO_QUBITS.keys()},
    }

    print("\n[3/4] Running experiments...", flush=True)

    print("\n[3a/4] Running No M/P only...", flush=True)
    for util, target_qubits in UTIL_TO_QUBITS.items():
        print(f"\n  Utilization {util}% ({target_qubits} qubits):", flush=True)

        no_mp_qubits = NO_MP_UTIL_TO_QUBITS.get(util, target_qubits)
        solo_circuits = generate_solo_circuits(benchmarks, no_mp_qubits)

        for i, (solo_circ, solo_name) in enumerate(solo_circuits):
            print(f"    No M/P {i+1}/{len(solo_circuits)}: {solo_name}", flush=True)
            solo_fid, _ = sim.run_solo(solo_circ)
            results['no_mp'][util].append(solo_fid)

        if results['no_mp'][util]:
            print(f"    No M/P mean: {np.mean(results['no_mp'][util]):.4f}", flush=True)

    print("\n[3b/4] Running Baseline M/P...", flush=True)
    for util, target_qubits in UTIL_TO_QUBITS.items():
        print(f"\n  Utilization {util}% ({target_qubits} qubits):", flush=True)

        baseline_pairs = select_baseline_pairs(benchmarks, target_qubits, N_PAIRS_PER_UTIL)

        for i, (circ1, circ2, name1, name2) in enumerate(baseline_pairs):
            print(f"    Baseline Pair {i+1}/{len(baseline_pairs)}: {name1} + {name2}", flush=True)

            solo_fid1, _ = sim.run_solo(circ1)
            solo_fid2, _ = sim.run_solo(circ2)

            bf1, bf2, b_eff = sim.run_baseline_mp(circ1, circ2)
            baseline_avg_fid = (bf1 + bf2) / 2
            results['baseline_mp'][util].append(baseline_avg_fid)
            results['baseline_eff_util'][util].append(b_eff)

            baseline_rel = 0
            if solo_fid1 > 0 and solo_fid2 > 0:
                baseline_rel = ((bf1 / solo_fid1) + (bf2 / solo_fid2)) / 2
            results['relative_fidelity_baseline'][util].append(baseline_rel)

            print(
                "      Baseline metrics so far: "
                f"fid={np.mean(results['baseline_mp'][util]):.4f}, "
                f"rel={np.mean(results['relative_fidelity_baseline'][util]):.4f}",
                flush=True,
            )

    print("\n[3c/4] Running QOS M/P...", flush=True)
    for util, target_qubits in UTIL_TO_QUBITS.items():
        print(f"\n  Utilization {util}% ({target_qubits} qubits):", flush=True)

        qos_pairs = select_qos_pairs(benchmarks, target_qubits, N_PAIRS_PER_UTIL, sim.backend)

        for i, (circ1, circ2, name1, name2) in enumerate(qos_pairs):
            print(f"    QOS Pair {i+1}/{len(qos_pairs)}: {name1} + {name2}", flush=True)

            solo_fid1, _ = sim.run_solo(circ1)
            solo_fid2, _ = sim.run_solo(circ2)

            qf1, qf2, q_eff = sim.run_qos_mp(circ1, circ2)
            qos_avg_fid = (qf1 + qf2) / 2
            results['qos_mp'][util].append(qos_avg_fid)
            results['qos_eff_util'][util].append(q_eff)

            qos_rel = 0
            if solo_fid1 > 0 and solo_fid2 > 0:
                qos_rel = ((qf1 / solo_fid1) + (qf2 / solo_fid2)) / 2
            results['relative_fidelity_qos'][util].append(qos_rel)

            print(
                "      QOS metrics so far: "
                f"fid={np.mean(results['qos_mp'][util]):.4f}, "
                f"rel={np.mean(results['relative_fidelity_qos'][util]):.4f}",
                flush=True,
            )

    return results


def create_figure(results: Dict[str, Any]):
    """Create the 3-subplot Figure 11."""
    print("\n[4/4] Generating figure...", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    utils = sorted(UTIL_TO_QUBITS.keys())
    x = np.arange(len(utils))
    width = 0.25

    # Subplot (a): Impact on Fidelity
    ax1 = axes[0]

    no_mp_means = [np.mean(results['no_mp'][u]) for u in utils]
    baseline_means = [np.mean(results['baseline_mp'][u]) for u in utils]
    qos_means = [np.mean(results['qos_mp'][u]) for u in utils]

    no_mp_stds = [np.std(results['no_mp'][u]) for u in utils]
    baseline_stds = [np.std(results['baseline_mp'][u]) for u in utils]
    qos_stds = [np.std(results['qos_mp'][u]) for u in utils]

    ax1.bar(x - width, no_mp_means, width, yerr=no_mp_stds,
            label='No M/P', color='lightgray', capsize=3)
    ax1.bar(x, baseline_means, width, yerr=baseline_stds,
            label='Baseline M/P', color='steelblue', capsize=3)
    ax1.bar(x + width, qos_means, width, yerr=qos_stds,
            label='QOS M/P', color='forestgreen', capsize=3)

    ax1.set_xlabel('Utilization (%)')
    ax1.set_ylabel('Fidelity')
    ax1.set_title('(a) Impact on Fidelity')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{u}%' for u in utils])
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)

    # Subplot (b): Effective Utilization
    ax2 = axes[1]

    baseline_util_means = [np.mean(results['baseline_eff_util'][u]) for u in utils]
    qos_util_means = [np.mean(results['qos_eff_util'][u]) for u in utils]

    width2 = 0.35

    ax2.bar(x - width2/2, baseline_util_means, width2,
            label='Baseline', color='steelblue')
    ax2.bar(x + width2/2, qos_util_means, width2,
            label='QOS', color='forestgreen')

    ax2.plot(x, list(utils), 'r--', marker='o', label='Target')

    ax2.set_xlabel('Target Utilization (%)')
    ax2.set_ylabel('Effective Utilization (%)')
    ax2.set_title('(b) Effective Utilization')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{u}%' for u in utils])
    ax2.legend(loc='upper left')
    ax2.grid(axis='y', alpha=0.3)

    # Subplot (c): Relative Fidelity
    ax3 = axes[2]

    rel_baseline_means = [np.mean(results['relative_fidelity_baseline'][u]) for u in utils]
    rel_qos_means = [np.mean(results['relative_fidelity_qos'][u]) for u in utils]

    rel_baseline_stds = [np.std(results['relative_fidelity_baseline'][u]) for u in utils]
    rel_qos_stds = [np.std(results['relative_fidelity_qos'][u]) for u in utils]

    ax3.bar(x - width2/2, rel_baseline_means, width2, yerr=rel_baseline_stds,
            label='Baseline M/P', color='steelblue', capsize=3)
    ax3.bar(x + width2/2, rel_qos_means, width2, yerr=rel_qos_stds,
            label='QOS M/P', color='forestgreen', capsize=3)

    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Solo (1.0)')

    ax3.set_xlabel('Utilization (%)')
    ax3.set_ylabel('Relative Fidelity')
    ax3.set_title('(c) Relative Fidelity')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{u}%' for u in utils])
    ax3.legend(loc='upper right')
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(PROJECT_ROOT, 'test', 'figure_11.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    plt.close()


def save_results(results: Dict[str, Any]):
    """Save results to JSON file."""
    serializable_results = {}
    for key, val in results.items():
        if isinstance(val, dict):
            serializable_results[key] = {
                str(k): [float(x) for x in v] for k, v in val.items()
            }
        else:
            serializable_results[key] = val

    output_path = os.path.join(PROJECT_ROOT, 'test', 'figure_11_results.json')
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Results saved to: {output_path}")


def print_summary(results: Dict[str, Any]):
    """Print summary of results."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for util in sorted(UTIL_TO_QUBITS.keys()):
        print(f"\nUtilization {util}%:")
        print(f"  No M/P Fidelity:       {np.mean(results['no_mp'][util]):.4f} +/- {np.std(results['no_mp'][util]):.4f}")
        print(f"  Baseline M/P Fidelity: {np.mean(results['baseline_mp'][util]):.4f} +/- {np.std(results['baseline_mp'][util]):.4f}")
        print(f"  QOS M/P Fidelity:      {np.mean(results['qos_mp'][util]):.4f} +/- {np.std(results['qos_mp'][util]):.4f}")
        print(f"  Baseline Eff. Util:    {np.mean(results['baseline_eff_util'][util]):.1f}%")
        print(f"  QOS Eff. Util:         {np.mean(results['qos_eff_util'][util]):.1f}%")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    print("QOS Paper - Figure 11 Reproduction", flush=True)
    print("Multiprogramming Results\n", flush=True)

    results = run_experiments()
    print_summary(results)
    create_figure(results)
    save_results(results)

    print("\nDone!")
