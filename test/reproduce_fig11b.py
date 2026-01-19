#!/usr/bin/env python3
"""
Fast reproduction of Figure 11(b): Effective Utilization only.

This script avoids noisy/ideal simulation and computes effective utilization
directly from circuit depth/qubits via the QOS multiprogrammer.

Outputs:
  - figure_11b.png
  - figure_11_depth_cdf.png
  - figure_11_subcircuit_depth_cdf.png
  - figure_11b_results.json
"""

import os
import sys
import json
import random
import warnings
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qiskit_ibm_runtime.fake_provider import FakeKolkataV2

# Suppress warnings
warnings.filterwarnings("ignore")

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Reuse pairing + utilization logic from main script
from reproduce_fig11 import (  # noqa: E402
    load_benchmarks,
    select_baseline_pairs,
    select_qos_pairs,
    compute_effective_utilization,
    create_depth_cdf,
    create_subcircuit_depth_cdf,
    UTIL_TO_QUBITS,
    N_PAIRS_PER_UTIL as DEFAULT_PAIRS_PER_UTIL,
)

# Configure matplotlib
rcParams["font.size"] = 12
rcParams["axes.labelsize"] = 14
rcParams["axes.titlesize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["legend.fontsize"] = 11

# Allow override: FIG11B_PAIRS=12 python reproduce_fig11b.py
env_pairs = os.getenv("FIG11B_PAIRS")
if env_pairs is None:
    N_PAIRS_PER_UTIL = None
else:
    N_PAIRS_PER_UTIL = int(env_pairs)


def run_effective_utilization_only() -> Dict[str, Any]:
    print("=" * 60, flush=True)
    print("Reproducing Figure 11(b): Effective Utilization (fast)", flush=True)
    print("=" * 60, flush=True)

    print("\n[1/3] Loading benchmark circuits...", flush=True)
    benchmarks = load_benchmarks()
    print(f"Loaded benchmarks: {list(benchmarks.keys())}", flush=True)

    print("\n[2/3] Initializing backend...", flush=True)
    backend = FakeKolkataV2()

    results = {
        "baseline_eff_util": {util: [] for util in UTIL_TO_QUBITS.keys()},
        "qos_eff_util": {util: [] for util in UTIL_TO_QUBITS.keys()},
        "baseline_depth_diff": {util: [] for util in UTIL_TO_QUBITS.keys()},
        "qos_depth_diff": {util: [] for util in UTIL_TO_QUBITS.keys()},
        "baseline_subcircuit_depths": [],
        "qos_subcircuit_depths": [],
    }

    print("\n[3/3] Computing effective utilization...", flush=True)
    for util, target_qubits in UTIL_TO_QUBITS.items():
        print(f"\n  Utilization {util}% ({target_qubits} qubits):", flush=True)

        baseline_pairs = select_baseline_pairs(
            benchmarks, target_qubits, N_PAIRS_PER_UTIL
        )
        for i, (circ1, circ2, name1, name2) in enumerate(baseline_pairs):
            print(
                f"    Baseline Pair {i+1}/{len(baseline_pairs)}: "
                f"{name1} + {name2}",
                flush=True,
            )
            eff = compute_effective_utilization(circ1, circ2, backend)
            results["baseline_eff_util"][util].append(eff)
            results["baseline_depth_diff"][util].append(abs(circ1.depth() - circ2.depth()))
            results["baseline_subcircuit_depths"].extend([circ1.depth(), circ2.depth()])

        qos_pairs = select_qos_pairs(
            benchmarks, target_qubits, N_PAIRS_PER_UTIL, backend
        )
        for i, (circ1, circ2, name1, name2) in enumerate(qos_pairs):
            print(
                f"    QOS Pair {i+1}/{len(qos_pairs)}: "
                f"{name1} + {name2}",
                flush=True,
            )
            eff = compute_effective_utilization(circ1, circ2, backend)
            results["qos_eff_util"][util].append(eff)
            results["qos_depth_diff"][util].append(abs(circ1.depth() - circ2.depth()))
            results["qos_subcircuit_depths"].extend([circ1.depth(), circ2.depth()])

        if results["baseline_eff_util"][util]:
            print(
                f"    Baseline mean: {np.mean(results['baseline_eff_util'][util]):.2f}",
                flush=True,
            )
        if results["qos_eff_util"][util]:
            print(
                f"    QOS mean: {np.mean(results['qos_eff_util'][util]):.2f}",
                flush=True,
            )

    return results


def create_figure_b(results: Dict[str, Any]):
    """Create Figure 11(b) only."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    utils = sorted(UTIL_TO_QUBITS.keys())
    x = np.arange(len(utils))
    width = 0.35

    baseline_util_means = [np.mean(results["baseline_eff_util"][u]) for u in utils]
    qos_util_means = [np.mean(results["qos_eff_util"][u]) for u in utils]

    ax.bar(x - width / 2, baseline_util_means, width, label="Baseline", color="steelblue")
    ax.bar(x + width / 2, qos_util_means, width, label="QOS", color="forestgreen")

    ax.plot(x, list(utils), "r--", marker="o", label="Target")

    ax.set_xlabel("Target Utilization (%)")
    ax.set_ylabel("Effective Utilization (%)")
    ax.set_title("Figure 11(b): Effective Utilization")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{u}%" for u in utils])
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(PROJECT_ROOT, "test", "figure_11b.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}", flush=True)
    plt.close(fig)


def save_results(results: Dict[str, Any]):
    serializable_results = {
        "baseline_eff_util": {
            str(k): [float(x) for x in v] for k, v in results["baseline_eff_util"].items()
        },
        "qos_eff_util": {
            str(k): [float(x) for x in v] for k, v in results["qos_eff_util"].items()
        },
        "baseline_depth_diff": {
            str(k): [float(x) for x in v] for k, v in results["baseline_depth_diff"].items()
        },
        "qos_depth_diff": {
            str(k): [float(x) for x in v] for k, v in results["qos_depth_diff"].items()
        },
        "baseline_subcircuit_depths": [float(x) for x in results["baseline_subcircuit_depths"]],
        "qos_subcircuit_depths": [float(x) for x in results["qos_subcircuit_depths"]],
    }
    output_path = os.path.join(PROJECT_ROOT, "test", "figure_11b_results.json")
    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Results saved to: {output_path}", flush=True)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    results = run_effective_utilization_only()
    create_figure_b(results)
    create_depth_cdf(results)
    create_subcircuit_depth_cdf(results)
    save_results(results)
