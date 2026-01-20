#!/usr/bin/env python3
"""
Smoke-test Multiprogrammer.process_qernels with simulated estimator outputs.

This avoids the full estimator pipeline by creating a minimal qernel_dict:
  - layout: simple disjoint (when possible) or identity layout
  - backend: FakeKolkataV2
  - fidelity: placeholder value (1.0)
"""

import os
import sys
import random

from qiskit_ibm_runtime.fake_provider import FakeKolkataV2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from reproduce_fig11 import load_benchmarks  # noqa: E402
from qos.types.types import Qernel  # noqa: E402
from qos.multiprogrammer.multiprogrammer import Multiprogrammer  # noqa: E402
from qos.multiprogrammer.tools import check_layout_overlap  # noqa: E402


def build_qernel_dict(benchmarks, target_qubits, max_qernels, backend):
    pool = []
    for name, sizes in benchmarks.items():
        for nq, circ in sizes.items():
            if nq <= target_qubits:
                pool.append((name, nq, circ))

    random.shuffle(pool)
    pool = pool[:max_qernels]

    qernel_dict = {}
    offset = 0
    for name, nq, circ in pool:
        q = Qernel(circ)
        q.edit_metadata({"label": f"{name}-{nq}"})

        if offset + nq <= backend.num_qubits:
            layout = list(range(offset, offset + nq))
            offset += nq
        else:
            layout = list(range(nq))

        qernel_dict[q] = [(layout, backend, 1.0)]

    return qernel_dict


if __name__ == "__main__":
    random.seed(42)

    target_qubits = int(os.getenv("TARGET_QUBITS", "16"))
    max_qernels = int(os.getenv("MAX_QERNELS", "12"))
    threshold = float(os.getenv("MATCH_THRESHOLD", "0.0"))

    backend = FakeKolkataV2()
    benchmarks = load_benchmarks()
    qernel_dict = build_qernel_dict(benchmarks, target_qubits, max_qernels, backend)

    print(f"Built {len(qernel_dict)} qernels (target<= {target_qubits} qubits)")
    mp = Multiprogrammer()
    selected = mp.process_qernels(qernel_dict, threshold=threshold, dry_run=True)

    if selected is None:
        print("No pair selected.")
        raise SystemExit(0)

    q1, q2, layout1, layout2, score, spatial_util, _ = selected
    label1 = q1.get_metadata().get("label", "q1")
    label2 = q2.get_metadata().get("label", "q2")
    overlap = check_layout_overlap(layout1, layout2)

    print("Selected pair:")
    print(f"  {label1} + {label2}")
    print(f"  matching_score={score:.4f} spatial_util={spatial_util:.4f}")
    print(f"  overlap={overlap} layout1={layout1} layout2={layout2}")
