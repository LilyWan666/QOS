import math
import os
import sys
import numpy as np
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm optional
    def tqdm(x, **_kwargs):
        return x

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.openevolve_pairing import evaluator
import config

# Full sweep settings
config.CANDIDATE_LIMIT = None
config.SHOTS = 1000

_e = evaluator
_e._init()

# Collect fidelity per pair
fids = []
# Collect per-pair gate-type metrics
metrics = {}

base_keys = [
    "num_1q_gates",
    "num_2q_gates",
    "num_cnot_gates",
    "num_nonlocal_gates",
    "num_measurements",
    "depth",
    "number_instructions",
]

for i in tqdm(range(len(_e._CANDIDATES)), desc="Computing pair gate metrics"):
    q1, q2 = _e._QERNEL_PAIRS[i]
    m1 = q1.get_metadata()
    m2 = q2.get_metadata()
    fid = _e._get_pair_metrics(i)[1]
    fids.append(fid)

    for k in base_keys:
        v1 = float(m1.get(k, 0.0))
        v2 = float(m2.get(k, 0.0))
        s = v1 + v2
        d = abs(v1 - v2)
        mx = max(v1, v2)
        mn = min(v1, v2)
        avg = s / 2.0
        ratio = (mn / mx) if mx > 0 else 1.0
        # store
        metrics.setdefault(f"{k}_sum", []).append(s)
        metrics.setdefault(f"{k}_diff", []).append(d)
        metrics.setdefault(f"{k}_max", []).append(mx)
        metrics.setdefault(f"{k}_avg", []).append(avg)
        metrics.setdefault(f"{k}_ratio", []).append(ratio)

fids = np.array(fids, dtype=float)

results = []
for k, vals in metrics.items():
    vals = np.array(vals, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(fids)
    if mask.sum() < 3:
        continue
    v = vals[mask]
    y = fids[mask]
    if np.allclose(v, v[0]):
        continue
    corr = float(np.corrcoef(v, y)[0, 1])
    if math.isnan(corr):
        continue
    results.append((k, corr))

results.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"[GateType Pair Sweep] N={len(fids)} pairs, shots={config.SHOTS}")
print("metric\tcorr")
for k, c in results:
    print(f"{k}\t{c:.4f}")
