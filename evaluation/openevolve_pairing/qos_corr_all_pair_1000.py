#!/usr/bin/env python3
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

# Feature keys from evaluator._FEATURES
all_feature_keys = set()
for f in _e._FEATURES:
    for k, v in f.items():
        if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
            all_feature_keys.add(k)

# Gate-type derived metrics (subset of metadata)
base_keys = [
    "num_1q_gates",
    "num_2q_gates",
    "num_cnot_gates",
    "num_nonlocal_gates",
    "num_measurements",
    "depth",
    "number_instructions",
]

meta_keys = set()
for q1, q2 in _e._QERNEL_PAIRS:
    m1 = q1.get_metadata()
    m2 = q2.get_metadata()
    for k, v in m1.items():
        if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
            meta_keys.add(k)
    for k, v in m2.items():
        if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
            meta_keys.add(k)

meta_keys = sorted(meta_keys)

metrics = {k: [] for k in all_feature_keys}
for k in base_keys:
    for suffix in ("sum", "diff", "max", "avg", "ratio"):
        metrics.setdefault(f"gate_{k}_{suffix}", [])
for k in meta_keys:
    for suffix in ("sum", "diff", "max", "min", "avg", "ratio"):
        metrics.setdefault(f"meta_{k}_{suffix}", [])
metrics.setdefault("proxy_instr_diff_norm", [])
metrics.setdefault("proxy_instr_diff_inv", [])

def _append_feature_metrics(f):
    for k in all_feature_keys:
        v = f.get(k, float("nan"))
        if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
            metrics[k].append(float(v))
        else:
            metrics[k].append(float("nan"))

def _append_gate_metrics(m1, m2):
    for k in base_keys:
        v1 = float(m1.get(k, 0.0))
        v2 = float(m2.get(k, 0.0))
        s = v1 + v2
        d = abs(v1 - v2)
        mx = max(v1, v2)
        mn = min(v1, v2)
        avg = s / 2.0
        ratio = (mn / mx) if mx > 0 else 1.0
        # Prefix to avoid clashing with evaluator feature names (e.g., depth_sum).
        metrics.setdefault(f"gate_{k}_sum", []).append(s)
        metrics.setdefault(f"gate_{k}_diff", []).append(d)
        metrics.setdefault(f"gate_{k}_max", []).append(mx)
        metrics.setdefault(f"gate_{k}_avg", []).append(avg)
        metrics.setdefault(f"gate_{k}_ratio", []).append(ratio)

def _append_meta_metrics(m1, m2):
    for k in meta_keys:
        v1 = float(m1.get(k, 0.0))
        v2 = float(m2.get(k, 0.0))
        s = v1 + v2
        d = abs(v1 - v2)
        mx = max(v1, v2)
        mn = min(v1, v2)
        avg = s / 2.0
        ratio = (mn / mx) if mx > 0 else 1.0
        metrics.setdefault(f"meta_{k}_sum", []).append(s)
        metrics.setdefault(f"meta_{k}_diff", []).append(d)
        metrics.setdefault(f"meta_{k}_max", []).append(mx)
        metrics.setdefault(f"meta_{k}_min", []).append(mn)
        metrics.setdefault(f"meta_{k}_avg", []).append(avg)
        metrics.setdefault(f"meta_{k}_ratio", []).append(ratio)

def _append_proxy_metrics(m1, m2):
    instr1 = float(m1.get("number_instructions", 0.0))
    instr2 = float(m2.get("number_instructions", 0.0))
    instr_sum = instr1 + instr2
    instr_diff = abs(instr1 - instr2)
    instr_diff_norm = instr_diff / max(instr_sum, 1.0)
    proxy_norm = 1.0 - instr_diff_norm
    proxy_inv = 1.0 / (1.0 + instr_diff)
    metrics["proxy_instr_diff_norm"].append(proxy_norm)
    metrics["proxy_instr_diff_inv"].append(proxy_inv)

for i in tqdm(range(len(_e._CANDIDATES)), desc="Computing pair metrics"):
    q1, q2 = _e._QERNEL_PAIRS[i]
    m1 = q1.get_metadata()
    m2 = q2.get_metadata()
    fid = _e._get_pair_metrics(i)[1]
    fids.append(fid)

    _append_feature_metrics(_e._FEATURES[i])
    _append_gate_metrics(m1, m2)
    _append_meta_metrics(m1, m2)
    _append_proxy_metrics(m1, m2)

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

print(
    f"[All Pair Sweep] N={len(fids)} pairs, shots={config.SHOTS}"
)
print("metric\tcorr")
for k, c in results:
    print(f"{k}\t{c:.4f}")
