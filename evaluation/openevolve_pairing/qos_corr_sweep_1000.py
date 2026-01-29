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
config.CANDIDATE_LIMIT = 120
config.SHOTS = 1000

_e = evaluator
_e._init()
_e._ensure_pair_ranks()

fids = []
for i in tqdm(range(len(_e._CANDIDATES)), desc="Computing pair metrics"):
    fids.append(_e._get_pair_metrics(i)[1])

fids = np.array(fids, dtype=float)

all_keys = set()
for f in _e._FEATURES:
    for k, v in f.items():
        if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
            all_keys.add(k)

results = []
for k in sorted(all_keys):
    vals = []
    for f in _e._FEATURES:
        v = f.get(k, float('nan'))
        if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
            vals.append(float(v))
        else:
            vals.append(float('nan'))
    vals = np.array(vals, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(fids)
    if mask.sum() < 3:
        continue
    v = vals[mask]
    y = fids[mask]
    if np.allclose(v, v[0]):
        corr = float('nan')
    else:
        corr = float(np.corrcoef(v, y)[0, 1])
    if math.isnan(corr):
        continue
    results.append((k, corr))

results.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"[Sweep] N={len(fids)} pairs, shots={config.SHOTS}, candidate_limit={config.CANDIDATE_LIMIT}")
print("metric\tcorr")
for k, c in results:
    print(f"{k}\t{c:.4f}")
