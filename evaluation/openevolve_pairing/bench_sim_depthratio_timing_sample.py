#!/usr/bin/env python3
import time, csv, json, random, sys
from pathlib import Path

sys.path.insert(0, '/work/nvme/betu/lily/QOS')
sys.path.insert(0, '/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing')

import evaluator_orig as ev
import config

PLAN=[(30,1024,20),(60,1024,20),(88,1024,20),(30,8192,4),(60,8192,4),(88,8192,4)]
SEED=42


def reset_ev():
    ev._INIT_DONE=False
    ev._BENCHMARKS=None
    ev._CANDIDATES=None
    ev._FEATURES=None
    ev._QERNEL_PAIRS=None
    ev._MP=None
    ev._PAIR_METRICS={}
    ev._PAIR_RANKS=None
    ev._PAIR_PROXY=None
    ev._SIM=None

rng=random.Random(SEED)
rows=[]
for util, shots, kreq in PLAN:
    config.TARGET_UTIL=util
    config.SHOTS=shots
    config.CANDIDATE_LIMIT=None
    ev.config.TARGET_UTIL=util
    ev.config.SHOTS=shots
    ev.config.CANDIDATE_LIMIT=None
    reset_ev()

    t0=time.perf_counter(); ev._init(); t_init=time.perf_counter()-t0
    n=len(ev._CANDIDATES)
    k=min(kreq,n)
    idxs=rng.sample(list(range(n)), k)

    t1=time.perf_counter()
    for i in idxs:
        ev._get_pair_metrics(i)
    t_sim=time.perf_counter()-t1

    t2=time.perf_counter()
    _=sum(float(ev._FEATURES[i].get('depth_ratio',0.0)) for i in idxs)
    t_depth=time.perf_counter()-t2

    row={
        'util':util,
        'shots':shots,
        'num_pairs_total':n,
        'num_pairs_sampled':k,
        'init_sec':t_init,
        'sim_sample_total_sec':t_sim,
        'sim_sec_per_pair_est':t_sim/max(k,1),
        'depth_sample_total_sec':t_depth,
        'depth_sec_per_pair_est':t_depth/max(k,1),
    }
    rows.append(row)
    print(row, flush=True)

out_dir=Path('/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing/openevolve_output/_timing_sim1024_8192_depthratio_sample_20260303')
out_dir.mkdir(parents=True, exist_ok=True)
out_csv=out_dir/'timing_sample_summary.csv'
out_json=out_dir/'timing_sample_summary.json'
with open(out_csv,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
with open(out_json,'w') as f:
    json.dump(rows,f,indent=2)
print(f'WROTE {out_csv}')
print(f'WROTE {out_json}')
