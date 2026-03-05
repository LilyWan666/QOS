#!/usr/bin/env python3
import csv
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.openevolve_pairing import config, evaluator

ROOT = Path('/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing')
OUT = ROOT / 'openevolve_output' / '_proxy_multimethod_trainutil_scatter_physical8196_20260302'
PAIR = ROOT / 'pairing_metadata'

TARGET_ORIG = ROOT / 'target_orig.py'

METHODS = {
    'depth_ratio_avg306088': {
        'proxy': 'depth_ratio',
        'train_util': 'avg306088',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260224_014256_avg306088_proxy_depth_ratio_iter200_20260224_014243/best/best_program.py',
    },
    'depth_ratio_u30': {
        'proxy': 'depth_ratio',
        'train_util': 'u30',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260220_002831_u30_proxy_depth_ratio_iter200_20260220_002816/best/best_program.py',
    },
    'depth_ratio_u60': {
        'proxy': 'depth_ratio',
        'train_util': 'u60',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260220_002830_u60_proxy_depth_ratio_iter200_20260220_002816/best/best_program.py',
    },
    'depth_ratio_u88': {
        'proxy': 'depth_ratio',
        'train_util': 'u88',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260220_002831_u88_proxy_depth_ratio_iter200_20260220_002816/best/best_program.py',
    },
    'instr_sum_avg306088': {
        'proxy': 'instr_sum',
        'train_util': 'avg306088',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260224_014800_avg306088_proxy_instr_sum_iter200_20260224_014450/best/best_program.py',
    },
    'instr_sum_u30': {
        'proxy': 'instr_sum',
        'train_util': 'u30',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260224_164709_u30_proxy_instr_sum_iter200_20260224_164553/best/best_program.py',
    },
    'instr_sum_u60': {
        'proxy': 'instr_sum',
        'train_util': 'u60',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260224_164922_u60_proxy_instr_sum_iter200_20260224_164553/best/best_program.py',
    },
    'instr_sum_u88': {
        'proxy': 'instr_sum',
        'train_util': 'u88',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260224_165718_u88_proxy_instr_sum_iter200_20260224_164553/best/best_program.py',
    },
    'measure_diff_avg306088': {
        'proxy': 'measure_diff',
        'train_util': 'avg306088',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260226_035035_avg306088_proxy_measure_diff_iter200_20260226_010631/best/best_program.py',
    },
    'measure_diff_u30': {
        'proxy': 'measure_diff',
        'train_util': 'u30',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260226_213415_u30_proxy_measure_diff_iter200_singleutil_cnot_measurediff_20260226_152810/best/best_program.py',
    },
    'measure_diff_u60': {
        'proxy': 'measure_diff',
        'train_util': 'u60',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260226_213418_u60_proxy_measure_diff_iter200_singleutil_cnot_measurediff_20260226_152810/best/best_program.py',
    },
    'measure_diff_u88': {
        'proxy': 'measure_diff',
        'train_util': 'u88',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260226_213415_u88_proxy_measure_diff_iter200_singleutil_cnot_measurediff_20260226_152810/best/best_program.py',
    },
    'cnot_ratio_avg306088': {
        'proxy': 'cnot_ratio',
        'train_util': 'avg306088',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260224_014258_avg306088_proxy_cnot_ratio_iter200_20260224_014243/best/best_program.py',
    },
    'cnot_ratio_u30': {
        'proxy': 'cnot_ratio',
        'train_util': 'u30',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260226_213423_u30_proxy_cnot_ratio_iter200_singleutil_cnot_measurediff_20260226_152810/best/best_program.py',
    },
    'cnot_ratio_u60': {
        'proxy': 'cnot_ratio',
        'train_util': 'u60',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260226_213417_u60_proxy_cnot_ratio_iter200_singleutil_cnot_measurediff_20260226_152810/best/best_program.py',
    },
    'cnot_ratio_u88': {
        'proxy': 'cnot_ratio',
        'train_util': 'u88',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260226_213418_u88_proxy_cnot_ratio_iter200_singleutil_cnot_measurediff_20260226_152810/best/best_program.py',
    },
    'depth_ratio_avg306088_qwen100': {
        'proxy': 'depth_ratio',
        'train_util': 'avg306088_qwen100',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter500_20260213_130246_u306088_s1000_top10_avgrank_iter500/checkpoints/checkpoint_100/best_program.py',
    },
    'depth_ratio_avg306088_gpt53codex': {
        'proxy': 'depth_ratio',
        'train_util': 'avg306088_gpt53codex',
        'target': ROOT / 'openevolve_output/OpenAI_gpt-5.3-codex_iter100_20260303_005505_proxy_depthratio_avg306088_gpt53codex_iter100_20260303_login/best/best_program.py',
    },
    'fidelity_phys_avg306088': {
        'proxy': 'fidelity_phys',
        'train_util': 'avg306088_physfid',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260301_161639_avg306088_fidelity_physical8196_iter200_20260301_161540/best/best_program.py',
    },
    'fidelity_phys_u30': {
        'proxy': 'fidelity_phys',
        'train_util': 'u30_physfid',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260302_124223_u30_fidelity_physical8196_iter200_20260302_124207/best/best_program.py',
    },
    'fidelity_phys_u60': {
        'proxy': 'fidelity_phys',
        'train_util': 'u60_physfid',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260302_124228_u60_fidelity_physical8196_iter200_20260302_124207/best/best_program.py',
    },
    'fidelity_phys_u88': {
        'proxy': 'fidelity_phys',
        'train_util': 'u88_physfid',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260302_124313_u88_fidelity_physical8196_iter200_20260302_124207/best/best_program.py',
    },
    'fidelity_sim_avg306088_iter200': {
        'proxy': 'fidelity_sim',
        'train_util': 'avg306088_sim200',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter500_20260213_130246_u306088_s1000_top10_avgrank_iter500/checkpoints/checkpoint_200/best_program.py',
    },
    'fidelity_sim_avg306088_ckpt500': {
        'proxy': 'fidelity_sim',
        'train_util': 'avg306088_sim500',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter500_20260213_130246_u306088_s1000_top10_avgrank_iter500/checkpoints/checkpoint_500/best_program.py',
    },
    'number_instructions_ratio_avg306088_negraw': {
        'proxy': 'number_instructions_ratio',
        'train_util': 'avg306088_negraw',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260303_173618_avg306088_proxy_number_instructions_ratio_iter200_negraw_20260303_173550/best/best_program.py',
    },
    'num_measurements_avg_avg306088_negraw': {
        'proxy': 'num_measurements_avg',
        'train_util': 'avg306088_negraw',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260303_173618_avg306088_proxy_num_measurements_avg_iter200_negraw_20260303_173550/best/best_program.py',
    },
    'number_instructions_max_avg306088_negraw': {
        'proxy': 'number_instructions_max',
        'train_util': 'avg306088_negraw',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260303_173717_avg306088_proxy_number_instructions_max_iter200_negraw_20260303_173550/best/best_program.py',
    },
    'num_nonlocal_gates_max_avg306088_negraw': {
        'proxy': 'num_nonlocal_gates_max',
        'train_util': 'avg306088_negraw',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260303_173758_avg306088_proxy_num_nonlocal_gates_max_iter200_negraw_20260303_173550/best/best_program.py',
    },
    'number_instructions_avg_avg306088_negraw': {
        'proxy': 'number_instructions_avg',
        'train_util': 'avg306088_negraw',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260303_173758_avg306088_proxy_number_instructions_avg_iter200_negraw_20260303_173550/best/best_program.py',
    },
    'num_nonlocal_gates_avg_avg306088_negraw': {
        'proxy': 'num_nonlocal_gates_avg',
        'train_util': 'avg306088_negraw',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260303_173758_avg306088_proxy_num_nonlocal_gates_avg_iter200_negraw_20260303_173550/best/best_program.py',
    },
    'depth_max_avg306088_negraw': {
        'proxy': 'depth_max',
        'train_util': 'avg306088_negraw',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260303_173758_avg306088_proxy_depth_max_iter200_negraw_20260303_173550/best/best_program.py',
    },
    'number_instructions_diff_avg306088_negraw': {
        'proxy': 'number_instructions_diff',
        'train_util': 'avg306088_negraw',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260303_173933_avg306088_proxy_number_instructions_diff_iter200_negraw_20260303_173550/best/best_program.py',
    },
    'num_nonlocal_gates_diff_avg306088_negraw': {
        'proxy': 'num_nonlocal_gates_diff',
        'train_util': 'avg306088_negraw',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260303_174157_avg306088_proxy_num_nonlocal_gates_diff_iter200_negraw_20260303_173550/best/best_program.py',
    },
    'num_measurements_diff_avg306088_negraw': {
        'proxy': 'num_measurements_diff',
        'train_util': 'avg306088_negraw',
        'target': ROOT / 'openevolve_output/Qwen2.5-14B-Instruct_iter200_20260303_183830_avg306088_proxy_num_measurements_diff_iter200_negraw_20260303_173550/best/best_program.py',
    },
}

EVAL_UTILS = [30, 60, 88]
TOP_RATIOS = [0.10, 0.20]


def _reset_eval():
    evaluator._INIT_DONE = False
    evaluator._BENCHMARKS = None
    evaluator._CANDIDATES = None
    evaluator._FEATURES = None
    evaluator._QERNEL_PAIRS = None
    evaluator._MP = None
    evaluator._PAIR_METRICS = {}
    evaluator._PAIR_RANKS = None
    evaluator._PAIR_PROXY = None
    evaluator._SIM = None


def _load_rows(util: int):
    path = PAIR / f'pair_metrics_util{util}_shots8196.csv'
    rows = []
    with path.open('r', encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            rows.append({
                'name_1': r['name_1'],
                'name_2': r['name_2'],
                'effective_utilization': float(r['effective_utilization']),
                'fidelity': float(r['fidelity']),
            })
    return rows


def _scores_for_target(score_fn):
    vals = []
    for idx in range(len(evaluator._CANDIDATES)):
        q1, q2 = evaluator._QERNEL_PAIRS[idx]
        try:
            s = float(score_fn(evaluator._MP, q1, q2, evaluator._SIM.backend, weighted=False, weights=[]))
        except Exception:
            s = -1e9
        vals.append(s)
    return np.asarray(vals, dtype=float)


def _select_name_pairs(scores: np.ndarray, name_map: dict[tuple[str, str], int], top_ratio: float):
    k = max(1, int(math.ceil(len(scores) * top_ratio)))
    top_idx = set(np.argsort(scores)[-k:].tolist())
    names = set()
    for (n1, n2), i in name_map.items():
        if i in top_idx:
            names.add((n1, n2))
    return names


def _mean_pair(rows, selected):
    if not selected:
        return float('nan'), float('nan')
    f = []
    u = []
    for r in rows:
        if (r['name_1'], r['name_2']) in selected:
            f.append(r['fidelity'])
            u.append(r['effective_utilization'])
    if not f:
        return float('nan'), float('nan')
    return float(np.mean(f)), float(np.mean(u))


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    for mid, meta in METHODS.items():
        if not meta['target'].is_file():
            raise FileNotFoundError(f'Missing target for {mid}: {meta["target"]}')

    # per-method case stats over (util, top)
    method_case = {mid: [] for mid in METHODS}
    baseline_case = []

    for util in EVAL_UTILS:
        rows = _load_rows(util)

        config.TARGET_UTIL = util
        config.SHOTS = 1000
        config.CANDIDATE_LIMIT = None
        config.EVAL_RESTRICT_TO_CSV = False
        evaluator.config.TARGET_UTIL = config.TARGET_UTIL
        evaluator.config.SHOTS = config.SHOTS
        evaluator.config.CANDIDATE_LIMIT = config.CANDIDATE_LIMIT
        evaluator.config.EVAL_RESTRICT_TO_CSV = config.EVAL_RESTRICT_TO_CSV
        _reset_eval()
        evaluator._init()

        name_map = {}
        for i, (_c1, _c2, n1, n2) in enumerate(evaluator._CANDIDATES):
            name_map[(n1, n2)] = i

        # orig scores once
        _, score_fn_orig = evaluator._load_score_fn(str(TARGET_ORIG))
        scores_orig = _scores_for_target(score_fn_orig)

        method_scores = {}
        for mid, meta in METHODS.items():
            _, sf = evaluator._load_score_fn(str(meta['target']))
            method_scores[mid] = _scores_for_target(sf)

        for tr in TOP_RATIOS:
            sel_orig = _select_name_pairs(scores_orig, name_map, tr)
            orig_f, orig_u = _mean_pair(rows, sel_orig)
            baseline_case.append((orig_f, orig_u))

            for mid in METHODS:
                sel_new = _select_name_pairs(method_scores[mid], name_map, tr)
                new_f, new_u = _mean_pair(rows, sel_new)
                method_case[mid].append((new_f, new_u, orig_f, orig_u))

    out_rows = []
    for mid, meta in METHODS.items():
        vals = method_case[mid]
        new_f = float(np.mean([x[0] for x in vals]))
        new_u = float(np.mean([x[1] for x in vals]))
        orig_f = float(np.mean([x[2] for x in vals]))
        orig_u = float(np.mean([x[3] for x in vals]))
        out_rows.append({
            'method_id': mid,
            'proxy': meta['proxy'],
            'train_util': meta['train_util'],
            'overall_new_mean_fid': new_f,
            'overall_new_mean_effutil': new_u,
            'overall_orig_mean_fid': orig_f,
            'overall_orig_mean_effutil': orig_u,
        })

    out_rows.sort(key=lambda r: r['method_id'])
    out_csv = OUT / 'proxy_trainutil_overall_points.csv'
    with out_csv.open('w', encoding='utf-8', newline='') as f:
        wr = csv.DictWriter(
            f,
            fieldnames=[
                'method_id', 'proxy', 'train_util',
                'overall_new_mean_fid', 'overall_new_mean_effutil',
                'overall_orig_mean_fid', 'overall_orig_mean_effutil',
            ],
        )
        wr.writeheader()
        wr.writerows(out_rows)

    # Scatter plot
    color_map = {
        'depth_ratio': '#1f77b4',
        'instr_sum': '#ff7f0e',
        'measure_diff': '#2ca02c',
        'cnot_ratio': '#d62728',
        'fidelity_phys': '#9467bd',
        'fidelity_sim': '#8c564b',
        'number_instructions_ratio': '#17becf',
        'num_measurements_avg': '#bcbd22',
        'number_instructions_max': '#e377c2',
        'num_nonlocal_gates_max': '#7f7f7f',
        'number_instructions_avg': '#aec7e8',
        'num_nonlocal_gates_avg': '#ffbb78',
        'depth_max': '#98df8a',
        'number_instructions_diff': '#c5b0d5',
        'num_nonlocal_gates_diff': '#c49c94',
        'num_measurements_diff': '#f7b6d2',
    }
    marker_map = {
        'avg306088': 'o',
        'u30': '^',
        'u60': 's',
        'u88': 'D',
        'avg306088_qwen100': 'P',
        'avg306088_gpt53codex': 'X',
        'avg306088_physfid': 'h',
        'u30_physfid': '^',
        'u60_physfid': 's',
        'u88_physfid': 'D',
        'avg306088_sim200': 'v',
        'avg306088_sim500': '<',
        'avg306088_negraw': 'X',
    }

    fig, ax = plt.subplots(figsize=(13, 8.5), dpi=220)

    # Baseline point
    base_f = float(np.mean([x[0] for x in baseline_case]))
    base_u = float(np.mean([x[1] for x in baseline_case]))
    ax.scatter(base_u, base_f, s=260, marker='*', c='black', label='orig baseline', zorder=5)

    all_x = [r['overall_new_mean_effutil'] for r in out_rows] + [base_u]
    all_y = [r['overall_new_mean_fid'] for r in out_rows] + [base_f]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_span = max(1e-9, x_max - x_min)
    y_span = max(1e-9, y_max - y_min)

    placed_labels = []
    for r in out_rows:
        x = r['overall_new_mean_effutil']
        y = r['overall_new_mean_fid']
        c = color_map.get(r['proxy'], '#777777')
        m = marker_map.get(r['train_util'], 'o')
        ax.scatter(x, y, s=145, c=c, marker=m, edgecolors='black', linewidths=0.35, alpha=0.95, zorder=4)

        # Greedy label placement to reduce overlap in dense clusters.
        tx = x + 0.010 * x_span
        ty = y + 0.010 * y_span
        tries = 0
        while any(abs(tx - px) < 0.020 * x_span and abs(ty - py) < 0.020 * y_span for px, py in placed_labels):
            ty += 0.012 * y_span
            tries += 1
            if ty > y_max + 0.01 * y_span:
                ty = y - 0.012 * y_span
            if tries % 5 == 0:
                tx += 0.010 * x_span
            if tries > 30:
                break
        placed_labels.append((tx, ty))
        ax.annotate(
            r['method_id'],
            xy=(x, y),
            xytext=(tx, ty),
            textcoords='data',
            fontsize=7.5,
            alpha=0.92,
            arrowprops=dict(arrowstyle='-', lw=0.55, color='#666666', alpha=0.7),
            bbox=dict(boxstyle='round,pad=0.14', fc='white', ec='none', alpha=0.72),
            zorder=6,
        )

    ax.set_xlabel('Overall Mean Effective Utilization (avg over util30/60/88 and Top10/Top20)')
    ax.set_ylabel('Overall Mean Fidelity (physical shots=8196)')
    ax.set_title('Proxy Multimethod Train-Util Scatter (Physical 8196, avg eval util30/60/88)')
    ax.grid(alpha=0.25)

    # legends
    proxy_handles = []
    for p, c in color_map.items():
        proxy_handles.append(plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=c, markeredgecolor='white', markersize=9, label=p))
    util_handles = []
    for t, m in marker_map.items():
        util_handles.append(plt.Line2D([0], [0], marker=m, color='#666666', linestyle='None', markersize=9, label=t))

    lg1 = ax.legend(handles=proxy_handles, title='Proxy', loc='lower right', fontsize=8)
    ax.add_artist(lg1)
    ax.legend(handles=util_handles, title='Train util', loc='upper left', fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / 'proxy_trainutil_overall_expanded_scatter_avg306088eval.png', dpi=280)
    fig.savefig(OUT / 'proxy_trainutil_overall_expanded_scatter_avg306088eval.pdf')
    plt.close(fig)

    print('[DONE] overwritten proxy_trainutil_overall_expanded_scatter_avg306088eval with pair_metrics util30/60/88 shots8196')
    print(f'[BASE] fid={base_f:.6f}, effutil={base_u:.6f}')


if __name__ == '__main__':
    main()
