#!/usr/bin/env python3
import csv
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing')
OUT = ROOT / 'openevolve_output' / '_proxy3_vs_fid_realfid_8196_20260302'
PLOT = ROOT / 'plot_qos_vs_evolved_from_csv.py'
PYTHON = Path('/work/nvme/becn/lily/miniconda3/envs/qos_fig11/bin/python')
TARGET_ORIG = ROOT / 'target_orig.py'

METHOD_TO_TARGET = {
    'depth_ratio': ROOT / 'openevolve_output' / 'Qwen2.5-14B-Instruct_iter200_20260224_014256_avg306088_proxy_depth_ratio_iter200_20260224_014243' / 'best' / 'best_program.py',
    'instr_sum': ROOT / 'openevolve_output' / 'Qwen2.5-14B-Instruct_iter200_20260224_014800_avg306088_proxy_instr_sum_iter200_20260224_014450' / 'best' / 'best_program.py',
    'measure_diff': ROOT / 'openevolve_output' / 'Qwen2.5-14B-Instruct_iter200_20260226_035035_avg306088_proxy_measure_diff_iter200_20260226_010631' / 'best' / 'best_program.py',
    'fid200': ROOT / 'openevolve_output' / 'Qwen2.5-14B-Instruct_iter200_20260301_161639_avg306088_fidelity_physical8196_iter200_20260301_161540' / 'best' / 'best_program.py',
    'fid500': ROOT / 'openevolve_output' / 'Qwen2.5-14B-Instruct_iter500_20260213_130246_u306088_s1000_top10_avgrank_iter500' / 'checkpoints' / 'checkpoint_500' / 'best_program.py',
}
EVAL_UTILS = [30, 60, 88]
TOPS = [10, 20]


def clean_csv(src: Path, dst: Path) -> None:
    with open(src, 'r', encoding='utf-8', newline='') as fin:
        rd = csv.DictReader(fin)
        if not rd.fieldnames:
            raise RuntimeError(f'No header in {src}')
        fields = [h for h in rd.fieldnames if h is not None]
        with open(dst, 'w', encoding='utf-8', newline='') as fout:
            wr = csv.DictWriter(fout, fieldnames=fields)
            wr.writeheader()
            for row in rd:
                row.pop(None, None)
                wr.writerow({k: row.get(k, '') for k in fields})


def mean(xs):
    xs = [x for x in xs if math.isfinite(x)]
    if not xs:
        return float('nan')
    return float(sum(xs) / len(xs))


def run_one(method: str, target_py: Path, util: int, top: int, csv_in: Path) -> tuple[int, int, float, float]:
    stem = f'{method}_u{util}_t{top}'
    out_png = OUT / f'{stem}.png'
    out_csv = OUT / f'{stem}.csv'
    bar_png = OUT / f'{stem}_utilbin_mean_fid.png'
    bar_csv = OUT / f'{stem}_bar.csv'
    summary_csv = OUT / f'{stem}_summary.csv'

    cmd = [
        str(PYTHON), str(PLOT),
        '--csv', str(csv_in),
        '--util', str(util),
        '--shots', '8196',
        '--top-k-ratio', str(top),
        '--orig-target', str(TARGET_ORIG),
        '--new-target', str(target_py),
        '--out', str(out_png),
        '--out-csv', str(out_csv),
        '--bar-out', str(bar_png),
        '--bar-out-csv', str(bar_csv),
        '--summary-csv', str(summary_csv),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    orig_vals = []
    new_vals = []
    with open(out_csv, 'r', encoding='utf-8', newline='') as f:
        rd = csv.DictReader(f)
        for r in rd:
            try:
                fid = float(r['fidelity'])
            except Exception:
                continue
            if str(r.get('topk_orig', '')).strip().lower() in {'true', '1', 'yes'}:
                orig_vals.append(fid)
            if str(r.get('topk_new', '')).strip().lower() in {'true', '1', 'yes'}:
                new_vals.append(fid)

    return len(orig_vals), len(new_vals), mean(orig_vals), mean(new_vals)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    clean_inputs = {}
    for u in EVAL_UTILS:
        src = ROOT / 'pairing_metadata' / f'pair_metrics_util{u}_shots8196.csv'
        dst = OUT / f'_clean_pair_metrics_util{u}_shots8196.csv'
        clean_csv(src, dst)
        clean_inputs[u] = dst

    detail = []
    for method, target in METHOD_TO_TARGET.items():
        if not target.is_file():
            raise FileNotFoundError(f'Missing target: {target}')
        for u in EVAL_UTILS:
            for t in TOPS:
                print(f'[RUN] {method} u{u} t{t}', flush=True)
                n_orig, n_new, orig_mean, new_mean = run_one(method, target, u, t, clean_inputs[u])
                detail.append({
                    'method': method,
                    'util': u,
                    'top': t,
                    'n_orig': n_orig,
                    'n_new': n_new,
                    'orig_mean': orig_mean,
                    'new_mean': new_mean,
                    'delta': new_mean - orig_mean,
                })

    detail_path = OUT / 'compare_proxy3_fid_realfid_detail.csv'
    with open(detail_path, 'w', encoding='utf-8', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['method', 'util', 'top', 'n_orig', 'n_new', 'orig_mean', 'new_mean', 'delta'])
        wr.writeheader()
        wr.writerows(detail)

    overall = []
    methods = list(METHOD_TO_TARGET.keys())
    for method in methods:
        rows = [r for r in detail if r['method'] == method]
        t10 = [r for r in rows if r['top'] == 10]
        t20 = [r for r in rows if r['top'] == 20]
        top10_new = mean([r['new_mean'] for r in t10])
        top20_new = mean([r['new_mean'] for r in t20])
        top10_orig = mean([r['orig_mean'] for r in t10])
        top20_orig = mean([r['orig_mean'] for r in t20])
        overall_new = mean([top10_new, top20_new])
        overall_orig = mean([top10_orig, top20_orig])
        overall.append({
            'method': method,
            'overall_new_mean': overall_new,
            'overall_delta_vs_orig': overall_new - overall_orig,
            'top10_new_mean': top10_new,
            'top20_new_mean': top20_new,
            'top10_delta': top10_new - top10_orig,
            'top20_delta': top20_new - top20_orig,
        })

    overall_sorted = sorted(overall, key=lambda r: r['overall_new_mean'], reverse=True)
    overall_path = OUT / 'compare_proxy3_fid_realfid_overall.csv'
    with open(overall_path, 'w', encoding='utf-8', newline='') as f:
        wr = csv.DictWriter(
            f,
            fieldnames=['method', 'overall_new_mean', 'overall_delta_vs_orig', 'top10_new_mean', 'top20_new_mean', 'top10_delta', 'top20_delta'],
        )
        wr.writeheader()
        wr.writerows(overall_sorted)

    # Summary figure
    x = np.arange(len(overall_sorted))
    labels = [r['method'] for r in overall_sorted]
    top10 = [r['top10_new_mean'] for r in overall_sorted]
    top20 = [r['top20_new_mean'] for r in overall_sorted]
    overall_vals = [r['overall_new_mean'] for r in overall_sorted]

    # Baseline from orig target on util30/60/88 (same for all methods)
    base_top10 = mean([r['orig_mean'] for r in detail if r['method'] == methods[0] and r['top'] == 10])
    base_top20 = mean([r['orig_mean'] for r in detail if r['method'] == methods[0] and r['top'] == 20])
    base_overall = mean([base_top10, base_top20])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=160)
    w = 0.38
    ax1.bar(x - w / 2, top10, width=w, color='#4c78a8', label='Top10%')
    ax1.bar(x + w / 2, top20, width=w, color='#f58518', label='Top20%')
    ax1.axhline(base_top10, color='#4c78a8', ls='--', lw=1.3, alpha=0.8, label=f'Baseline Top10% ({base_top10:.3f})')
    ax1.axhline(base_top20, color='#f58518', ls='--', lw=1.3, alpha=0.8, label=f'Baseline Top20% ({base_top20:.3f})')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha='right')
    ax1.set_ylabel('Physical Mean Fidelity')
    ax1.set_title('Physical Fidelity by Method')
    ax1.grid(axis='y', alpha=0.25)
    ax1.legend(loc='lower left', fontsize=9)

    colors = ['#54a24b' if not m.startswith('fid') else ('#b279a2' if m == 'fid200' else '#8f63c5') for m in labels]
    ax2.bar(x, overall_vals, color=colors)
    ax2.axhline(base_overall, color='gray', ls='--', lw=1.2, alpha=0.8, label=f'Baseline overall ({base_overall:.3f})')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha='right')
    ax2.set_ylabel('Mean(Top10%, Top20%)')
    ax2.set_title('Overall Score (Higher Better)')
    ax2.grid(axis='y', alpha=0.25)
    ax2.legend(loc='upper right', fontsize=9)
    for i, v in enumerate(overall_vals):
        ax2.text(i, v + 0.003, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Proxy3 vs Fidelity@ckpt200/ckpt500 on Physical Data (util30/60/88, shots=8196)', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / 'proxy3_vs_fid_realfid_8196_summary.png')
    fig.savefig(OUT / 'proxy3_vs_fid_realfid_8196_summary.pdf')
    plt.close(fig)

    print('[DONE] proxy3_vs_fid_realfid_8196 repaired with pair_metrics util30/60/88 shots8196')


if __name__ == '__main__':
    main()
