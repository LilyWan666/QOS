from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path('/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing')
DETAIL_CSV = ROOT / 'figures' / 'ablation_proxy_choice' / 'proxy_choice_detail.csv'
OUT_BASE = ROOT / 'figures' / 'ablation_proxy_choice'

METHOD_ORDER = [
    'depth_ratio',
    'instr_ratio',
    'cnot_ratio',
    'nonlocal_ratio',
    'measure_ratio',
    'simulation_fidelity',
]
METHOD_LABELS = {
    'depth_ratio': 'Depth Ratio',
    'instr_ratio': 'Instr. Ratio',
    'cnot_ratio': 'CNOT Ratio',
    'nonlocal_ratio': 'Nonlocal Ratio',
    'measure_ratio': 'Measure Ratio',
    'simulation_fidelity': 'Sim. Fidelity',
}
FID_COLOR = '#4C78A8'
UTIL_COLOR = '#59A14F'

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'font.size': 13,
    'axes.grid': False,
})


def load_rows() -> list[dict[str, str]]:
    with DETAIL_CSV.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def build_run_metrics(rows: list[dict[str, str]]):
    grouped: dict[tuple[str, str, int, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row['machine'], row['method'], int(row['run_index']), float(row['top_ratio']))].append(row)

    ratio_metrics: dict[tuple[str, str, int], dict[float, dict[str, float]]] = defaultdict(dict)
    for (machine, method, run_index, top_ratio), sub in grouped.items():
        mean_fid = float(np.mean([float(r['mean_fidelity']) for r in sub]))
        mean_util = float(np.mean([float(r['mean_normalized_effective_utilization']) for r in sub])) * 100.0
        ratio_metrics[(machine, method, run_index)][top_ratio] = {
            'mean_fidelity': mean_fid,
            'mean_norm_util_pct': mean_util,
        }

    per_run: dict[tuple[str, str], dict[int, dict[str, float]]] = defaultdict(dict)
    per_machine: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (machine, method, run_index), ratios in ratio_metrics.items():
        if 0.1 not in ratios or 0.2 not in ratios:
            continue
        avg_fid = (ratios[0.1]['mean_fidelity'] + ratios[0.2]['mean_fidelity']) / 2.0
        avg_util = (ratios[0.1]['mean_norm_util_pct'] + ratios[0.2]['mean_norm_util_pct']) / 2.0
        per_run[(machine, method)][run_index] = {'avg_fid': avg_fid, 'avg_util': avg_util}

    for machine in {k[0] for k in per_run.keys()}:
        for method in METHOD_ORDER:
            run_map = per_run[(machine, method)]
            avg_fids = [run_map[idx]['avg_fid'] for idx in sorted(run_map)]
            avg_utils = [run_map[idx]['avg_util'] for idx in sorted(run_map)]
            per_machine[machine][method] = {
                'avg_fid_mean': float(np.mean(avg_fids)),
                'avg_fid_std': float(np.std(avg_fids, ddof=1)) if len(avg_fids) > 1 else 0.0,
                'avg_util_mean': float(np.mean(avg_utils)),
                'avg_util_std': float(np.std(avg_utils, ddof=1)) if len(avg_utils) > 1 else 0.0,
                'n_runs': len(avg_fids),
            }
    return per_machine, per_run


def build_cross_machine(per_run):
    merged = {}
    for method in METHOD_ORDER:
        run_ids = sorted(set(per_run[('torino', method)].keys()) & set(per_run[('marrakesh', method)].keys()))
        avg_fids = []
        avg_utils = []
        for run_id in run_ids:
            avg_fids.append((per_run[('torino', method)][run_id]['avg_fid'] + per_run[('marrakesh', method)][run_id]['avg_fid']) / 2.0)
            avg_utils.append((per_run[('torino', method)][run_id]['avg_util'] + per_run[('marrakesh', method)][run_id]['avg_util']) / 2.0)
        merged[method] = {
            'avg_fid_mean': float(np.mean(avg_fids)),
            'avg_fid_std': float(np.std(avg_fids, ddof=1)) if len(avg_fids) > 1 else 0.0,
            'avg_util_mean': float(np.mean(avg_utils)),
            'avg_util_std': float(np.std(avg_utils, ddof=1)) if len(avg_utils) > 1 else 0.0,
            'n_runs': len(avg_fids),
        }
    return merged


def save_summary(out_dir: Path, title_key: str, metrics: dict[str, dict[str, float]]) -> None:
    out_csv = out_dir / 'summary.csv'
    with out_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['group', 'method', 'label', 'n_runs', 'avg_fid_mean', 'avg_fid_std', 'avg_util_mean', 'avg_util_std'],
        )
        writer.writeheader()
        for method in METHOD_ORDER:
            row = metrics[method]
            writer.writerow({
                'group': title_key,
                'method': method,
                'label': METHOD_LABELS[method],
                'n_runs': row['n_runs'],
                'avg_fid_mean': row['avg_fid_mean'],
                'avg_fid_std': row['avg_fid_std'],
                'avg_util_mean': row['avg_util_mean'],
                'avg_util_std': row['avg_util_std'],
            })


def plot_group(out_dir: Path, title_key: str, metrics: dict[str, dict[str, float]]) -> None:
    labels = [METHOD_LABELS[m] for m in METHOD_ORDER]
    x = np.arange(len(labels))
    width = 0.36
    fid_means = [metrics[m]['avg_fid_mean'] for m in METHOD_ORDER]
    fid_stds = [metrics[m]['avg_fid_std'] for m in METHOD_ORDER]
    util_means = [metrics[m]['avg_util_mean'] for m in METHOD_ORDER]
    util_stds = [metrics[m]['avg_util_std'] for m in METHOD_ORDER]

    fig, ax1 = plt.subplots(figsize=(10.6, 3.9))
    ax2 = ax1.twinx()

    ax1.bar(x - width/2, fid_means, width, yerr=fid_stds, color=FID_COLOR, edgecolor='#333333', linewidth=1.0, capsize=4)
    ax2.bar(x + width/2, util_means, width, yerr=util_stds, color=UTIL_COLOR, edgecolor='#333333', linewidth=1.0, capsize=4)

    ax1.set_xlabel('Second Metric Used During Evolution')
    ax1.set_ylabel('Avg Fidelity')
    ax2.set_ylabel('Avg Normalized Eff. Util. (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=12, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.35)

    fid_low = min(m - s for m, s in zip(fid_means, fid_stds))
    fid_high = max(m + s for m, s in zip(fid_means, fid_stds))
    util_low = min(m - s for m, s in zip(util_means, util_stds))
    util_high = max(m + s for m, s in zip(util_means, util_stds))
    fid_pad = max(0.01, (fid_high - fid_low) * 0.18)
    util_pad = max(0.3, (util_high - util_low) * 0.18)
    ax1.set_ylim(fid_low - fid_pad, fid_high + fid_pad)
    ax2.set_ylim(util_low - util_pad, util_high + util_pad)

    legend_handles = [
        Patch(facecolor=FID_COLOR, edgecolor='#333333', label='Avg Fidelity'),
        Patch(facecolor=UTIL_COLOR, edgecolor='#333333', label='Avg Norm. Eff. Util.'),
    ]
    ax1.legend(handles=legend_handles, loc='upper right', frameon=True)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out_dir / f'figure_proxy_choice_ablation_dual_axis.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    save_summary(out_dir, title_key, metrics)




def plot_scatter_group(out_dir: Path, title_key: str, metrics: dict[str, dict[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    colors = {
        'depth_ratio': '#4C78A8',
        'instr_ratio': '#F58518',
        'cnot_ratio': '#54A24B',
        'nonlocal_ratio': '#E45756',
        'measure_ratio': '#B279A2',
        'simulation_fidelity': '#72B7B2',
    }
    handles = []
    for method in METHOD_ORDER:
        row = metrics[method]
        x = row['avg_util_mean']
        y = row['avg_fid_mean']
        xerr = row['avg_util_std']
        yerr = row['avg_fid_std']
        c = colors[method]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', ms=10, mec='#333333', mew=1.0,
                    color=c, ecolor=c, elinewidth=1.2, capsize=3, alpha=0.95)
        handles.append(Patch(facecolor=c, edgecolor='#333333', label=METHOD_LABELS[method]))

    ax.set_xlabel('Avg Normalized Eff. Util. (%)')
    ax.set_ylabel('Avg Fidelity')
    ax.grid(True, linestyle='--', alpha=0.35)

    xs = [metrics[m]['avg_util_mean'] for m in METHOD_ORDER]
    ys = [metrics[m]['avg_fid_mean'] for m in METHOD_ORDER]
    xerrs = [metrics[m]['avg_util_std'] for m in METHOD_ORDER]
    yerrs = [metrics[m]['avg_fid_std'] for m in METHOD_ORDER]
    xmin = min(x - e for x, e in zip(xs, xerrs))
    xmax = max(x + e for x, e in zip(xs, xerrs))
    ymin = min(y - e for y, e in zip(ys, yerrs))
    ymax = max(y + e for y, e in zip(ys, yerrs))
    xpad = max(0.15, (xmax - xmin) * 0.18)
    ypad = max(0.01, (ymax - ymin) * 0.18)
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)

    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out_dir / f'figure_proxy_choice_scatter.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)

def main():
    rows = load_rows()
    per_machine, per_run = build_run_metrics(rows)
    run_tag = datetime.now().strftime('run_%Y%m%d_%H%M%S')

    tor_dir = OUT_BASE / f'torino_{run_tag}'
    mar_dir = OUT_BASE / f'marrakesh_{run_tag}'
    avg_dir = OUT_BASE / f'avg_{run_tag}'
    tor_dir.mkdir(parents=True, exist_ok=True)
    mar_dir.mkdir(parents=True, exist_ok=True)
    avg_dir.mkdir(parents=True, exist_ok=True)

    avg_metrics = build_cross_machine(per_run)
    plot_group(tor_dir, 'torino', per_machine['torino'])
    plot_group(mar_dir, 'marrakesh', per_machine['marrakesh'])
    plot_group(avg_dir, 'torino_marrakesh_avg', avg_metrics)
    plot_scatter_group(tor_dir, 'torino', per_machine['torino'])
    plot_scatter_group(mar_dir, 'marrakesh', per_machine['marrakesh'])
    plot_scatter_group(avg_dir, 'torino_marrakesh_avg', avg_metrics)
    print(tor_dir)
    print(mar_dir)
    print(avg_dir)

if __name__ == '__main__':
    main()
