#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.openevolve_pairing import config, evaluator  # noqa: E402
from evaluation.openevolve_pairing.build_physical_pair_csv import build_rows  # noqa: E402

ROOT = Path('/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing')
OPENEVOLVE_OUT = ROOT / 'openevolve_output'
IBM_JOBS = Path('/work/nvme/betu/lily/QOS/ibm_quantum/jobs')
FIG_BASE = ROOT / 'figures' / 'ablation_util_aggregation_physical'

EVAL_UTILS = [30, 60, 88]
PHYSICAL_TOP_RATIOS = [0.10, 0.20]
TARGET_QUBITS = {30: 8, 60: 16, 88: 24}
BACKEND_QUBITS = {'torino': 133, 'marrakesh': 156}

RUN_GROUPS: Dict[str, List[Path]] = {
    'mean(30,60,88)': [
        OPENEVOLVE_OUT / 'GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260310_143101_flash_top01_combined50_50_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100',
        OPENEVOLVE_OUT / 'GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260310_221737_flash_top01_combined50_50_r2_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100',
        OPENEVOLVE_OUT / 'GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_004743_flash_top01_combined50_50_r3_proxy_depthratio_avgutil_306088/run01_gemini-ge0_iter100',
    ],
    'util30': [
        OPENEVOLVE_OUT / 'GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260310_194318_flash_u30_top01_combined50_50_proxy_depthratio_1000shots/run01_gemini-ge0_iter100',
        OPENEVOLVE_OUT / 'GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_135849_flash_u30_top01_combined50_50_r2_proxy_depthratio_1000shots/run01_gemini-ge0_iter100',
        OPENEVOLVE_OUT / 'GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_135849_flash_u30_top01_combined50_50_r3_proxy_depthratio_1000shots/run01_gemini-ge0_iter100',
    ],
    'util60': [
        OPENEVOLVE_OUT / 'GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260310_194323_flash_u60_top01_combined50_50_proxy_depthratio_1000shots/run01_gemini-ge0_iter100',
        OPENEVOLVE_OUT / 'GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_135849_flash_u60_top01_combined50_50_r2_proxy_depthratio_1000shots/run01_gemini-ge0_iter100',
        OPENEVOLVE_OUT / 'GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_135855_flash_u60_top01_combined50_50_r3_proxy_depthratio_1000shots/run01_gemini-ge0_iter100',
    ],
    'util88': [
        OPENEVOLVE_OUT / 'GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260310_194318_flash_u88_top01_combined50_50_proxy_depthratio_1000shots/run01_gemini-ge0_iter100',
        OPENEVOLVE_OUT / 'GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_135857_flash_u88_top01_combined50_50_r2_proxy_depthratio_1000shots/run01_gemini-ge0_iter100',
        OPENEVOLVE_OUT / 'GeminiMulti_gemini-3-flash-preview_keys1_iter100_20260311_135854_flash_u88_top01_combined50_50_r3_proxy_depthratio_1000shots/run01_gemini-ge0_iter100',
    ],
}
DISPLAY_LABELS = {
    'mean(30,60,88)': 'Mean',
    'util30': 'Util 30%',
    'util60': 'Util 60%',
    'util88': 'Util 88%',
}


def reset_eval() -> None:
    evaluator._INIT_DONE = False
    evaluator._BENCHMARKS = None
    evaluator._CANDIDATES = None
    evaluator._FEATURES = None
    evaluator._QERNEL_PAIRS = None
    evaluator._MP = None
    evaluator._PAIR_METRICS = {}
    evaluator._PAIR_METADATA = {}
    evaluator._PAIR_METADATA_COLUMNS = []
    evaluator._PAIR_RANKS = None
    evaluator._PAIR_PROXY = None
    evaluator._SIM = None


def init_for_util(util: int) -> None:
    config.TARGET_UTIL = util
    config.SHOTS = 1000
    config.CANDIDATE_LIMIT = None
    config.EVAL_RESTRICT_TO_CSV = False
    evaluator.config.TARGET_UTIL = util
    evaluator.config.SHOTS = 1000
    evaluator.config.CANDIDATE_LIMIT = None
    evaluator.config.EVAL_RESTRICT_TO_CSV = False
    reset_eval()
    evaluator._init()


def scores_for_target(target_path: Path) -> np.ndarray:
    _, score_fn = evaluator._load_score_fn(str(target_path))
    scores = []
    for idx in range(len(evaluator._CANDIDATES)):
        q1, q2 = evaluator._QERNEL_PAIRS[idx]
        try:
            score = float(score_fn(evaluator._MP, q1, q2, evaluator._SIM.backend, weighted=False, weights=[]))
        except Exception:
            score = -1e9
        scores.append(score)
    return np.asarray(scores, dtype=float)


def select_name_pairs(scores: np.ndarray, name_map: Dict[tuple[str, str], int], top_ratio: float) -> set[tuple[str, str]]:
    k = max(1, int(math.ceil(len(scores) * top_ratio)))
    top_idx = set(np.argsort(scores)[-k:].tolist())
    return {(n1, n2) for (n1, n2), idx in name_map.items() if idx in top_idx}


def physical_map_for(machine: str, util: int) -> Dict[tuple[str, str], tuple[float, float]]:
    fidelity_dir = IBM_JOBS / f'util{util}_ibm_{machine}_shots8192' / 'fidelity'
    rows = build_rows(str(fidelity_dir))
    return {
        (str(r['name_1']), str(r['name_2'])): (float(r['effective_utilization']), float(r['fidelity']))
        for r in rows
    }


def mean_on_physical(selected_pairs: set[tuple[str, str]], physical_map: Dict[tuple[str, str], tuple[float, float]]) -> tuple[float, float]:
    effs, fids = [], []
    for key in selected_pairs:
        if key in physical_map:
            eff, fid = physical_map[key]
            effs.append(eff)
            fids.append(fid)
    return float(np.mean(effs)), float(np.mean(fids))


def compute_run_metrics(run_dir: Path, machine: str, eval_utils: List[int]) -> Dict[float, Dict[str, float]]:
    target_path = run_dir / 'best' / 'best_program.py'
    backend_qubits = BACKEND_QUBITS[machine]
    out: Dict[float, Dict[str, float]] = {}
    for phys_top_ratio in PHYSICAL_TOP_RATIOS:
        rels, effs, fids = [], [], []
        for util in eval_utils:
            init_for_util(util)
            name_map = {(n1, n2): idx for idx, (_c1, _c2, n1, n2) in enumerate(evaluator._CANDIDATES)}
            scores = scores_for_target(target_path)
            selected = select_name_pairs(scores, name_map, phys_top_ratio)
            eff, fid = mean_on_physical(selected, physical_map_for(machine, util))
            rel = eff / (TARGET_QUBITS[util] / backend_qubits)
            rels.append(rel)
            effs.append(eff)
            fids.append(fid)
        out[phys_top_ratio] = {
            'mean_normalized_eff_util': float(np.mean(rels)),
            'mean_effective_utilization': float(np.mean(effs)),
            'mean_fidelity': float(np.mean(fids)),
        }
    return out


def aggregate_group_metrics(run_dirs: List[Path], machine: str, eval_utils: List[int]) -> Dict[str, float]:
    by_ratio = {ratio: {'mean_normalized_eff_util': [], 'mean_effective_utilization': [], 'mean_fidelity': []} for ratio in PHYSICAL_TOP_RATIOS}
    for run_dir in run_dirs:
        run_metrics = compute_run_metrics(run_dir, machine, eval_utils)
        for ratio in PHYSICAL_TOP_RATIOS:
            for key, value in run_metrics[ratio].items():
                by_ratio[ratio][key].append(value)
    out = {}
    for ratio in PHYSICAL_TOP_RATIOS:
        prefix = 'top10' if abs(ratio - 0.10) < 1e-12 else 'top20'
        out[f'{prefix}_fidelity_mean'] = float(np.mean(by_ratio[ratio]['mean_fidelity']))
        out[f'{prefix}_fidelity_var'] = float(np.var(by_ratio[ratio]['mean_fidelity']))
        out[f'{prefix}_normalized_eff_util_mean'] = float(np.mean(by_ratio[ratio]['mean_normalized_eff_util']))
        out[f'{prefix}_normalized_eff_util_var'] = float(np.var(by_ratio[ratio]['mean_normalized_eff_util']))
    avg_fids, avg_norm_utils = [], []
    for i in range(len(run_dirs)):
        avg_fids.append(float(np.mean([by_ratio[0.10]['mean_fidelity'][i], by_ratio[0.20]['mean_fidelity'][i]])))
        avg_norm_utils.append(float(np.mean([by_ratio[0.10]['mean_normalized_eff_util'][i], by_ratio[0.20]['mean_normalized_eff_util'][i]])))
    out['avg_fidelity_mean'] = float(np.mean(avg_fids))
    out['avg_fidelity_var'] = float(np.var(avg_fids))
    out['avg_normalized_eff_util_mean'] = float(np.mean(avg_norm_utils))
    out['avg_normalized_eff_util_var'] = float(np.var(avg_norm_utils))
    return out


def write_local_plot_script(out_dir: Path, stem: str) -> None:
    script = f'''from pathlib import Path\nimport csv\nimport math\nimport matplotlib.pyplot as plt\n\nSCRIPT_DIR = Path(__file__).resolve().parent\nSTEM = "{stem}"\nCSV_PATH = SCRIPT_DIR / f"{{STEM}}.csv"\nPNG_PATH = SCRIPT_DIR / f"{{STEM}}.png"\nPDF_PATH = SCRIPT_DIR / f"{{STEM}}.pdf"\n\nrows = []\nwith CSV_PATH.open(newline='') as f:\n    reader = csv.DictReader(f)\n    for row in reader:\n        rows.append(row)\n\nlabels = [r['display'] for r in rows]\nfid_mean = [float(r['avg_fidelity_mean']) for r in rows]\nfid_std = [math.sqrt(float(r['avg_fidelity_var'])) for r in rows]\nutil_mean = [100.0 * float(r['avg_normalized_eff_util_mean']) for r in rows]\nutil_std = [100.0 * math.sqrt(float(r['avg_normalized_eff_util_var'])) for r in rows]\n\nx = list(range(len(labels)))\nwidth = 0.34\nfig, ax1 = plt.subplots(figsize=(7.2, 2.7), dpi=220)\nax2 = ax1.twinx()\nbar1 = ax1.bar([v - width/2 for v in x], fid_mean, width=width, yerr=fid_std, capsize=3, color='#4C78A8', alpha=0.92, label='Avg Fidelity')\nbar2 = ax2.bar([v + width/2 for v in x], util_mean, width=width, yerr=util_std, capsize=3, color='#F28E2B', alpha=0.85, label='Avg Norm. Eff. Util. (%)')\nax1.set_xticks(x)\nax1.set_xticklabels(labels, fontsize=11)\nax1.set_ylabel('Avg Fidelity', fontsize=11)\nax2.set_ylabel('Avg Norm. Eff. Util. (%)', fontsize=11)\nax1.tick_params(axis='y', labelsize=10)\nax2.tick_params(axis='y', labelsize=10)\nax1.grid(True, axis='y', linestyle='--', linewidth=0.55, alpha=0.28)\nax1.legend([bar1, bar2], ['Avg Fidelity', 'Avg Norm. Eff. Util. (%)'], loc='upper center', bbox_to_anchor=(0.5, 0.985), ncol=2, fontsize=10, frameon=True)\nax1.spines['top'].set_visible(False)\nax2.spines['top'].set_visible(False)\nax1.set_ylim(top=max(v + s for v, s in zip(fid_mean, fid_std)) + 0.03)\nax2.set_ylim(top=100.0)\nfor rect, val in zip(bar1, fid_mean):\n    ax1.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.0025, f"{{val:.3f}}", ha='center', va='bottom', fontsize=9.4, color='#4C78A8')\nfor rect, val in zip(bar2, util_mean):\n    ax2.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.08, f"{{val:.2f}}", ha='center', va='bottom', fontsize=9.2, color='#F28E2B')\nfig.tight_layout()\nfig.savefig(PNG_PATH, bbox_inches='tight')\nfig.savefig(PDF_PATH, bbox_inches='tight')\n'''
    (out_dir / f'{stem}.py').write_text(script, encoding='utf-8')


def plot_dual_axis(rows: List[Dict[str, float]], out_dir: Path, stem: str) -> None:
    labels = [r['display'] for r in rows]
    fid_mean = [r['avg_fidelity_mean'] for r in rows]
    fid_std = [math.sqrt(r['avg_fidelity_var']) for r in rows]
    util_mean = [r['avg_normalized_eff_util_mean'] * 100.0 for r in rows]
    util_std = [math.sqrt(r['avg_normalized_eff_util_var']) * 100.0 for r in rows]
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    fig, ax1 = plt.subplots(figsize=(7.2, 2.7), dpi=220)
    ax2 = ax1.twinx()
    bars1 = ax1.bar(x - width / 2, fid_mean, width=width, yerr=fid_std, capsize=3, color='#4C78A8', alpha=0.92, label='Avg Fidelity')
    bars2 = ax2.bar(x + width / 2, util_mean, width=width, yerr=util_std, capsize=3, color='#F28E2B', alpha=0.85, label='Avg Norm. Eff. Util. (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel('Avg Fidelity', fontsize=11)
    ax2.set_ylabel('Avg Norm. Eff. Util. (%)', fontsize=11)
    ax1.tick_params(axis='y', labelsize=10)
    ax2.tick_params(axis='y', labelsize=10)
    ax1.grid(True, axis='y', linestyle='--', linewidth=0.55, alpha=0.28)
    ax1.legend([bars1, bars2], ['Avg Fidelity', 'Avg Norm. Eff. Util. (%)'], loc='upper center', bbox_to_anchor=(0.5, 0.985), ncol=2, fontsize=10, frameon=True)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.set_ylim(top=max(v + s for v, s in zip(fid_mean, fid_std)) + 0.03)
    ax2.set_ylim(top=100.0)
    for rect, val in zip(bars1, fid_mean):
        ax1.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.0025, f'{val:.3f}', ha='center', va='bottom', fontsize=9.4, color='#4C78A8')
    for rect, val in zip(bars2, util_mean):
        ax2.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.08, f'{val:.2f}', ha='center', va='bottom', fontsize=9.2, color='#F28E2B')
    fig.tight_layout()
    fig.savefig(out_dir / f'{stem}.png', dpi=220, bbox_inches='tight')
    fig.savefig(out_dir / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)


def build_rows_for_machine(machine: str) -> List[Dict[str, float]]:
    rows = []
    for key, runs in RUN_GROUPS.items():
        eval_utils = EVAL_UTILS if key == 'mean(30,60,88)' else [int(key[-2:])]
        metrics = aggregate_group_metrics(runs, machine, eval_utils)
        rows.append({'setting': key, 'display': DISPLAY_LABELS[key], **metrics})
    return rows


def average_machine_rows(rows_a: List[Dict[str, float]], rows_b: List[Dict[str, float]]) -> List[Dict[str, float]]:
    by_a = {r['setting']: r for r in rows_a}
    by_b = {r['setting']: r for r in rows_b}
    out = []
    for key in DISPLAY_LABELS:
        a = by_a[key]
        b = by_b[key]
        avg = {'setting': key, 'display': DISPLAY_LABELS[key]}
        for fld in ['top10_fidelity_mean','top10_fidelity_var','top10_normalized_eff_util_mean','top10_normalized_eff_util_var','top20_fidelity_mean','top20_fidelity_var','top20_normalized_eff_util_mean','top20_normalized_eff_util_var','avg_fidelity_mean','avg_fidelity_var','avg_normalized_eff_util_mean','avg_normalized_eff_util_var']:
            avg[fld] = (a[fld] + b[fld]) / 2.0
        out.append(avg)
    return out


def write_csv(rows: List[Dict[str, float]], path: Path) -> None:
    fields = list(rows[0].keys())
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = FIG_BASE / f'avg_run_{now}'
    base_dir.mkdir(parents=True, exist_ok=True)

    torino_rows = build_rows_for_machine('torino')
    marr_rows = build_rows_for_machine('marrakesh')
    avg_rows = average_machine_rows(torino_rows, marr_rows)

    for name, rows in [('torino', torino_rows), ('marrakesh', marr_rows), ('avg', avg_rows)]:
        out_dir = base_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f'figure_util_aggregation_ablation_dual_axis_{name}'
        csv_path = out_dir / f'{stem}.csv'
        write_csv(rows, csv_path)
        write_local_plot_script(out_dir, stem)
        plot_dual_axis(rows, out_dir, stem)

    print(base_dir)

if __name__ == '__main__':
    main()
