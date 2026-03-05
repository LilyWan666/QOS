#!/usr/bin/env python3
import csv
import glob
import math
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing")
OUT = ROOT / "openevolve_output" / "_proxy_depthratio_singleutil_physical_306088_8196_20260220"
PLOT = ROOT / "plot_qos_vs_evolved_from_csv.py"
PY = Path("/work/nvme/becn/lily/miniconda3/envs/qos_fig11/bin/python")
TARGET_ORIG = ROOT / "target_orig.py"

# Keep the same train-util set currently in this folder.
TRAIN_UTILS = [30, 37, 45, 52, 60, 67, 81, 88]
EVAL_UTILS = [30, 60, 88]


def find_best_program(train_util: int) -> Path:
    pattern = str(
        ROOT
        / "openevolve_output"
        / f"Qwen2.5-14B-Instruct_iter200_*_u{train_util}_proxy_depth_ratio_iter200_*"
        / "best"
        / "best_program.py"
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing best_program for u{train_util}: {pattern}")
    return Path(matches[-1])


def clean_csv(src: Path, dst: Path) -> None:
    with open(src, "r", encoding="utf-8", newline="") as fin:
        rd = csv.DictReader(fin)
        if not rd.fieldnames:
            raise RuntimeError(f"No header in {src}")
        fieldnames = [h for h in rd.fieldnames if h is not None]
        with open(dst, "w", encoding="utf-8", newline="") as fout:
            wr = csv.DictWriter(fout, fieldnames=fieldnames)
            wr.writeheader()
            for row in rd:
                row.pop(None, None)
                wr.writerow({k: row.get(k, "") for k in fieldnames})


def weighted_mean_from_bar(path: Path, mean_key: str, count_key: str) -> tuple[float, int]:
    total = 0.0
    count = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            try:
                c = int(float(row.get(count_key, 0) or 0))
            except Exception:
                c = 0
            try:
                m = float(row.get(mean_key, "nan"))
            except Exception:
                m = float("nan")
            if c > 0 and math.isfinite(m):
                total += m * c
                count += c
    if count <= 0:
        return float("nan"), 0
    return float(total / count), count


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    methods: dict[int, Path] = {u: find_best_program(u) for u in TRAIN_UTILS}
    with open(OUT / "methods.tsv", "w", encoding="utf-8") as f:
        for u in TRAIN_UTILS:
            f.write(f"proxy_depth_ratio_u{u}\t{methods[u]}\n")

    clean_inputs: dict[int, Path] = {}
    for eu in EVAL_UTILS:
        raw = ROOT / "pairing_metadata" / f"pair_metrics_util{eu}_shots8196.csv"
        clean = OUT / f"_clean_pair_metrics_util{eu}_shots8196.csv"
        clean_csv(raw, clean)
        clean_inputs[eu] = clean

    for tu in TRAIN_UTILS:
        prog = methods[tu]
        for eu in EVAL_UTILS:
            csv_path = clean_inputs[eu]
            for tk in (10, 20):
                tag = f"proxy_depth_ratio_trainu{tu}_evalu{eu}_t{tk}"
                cmd = [
                    str(PY),
                    str(PLOT),
                    "--csv",
                    str(csv_path),
                    "--util",
                    str(eu),
                    "--shots",
                    "8196",
                    "--top-k-ratio",
                    str(tk),
                    "--orig-target",
                    str(TARGET_ORIG),
                    "--new-target",
                    str(prog),
                    "--out",
                    str(OUT / f"{tag}.png"),
                    "--out-csv",
                    str(OUT / f"{tag}.csv"),
                    "--bar-out",
                    str(OUT / f"{tag}_bar.png"),
                    "--bar-out-csv",
                    str(OUT / f"{tag}_bar.csv"),
                    "--summary-csv",
                    str(OUT / f"{tag}_summary.csv"),
                ]
                print(f"[RUN] {tag}", flush=True)
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    long_rows = []
    for tu in TRAIN_UTILS:
        for eu in EVAL_UTILS:
            for tk in (10, 20):
                bar_csv = OUT / f"proxy_depth_ratio_trainu{tu}_evalu{eu}_t{tk}_bar.csv"
                orig_mean, orig_count = weighted_mean_from_bar(bar_csv, "orig_mean_fidelity", "orig_count")
                new_mean, new_count = weighted_mean_from_bar(bar_csv, "new_mean_fidelity", "new_count")
                long_rows.append(
                    {
                        "train_util": tu,
                        "eval_util": eu,
                        "top_k": tk,
                        "orig_mean_fidelity": orig_mean,
                        "orig_count": orig_count,
                        "new_mean_fidelity": new_mean,
                        "new_count": new_count,
                    }
                )

    long_path = OUT / "proxy_depth_ratio_singleutil_physical_long.csv"
    with open(long_path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        wr.writeheader()
        wr.writerows(long_rows)

    agg_rows = []
    for tu in TRAIN_UTILS:
        sub = [r for r in long_rows if r["train_util"] == tu]
        t10 = [r for r in sub if r["top_k"] == 10]
        t20 = [r for r in sub if r["top_k"] == 20]
        top10_new = float(np.mean([r["new_mean_fidelity"] for r in t10]))
        top20_new = float(np.mean([r["new_mean_fidelity"] for r in t20]))
        top10_base = float(np.mean([r["orig_mean_fidelity"] for r in t10]))
        top20_base = float(np.mean([r["orig_mean_fidelity"] for r in t20]))
        overall_new = float((top10_new + top20_new) / 2.0)
        overall_base = float((top10_base + top20_base) / 2.0)
        agg_rows.append(
            {
                "train_util": tu,
                "top10_new_avg_eval306088": top10_new,
                "top20_new_avg_eval306088": top20_new,
                "top10_base_avg_eval306088": top10_base,
                "top20_base_avg_eval306088": top20_base,
                "overall_new": overall_new,
                "overall_base": overall_base,
                "delta_overall": float(overall_new - overall_base),
                "physical_shots": 8196,
                "eval_shots": 8196,
            }
        )

    agg_fields = [
        "train_util",
        "top10_new_avg_eval306088",
        "top20_new_avg_eval306088",
        "top10_base_avg_eval306088",
        "top20_base_avg_eval306088",
        "overall_new",
        "overall_base",
        "delta_overall",
        "physical_shots",
        "eval_shots",
    ]
    for name in (
        "proxy_depth_ratio_singleutil_physical_agg.csv",
        "proxy_depth_ratio_trainutil_on_physical_agg.csv",
        "proxy_depth_ratio_trainutil_on_physical_agg_with_u67_u81.csv",
    ):
        with open(OUT / name, "w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=agg_fields)
            wr.writeheader()
            wr.writerows(agg_rows)

    ranked = sorted(agg_rows, key=lambda r: r["overall_new"], reverse=True)
    with open(OUT / "proxy_depth_ratio_singleutil_physical_rank.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["rank", "train_util", "overall_new", "delta_overall"])
        wr.writeheader()
        for i, r in enumerate(ranked, start=1):
            wr.writerow(
                {
                    "rank": i,
                    "train_util": r["train_util"],
                    "overall_new": r["overall_new"],
                    "delta_overall": r["delta_overall"],
                }
            )

    labels = [f"u{r['train_util']}" for r in agg_rows]
    x = np.arange(len(labels))
    width = 0.35
    v10 = [r["top10_new_avg_eval306088"] for r in agg_rows]
    v20 = [r["top20_new_avg_eval306088"] for r in agg_rows]
    vo = [r["overall_new"] for r in agg_rows]
    b10 = float(np.mean([r["top10_base_avg_eval306088"] for r in agg_rows]))
    b20 = float(np.mean([r["top20_base_avg_eval306088"] for r in agg_rows]))
    bo = float(np.mean([r["overall_base"] for r in agg_rows]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
    ax1.bar(x - width / 2, v10, width, color="#1f77b4", label="Top10%")
    ax1.bar(x + width / 2, v20, width, color="#ff7f0e", label="Top20%")
    ax1.axhline(b10, color="#1f77b4", ls="--", lw=1, alpha=0.8, label=f"Baseline Top10% ({b10:.3f})")
    ax1.axhline(b20, color="#ff7f0e", ls="--", lw=1, alpha=0.8, label=f"Baseline Top20% ({b20:.3f})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Physical Mean Fidelity")
    ax1.set_xlabel("Train-Util Method")
    ax1.set_title("Physical Fidelity by Train-Util Method")
    ax1.grid(axis="y", alpha=0.2)
    ax1.legend(loc="upper right", fontsize=8)

    ax2.bar(x, vo, width=0.6, color="#2ca02c", label="Mean(Top10%, Top20%)")
    ax2.axhline(bo, color="gray", ls="--", lw=1, alpha=0.8, label=f"Baseline overall ({bo:.3f})")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Mean(Top10%, Top20%)")
    ax2.set_xlabel("Train-Util Method")
    ax2.set_title("Overall Score (Higher Better)")
    ax2.grid(axis="y", alpha=0.2)
    ax2.legend(loc="upper right", fontsize=8)
    for i, v in enumerate(vo):
        ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Proxy(depth_ratio) on Physical Data util30/60/88 (shots=8196)", fontsize=11)
    fig.tight_layout()
    for stem in ("proxy_depth_ratio_singleutil_physical_summary", "proxy_depth_ratio_trainutil_on_physical_summary"):
        fig.savefig(OUT / f"{stem}.png")
        fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)

    print("[DONE] overwritten depth_ratio outputs using pair_metrics_util30/60/88_shots8196.csv", flush=True)
    print(f"[BASELINE] top10={b10:.6f} top20={b20:.6f} overall={bo:.6f}", flush=True)


if __name__ == "__main__":
    run()
