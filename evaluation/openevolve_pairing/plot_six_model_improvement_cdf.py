#!/usr/bin/env python3
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing")
FIG_BASE = ROOT / "figures" / "six_model_improvement_cdf"
OVERVIEW_SELECTED = ROOT / "figures" / "six_model_overview" / "run_20260305_195621" / "selected_runs.json"
EPS = 1e-12


def comparable_score(metrics: Dict[str, float]) -> float:
    avg_rank = float(metrics.get("avg_rank", float("inf")))
    return -avg_rank if np.isfinite(avg_rank) else -100.0


def load_selected_runs() -> Dict[str, Dict[str, str]]:
    with OVERVIEW_SELECTED.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_scores(trace_path: Path) -> List[Tuple[int, float]]:
    points: List[Tuple[int, float]] = []
    with trace_path.open("r", encoding="utf-8") as f:
        first = json.loads(next(f))
        points.append((0, comparable_score(first["parent_metrics"])))
        points.append((1, comparable_score(first["child_metrics"])))
        for line in f:
            obj = json.loads(line)
            points.append((int(obj["iteration"]), comparable_score(obj["child_metrics"])))
    return points


def summarize_improvements(points: List[Tuple[int, float]]) -> Dict[str, int]:
    best_so_far = points[0][1]
    first_improvement = None
    last_improvement = None
    improvement_count = 0

    for iteration, score in points[1:]:
        if score > best_so_far + EPS:
            best_so_far = score
            improvement_count += 1
            if first_improvement is None:
                first_improvement = iteration
            last_improvement = iteration

    return {
        "first_improvement_iteration": int(first_improvement) if first_improvement is not None else len(points) - 1,
        "last_improvement_iteration": int(last_improvement) if last_improvement is not None else 0,
        "num_improvements": int(improvement_count),
    }


def ecdf(values: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(arr) + 1, dtype=float) / len(arr)
    return arr, y


def style_axes(fig, ax) -> None:
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(axis="both", color="#e4e4e4", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def save_figure(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=240)
    fig.savefig(out_dir / f"{stem}.pdf")


def main() -> None:
    selected_runs = load_selected_runs()
    out_dir = FIG_BASE / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    first_vals: List[int] = []
    last_vals: List[int] = []
    count_vals: List[int] = []

    for label, payload in selected_runs.items():
        run_dir = Path(payload["run_dir"])
        points = collect_scores(run_dir / "evolution_trace.jsonl")
        summary = summarize_improvements(points)
        rows.append(
            {
                "model": label,
                "run_dir": str(run_dir),
                **summary,
            }
        )
        first_vals.append(summary["first_improvement_iteration"])
        last_vals.append(summary["last_improvement_iteration"])
        count_vals.append(summary["num_improvements"])

    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), dpi=240)
    panels = [
        ("Iteration of First Improvement", first_vals, (1, 100)),
        ("Iteration of Last Improvement", last_vals, (1, 100)),
        ("Number of Improvements in 100 Iterations", count_vals, (0, 100)),
    ]

    for ax, (xlabel, values, xlim) in zip(axes, panels):
        style_axes(fig, ax)
        xs, ys = ecdf(values)
        ax.step(xs, ys, where="post", color="black", linewidth=2.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CDF")
        ax.set_xlim(*xlim)
        ax.set_ylim(0.0, 1.02)

    fig.tight_layout()
    save_figure(fig, out_dir, "figure_improvement_cdf")
    plt.close(fig)

    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "run_dir",
                "first_improvement_iteration",
                "last_improvement_iteration",
                "num_improvements",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with (out_dir / "selected_runs.json").open("w", encoding="utf-8") as f:
        json.dump(selected_runs, f, indent=2)

    print(f"[OK] Output: {out_dir}")
    print(f"  - {out_dir / 'figure_improvement_cdf.png'}")
    print(f"  - {out_dir / 'figure_improvement_cdf.pdf'}")
    print(f"  - {summary_csv}")


if __name__ == "__main__":
    main()
