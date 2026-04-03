#!/usr/bin/env python3
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


ROOT = Path("/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing")
FIG_BASE = ROOT / "figures" / "six_model_score_progress"
OVERVIEW_SELECTED = ROOT / "figures" / "six_model_overview" / "run_20260305_195621" / "selected_runs.json"


@dataclass(frozen=True)
class ProviderPanel:
    title: str
    labels: Tuple[str, ...]
    colors: Tuple[str, ...]


PANELS: Tuple[ProviderPanel, ...] = (
    ProviderPanel("Gemini", ("Gemini 3 Pro", "Gemini 3 Flash"), ("#4C78A8", "#54A24B")),
    ProviderPanel("GPT", ("GPT-5 mini", "GPT-5.3 Codex"), ("#E45756", "#7E6BC4")),
    ProviderPanel("Claude", ("Claude Sonnet 4.6", "Claude Opus 4.6"), ("#C9B458", "#63B5D1")),
)


def comparable_score(metrics: Dict[str, float]) -> float:
    score = metrics.get("combined_score", metrics.get("score"))
    if score is None:
        return -100.0
    value = float(score)
    return value if np.isfinite(value) else -100.0


def load_selected_runs() -> Dict[str, Dict[str, str]]:
    with OVERVIEW_SELECTED.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_progress(trace_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_scores: List[float] = []
    best_scores: List[float] = []
    iterations: List[int] = []

    with trace_path.open("r", encoding="utf-8") as f:
        first = json.loads(next(f))
        init_score = comparable_score(first["parent_metrics"])
        raw_scores.append(init_score)
        iterations.append(0)
        raw_scores.append(comparable_score(first["child_metrics"]))
        iterations.append(1)

        for line in f:
            obj = json.loads(line)
            raw_scores.append(comparable_score(obj["child_metrics"]))
            iterations.append(int(obj["iteration"]))

    best_so_far = float("-inf")
    for score in raw_scores:
        best_so_far = max(best_so_far, score)
        best_scores.append(best_so_far)

    return (
        np.asarray(iterations, dtype=int),
        np.asarray(raw_scores, dtype=float),
        np.asarray(best_scores, dtype=float),
    )


def style_axes(fig, ax) -> None:
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.9)
    ax.grid(axis="x", color="#efefef", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def save_figure(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=240)
    fig.savefig(out_dir / f"{stem}.pdf")


def main() -> None:
    selected_runs = load_selected_runs()
    out_dir = FIG_BASE / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    progress: Dict[str, Dict[str, np.ndarray]] = {}
    summary_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []

    y_min = float("inf")
    y_max = float("-inf")

    for label, payload in selected_runs.items():
        run_dir = Path(payload["run_dir"])
        trace_path = run_dir / "evolution_trace.jsonl"
        iterations, raw_scores, best_scores = collect_progress(trace_path)
        progress[label] = {
            "iterations": iterations,
            "raw_scores": raw_scores,
            "best_scores": best_scores,
        }
        y_min = min(y_min, float(np.min(best_scores)))
        y_max = max(y_max, float(np.max(best_scores)))
        summary_rows.append(
            {
                "model": label,
                "run_dir": str(run_dir),
                "initial_combined_score": float(raw_scores[0]),
                "best_combined_score": float(np.max(best_scores)),
                "final_combined_score": float(raw_scores[-1]),
                "iterations": int(iterations[-1]),
            }
        )
        for idx, raw_score, best_score in zip(iterations, raw_scores, best_scores):
            detail_rows.append(
                {
                    "model": label,
                    "run_dir": str(run_dir),
                    "iteration": int(idx),
                    "combined_score": float(raw_score),
                    "best_combined_score_so_far": float(best_score),
                }
            )

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.titlesize": 22,
            "axes.labelsize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 18,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True, constrained_layout=True)
    y_lower = 2.0 * np.floor(y_min / 2.0)
    y_upper = 2.0 * np.ceil(y_max / 2.0)
    if y_upper <= y_lower:
        y_upper = y_lower + 2.0

    for ax, panel in zip(axes, PANELS):
        style_axes(fig, ax)
        for label, color in zip(panel.labels, panel.colors):
            series = progress[label]
            x = series["iterations"]
            best = series["best_scores"]
            ax.plot(x, best, color=color, linewidth=2.2, drawstyle="steps-post", label=label)
        ax.set_title(panel.title)
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_ylim(y_lower, y_upper)
        ax.yaxis.set_major_locator(MultipleLocator(2.0))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)

    axes[0].set_ylabel("Best Score So Far")
    axes[1].set_xlabel("Iteration")
    save_figure(fig, out_dir, "figure_score_progress")
    plt.close(fig)

    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "run_dir",
                "initial_combined_score",
                "best_combined_score",
                "final_combined_score",
                "iterations",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    detail_csv = out_dir / "progress_detail.csv"
    with detail_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "run_dir",
                "iteration",
                "combined_score",
                "best_combined_score_so_far",
            ],
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    with (out_dir / "selected_runs.json").open("w", encoding="utf-8") as f:
        json.dump(selected_runs, f, indent=2)

    print(f"[OK] Output: {out_dir}")
    print(f"  - {out_dir / 'figure_score_progress.png'}")
    print(f"  - {out_dir / 'figure_score_progress.pdf'}")
    print(f"  - {summary_csv}")
    print(f"  - {detail_csv}")


if __name__ == "__main__":
    main()
