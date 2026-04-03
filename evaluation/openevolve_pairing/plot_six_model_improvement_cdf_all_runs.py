#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing")
BASE_DIR = ROOT / "openevolve_output"
FIG_BASE = ROOT / "figures" / "six_model_improvement_cdf_all_runs"

MAX_ITER = 100
NO_IMPROVEMENT_SENTINEL = MAX_ITER + 1
EPS = 1e-12

ITER_SUCCESS_RE = re.compile(r"Iteration\s+(\d+):\s+Program .* completed in ")
ITER_ERROR_RE = re.compile(r"Iteration\s+(\d+)\s+error:")
COMBINED_RE = re.compile(r"combined_score=([+-]?\d+(?:\.\d+)?)")
EVAL_SCORE_RE = re.compile(r"Evaluated program .*combined_score=([+-]?\d+(?:\.\d+)?)")
NEW_BEST_RE = re.compile(
    r"New best program .* \(combined_score:\s*([+-]?\d+(?:\.\d+)?)\s*→\s*([+-]?\d+(?:\.\d+)?)"
)

MODEL_SPECS = [
    ("gem3pro", "Gemini 3 Pro", "#4C72B0", ("gemini-3-pro", "gemini3pro")),
    ("gem3flash", "Gemini 3 Flash", "#55A868", ("gemini-3-flash", "gemini3flash")),
    ("gpt5mini", "GPT-5 mini", "#C44E52", ("gpt-5-mini", "gpt5mini")),
    ("gpt53codex", "GPT-5.3 Codex", "#8172B3", ("gpt-5.3-codex", "gpt53codex", "gptcodex5.3", "gptcodex53")),
    (
        "claude_sonnet46",
        "Claude Sonnet 4.6",
        "#CCB974",
        ("claude-sonnet-4-6", "claude-sonnet-4.6", "claude_sonnet46", "claude46"),
    ),
    (
        "claude_opus46",
        "Claude Opus 4.6",
        "#64B5CD",
        ("claude-opus-4-6", "claude-opus-4.6", "claude_opus46", "opus46"),
    ),
]


def _safe_float(v: str | None) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def _model_family(name: str) -> str | None:
    lower = name.lower()
    for key, _, _, patterns in MODEL_SPECS:
        if any(pattern in lower for pattern in patterns):
            return key
    return None


def _discover_run_dirs() -> List[Tuple[str, Path]]:
    runs: Dict[Path, str] = {}
    for logs_dir in BASE_DIR.rglob("logs"):
        if not logs_dir.is_dir():
            continue
        run_dir = logs_dir.parent
        family = _model_family(run_dir.name) or _model_family(run_dir.parent.name)
        if family is None:
            continue
        runs[run_dir] = family
    return sorted(((family, run_dir) for run_dir, family in runs.items()), key=lambda item: str(item[1]))


def _parse_run_improvements(run_dir: Path) -> Tuple[float, float, int]:
    logs_dir = run_dir / "logs"
    if not logs_dir.exists():
        return float(NO_IMPROVEMENT_SENTINEL), float(NO_IMPROVEMENT_SENTINEL), 0

    events: Dict[int, Tuple[str, float]] = {}
    first_eval_score = float("nan")
    first_newbest_old = float("nan")

    for log_path in sorted(logs_dir.glob("*.log")):
        lines = log_path.read_text(errors="ignore").splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]

            if not math.isfinite(first_eval_score):
                m_eval = EVAL_SCORE_RE.search(line)
                if m_eval:
                    first_eval_score = _safe_float(m_eval.group(1))

            if not math.isfinite(first_newbest_old):
                m_nb = NEW_BEST_RE.search(line)
                if m_nb:
                    first_newbest_old = _safe_float(m_nb.group(1))

            m_ok = ITER_SUCCESS_RE.search(line)
            if m_ok:
                iteration = int(m_ok.group(1))
                score = float("nan")
                if i + 1 < len(lines):
                    m_score = COMBINED_RE.search(lines[i + 1])
                    if m_score:
                        score = _safe_float(m_score.group(1))
                        i += 1
                events[iteration] = ("success", score)
                i += 1
                continue

            m_err = ITER_ERROR_RE.search(line)
            if m_err:
                iteration = int(m_err.group(1))
                events[iteration] = ("error", float("nan"))

            i += 1

    if math.isfinite(first_newbest_old):
        baseline = first_newbest_old
    elif math.isfinite(first_eval_score):
        baseline = first_eval_score
    else:
        first_success = min(
            (it for it, (status, score) in events.items() if status == "success" and math.isfinite(score)),
            default=None,
        )
        baseline = events[first_success][1] if first_success is not None else float("nan")

    improvements = 0
    first_improvement_iter = NO_IMPROVEMENT_SENTINEL
    last_improvement_iter = NO_IMPROVEMENT_SENTINEL
    best_so_far = baseline

    for iteration in range(1, MAX_ITER + 1):
        status, score = events.get(iteration, ("missing", float("nan")))
        if status != "success" or not math.isfinite(score):
            continue
        if not math.isfinite(best_so_far):
            best_so_far = score
            continue
        if score > best_so_far + EPS:
            improvements += 1
            if first_improvement_iter == NO_IMPROVEMENT_SENTINEL:
                first_improvement_iter = iteration
            last_improvement_iter = iteration
            best_so_far = score

    return float(first_improvement_iter), float(last_improvement_iter), improvements


def _cdf_xy(values: List[float]) -> Tuple[List[float], List[float]]:
    if not values:
        return [], []
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return sorted_vals, [i / n for i in range(1, n + 1)]


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _style_axis(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="both", linestyle="--", linewidth=0.9, alpha=0.28, color="#9aa0a6")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    out_dir = FIG_BASE / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_display = {key: name for key, name, _, _ in MODEL_SPECS}
    run_rows: List[Dict[str, object]] = []
    first_improv_by_model: Dict[str, List[float]] = defaultdict(list)
    last_improv_by_model: Dict[str, List[float]] = defaultdict(list)
    improv_count_by_model: Dict[str, List[float]] = defaultdict(list)

    for family, run_dir in _discover_run_dirs():
        first_iter, last_iter, n_impr = _parse_run_improvements(run_dir)
        first_improv_by_model[family].append(first_iter)
        last_improv_by_model[family].append(last_iter)
        improv_count_by_model[family].append(float(n_impr))
        run_rows.append(
            {
                "run_dir": str(run_dir),
                "run_name": str(run_dir.relative_to(BASE_DIR)),
                "model_family": family,
                "model_name": model_display[family],
                "first_improvement_iteration": int(first_iter),
                "last_improvement_iteration": int(last_iter),
                "improvements_within_100_iter": int(n_impr),
            }
        )

    all_first = [v for vals in first_improv_by_model.values() for v in vals if v <= MAX_ITER]
    all_last = [v for vals in last_improv_by_model.values() for v in vals if v <= MAX_ITER]
    all_counts = [v for vals in improv_count_by_model.values() for v in vals if v > 0]

    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 4.8), dpi=240, constrained_layout=True, facecolor="white")
    ax_first, ax_last, ax_count = axes

    for ax in axes:
        _style_axis(ax)
        ax.set_ylim(0.0, 1.02)

    if all_first:
        x1, y1 = _cdf_xy(all_first)
        ax_first.step(x1, y1, where="post", linewidth=2.8, color="black")
    ax_first.set_xlabel("First Improve Iter")
    ax_first.set_ylabel("CDF")
    if all_first:
        x_min = max(1, int(min(all_first)) - 1)
        x_max = min(MAX_ITER, int(max(all_first)) + 1)
        if x_max <= x_min:
            x_max = min(MAX_ITER, x_min + 1)
        ax_first.set_xlim(x_min, x_max)
        span = x_max - x_min
        if span <= 10:
            ticks = list(range(x_min, x_max + 1))
        else:
            step = span / 5.0
            ticks = sorted({int(round(x_min + i * step)) for i in range(6)})
            if ticks[0] != x_min:
                ticks = [x_min] + ticks
            if ticks[-1] != x_max:
                ticks = ticks + [x_max]
        ax_first.set_xticks(ticks)
    else:
        ax_first.set_xlim(1, MAX_ITER)
        ax_first.set_xticks([1, 20, 40, 60, 80, 100])

    if all_last:
        x2, y2 = _cdf_xy(all_last)
        ax_last.step(x2, y2, where="post", linewidth=2.8, color="black")
    ax_last.set_xlabel("Last Improve Iter")
    ax_last.set_ylabel("CDF")
    if all_last:
        x_min = max(1, int(min(all_last)) - 1)
        x_max = min(MAX_ITER, int(max(all_last)) + 1)
        if x_max <= x_min:
            x_max = min(MAX_ITER, x_min + 1)
        ax_last.set_xlim(x_min, x_max)
        span = x_max - x_min
        if span <= 10:
            ticks = list(range(x_min, x_max + 1))
        else:
            step = span / 5.0
            ticks = sorted({int(round(x_min + i * step)) for i in range(6)})
            if ticks[0] != x_min:
                ticks = [x_min] + ticks
            if ticks[-1] != x_max:
                ticks = ticks + [x_max]
        ax_last.set_xticks(ticks)
    else:
        ax_last.set_xlim(1, MAX_ITER)
        ax_last.set_xticks([1, 20, 40, 60, 80, 100])

    if all_counts:
        x3, y3 = _cdf_xy(all_counts)
        ax_count.step(x3, y3, where="post", linewidth=2.8, color="black")
        xmin = int(min(all_counts))
        xmax = int(max(all_counts))
        if xmin == xmax:
            ax_count.set_xlim(xmin - 1, xmax + 1)
            ax_count.set_xticks([xmin])
        else:
            ax_count.set_xlim(xmin, xmax)
    ax_count.set_xlabel("Improve Count (100 iters)")
    ax_count.set_ylabel("CDF")

    figure_png = out_dir / "figure_improvement_cdfs_all_runs.png"
    figure_pdf = out_dir / "figure_improvement_cdfs_all_runs.pdf"
    per_run_csv = out_dir / "improvement_stats_per_run.csv"

    fig.savefig(figure_png, bbox_inches="tight")
    fig.savefig(figure_pdf, bbox_inches="tight")
    plt.close(fig)

    _write_csv(
        per_run_csv,
        sorted(run_rows, key=lambda r: str(r["run_name"])),
        [
            "run_name",
            "run_dir",
            "model_family",
            "model_name",
            "first_improvement_iteration",
            "last_improvement_iteration",
            "improvements_within_100_iter",
        ],
    )

    print(f"[ok] wrote {figure_png}")
    print(f"[ok] wrote {figure_pdf}")
    print(f"[ok] wrote {per_run_csv}")
    print(f"[ok] runs={len(run_rows)}")


if __name__ == "__main__":
    main()
