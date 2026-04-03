#!/usr/bin/env python3
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path("/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing/openevolve_output")
OUTPUT_DIR = Path("/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing/figures/six_model_overview")


@dataclass(frozen=True)
class ModelSpec:
    label: str
    glob_pattern: str


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        label="Gemini 3 Pro",
        glob_pattern="GeminiMulti_gemini-3-pro-preview*_iter100_*/*run01_*",
    ),
    ModelSpec(
        label="Gemini 3 Flash",
        glob_pattern="GeminiMulti_gemini-3-flash-preview*_iter100_*/*run01_*",
    ),
    ModelSpec(
        label="GPT-5 mini",
        glob_pattern="OpenAI_gpt-5-mini_iter100_*",
    ),
    ModelSpec(
        label="GPT-5.3 Codex",
        glob_pattern="OpenAI_gpt-5.3-codex_iter100_*",
    ),
    ModelSpec(
        label="Claude Sonnet 4.6",
        glob_pattern="OpenAI_claude-sonnet-4-6_iter100_*",
    ),
    ModelSpec(
        label="Claude Opus 4.6",
        glob_pattern="OpenAI_claude-opus-4-6_iter100_*",
    ),
]


def pick_latest_run(pattern: str) -> Path:
    candidates = sorted(
        BASE_DIR.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No run directory matches pattern: {pattern}")
    run_dir = candidates[0]
    trace_file = run_dir / "evolution_trace.jsonl"
    cfg_file = run_dir / "openevolve_gpt.yaml"
    if not trace_file.exists() or not cfg_file.exists():
        raise FileNotFoundError(f"Missing trace/config under run directory: {run_dir}")
    return run_dir


def as_float(value) -> float:
    if value is None:
        return 0.0
    return float(value)


def collect_metrics(trace_path: Path) -> Dict[str, float]:
    prompt_tokens = 0.0
    completion_tokens = 0.0
    input_cost = 0.0
    output_cost = 0.0
    llm_time = 0.0
    eval_time = 0.0
    rows = 0

    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows += 1
            obj = json.loads(line)
            stats = obj.get("artifacts", {}).get("qos_iteration_stats_json", {})

            prompt_tokens += as_float(stats.get("prompt_tokens", stats.get("input_prompt_tokens")))
            completion_tokens += as_float(
                stats.get("output_tokens", stats.get("output_response_tokens"))
            )
            input_cost += as_float(stats.get("cost_input_usd"))
            output_cost += as_float(stats.get("cost_output_usd"))
            llm_time += as_float(stats.get("llm_time_sec"))
            eval_time += as_float(stats.get("evaluation_time_sec"))

    return {
        "iterations": rows,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": input_cost + output_cost,
        "llm_time_sec": llm_time,
        "eval_time_sec": eval_time,
        "total_time_sec": llm_time + eval_time,
    }


def read_primary_model(cfg_path: Path) -> str:
    pattern = re.compile(r"^\s*primary_model:\s*(.+?)\s*$")
    for line in cfg_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.match(line)
        if m:
            return m.group(1).strip()
    return ""


def apply_cost_fallback(
    metric: Dict[str, float],
    model_id: str,
    run_dir_name: str,
) -> Dict[str, float]:
    # Historical traces may have price fields left as 0. Backfill with model pricing.
    if metric["total_cost_usd"] > 0.0:
        return metric

    pricing: Dict[str, Tuple[float, float]] = {
        # OpenAI official pricing (2026-03-09): https://platform.openai.com/docs/pricing
        "gpt-5-mini": (0.25, 2.0),
        "gpt-5-mini:flex": (0.125, 1.0),
        "gpt-5.3-codex": (1.75, 14.0),
        # Gemini API pricing (2026-03-05): https://ai.google.dev/gemini-api/docs/pricing
        "gemini-3-pro-preview": (2.5, 15.0),
    }

    key = model_id
    if model_id == "gpt-5-mini" and "flex" in run_dir_name.lower():
        key = "gpt-5-mini:flex"

    if key not in pricing:
        return metric

    in_rate, out_rate = pricing[key]
    prompt_cost = metric["prompt_tokens"] / 1_000_000.0 * in_rate
    completion_cost = metric["completion_tokens"] / 1_000_000.0 * out_rate

    metric = dict(metric)
    metric["input_cost_usd"] = prompt_cost
    metric["output_cost_usd"] = completion_cost
    metric["total_cost_usd"] = prompt_cost + completion_cost
    return metric


def format_k(value: float) -> str:
    return f"{value / 1000.0:.1f}k"


def style_axes(fig, ax) -> None:
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#d9d9d9", linewidth=1.0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=13)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def save_figure(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=220)
    fig.savefig(out_dir / f"{stem}.pdf")


def make_plots(labels: List[str], stats: List[Dict[str, float]], out_dir: Path) -> None:
    x = np.arange(len(labels))

    prompt_tokens = np.array([s["prompt_tokens"] for s in stats])
    completion_tokens = np.array([s["completion_tokens"] for s in stats])
    input_cost = np.array([s["input_cost_usd"] for s in stats])
    output_cost = np.array([s["output_cost_usd"] for s in stats])
    llm_time = np.array([s["llm_time_sec"] for s in stats])
    eval_time = np.array([s["eval_time_sec"] for s in stats])

    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "legend.fontsize": 16,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
        }
    )

    # Figure 1: Tokens
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    style_axes(fig1, ax1)
    ax1.bar(x, prompt_tokens, label="Prompt", color="#d9824f", edgecolor="black")
    ax1.bar(
        x,
        completion_tokens,
        bottom=prompt_tokens,
        label="Completion",
        color="#7f6db0",
        edgecolor="black",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("Total Tokens")
    ax1.legend(loc="upper left")
    for i, total in enumerate(prompt_tokens + completion_tokens):
        ax1.text(i, total + max(total * 0.01, 1), format_k(total), ha="center", va="bottom", fontsize=16)
    fig1.tight_layout()
    save_figure(fig1, out_dir, "figure_tokens")
    plt.close(fig1)

    # Figure 2: Cost (USD)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    style_axes(fig2, ax2)
    ax2.bar(x, input_cost, label="Prompt/Input Cost", color="#4c72b0", edgecolor="black")
    ax2.bar(
        x,
        output_cost,
        bottom=input_cost,
        label="Completion/Output Cost",
        color="#dd8452",
        edgecolor="black",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("Estimated Total Cost (USD)")
    ax2.legend(loc="upper left")
    for i, total in enumerate(input_cost + output_cost):
        ax2.text(i, total + max(total * 0.01, 0.01), f"${total:.2f}", ha="center", va="bottom", fontsize=16)
    fig2.tight_layout()
    save_figure(fig2, out_dir, "figure_cost_usd")
    plt.close(fig2)

    # Figure 3: Runtime (sec)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    style_axes(fig3, ax3)
    ax3.bar(x, llm_time, label="LLM", color="#4c72b0", edgecolor="black")
    ax3.bar(
        x,
        eval_time,
        bottom=llm_time,
        label="Evaluation",
        color="#55a868",
        edgecolor="black",
    )
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=15, ha="right")
    ax3.set_ylabel("Total Time (s)")
    ax3.legend(loc="upper left")
    total_time = llm_time + eval_time
    for i, total in enumerate(total_time):
        ax3.text(i, total + max(total * 0.01, 1), f"{total / 3600.0:.2f}h", ha="center", va="bottom", fontsize=16)
    fig3.tight_layout()
    save_figure(fig3, out_dir, "figure_runtime")
    plt.close(fig3)


def main() -> None:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / f"run_{now}"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels: List[str] = []
    stats: List[Dict[str, float]] = []
    run_paths: Dict[str, Path] = {}
    model_ids: Dict[str, str] = {}

    for spec in MODEL_SPECS:
        run_dir = pick_latest_run(spec.glob_pattern)
        run_paths[spec.label] = run_dir
        trace_path = run_dir / "evolution_trace.jsonl"
        model_id = read_primary_model(run_dir / "openevolve_gpt.yaml")
        model_ids[spec.label] = model_id
        metric = collect_metrics(trace_path)
        metric = apply_cost_fallback(metric=metric, model_id=model_id, run_dir_name=run_dir.name)
        labels.append(spec.label)
        stats.append(metric)

    make_plots(labels, stats, out_dir)

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "run_dir",
                "iterations",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "input_cost_usd",
                "output_cost_usd",
                "total_cost_usd",
                "llm_time_sec",
                "eval_time_sec",
                "total_time_sec",
            ]
        )
        for label, metric in zip(labels, stats):
            writer.writerow(
                [
                    label,
                    str(run_paths[label]),
                    int(metric["iterations"]),
                    f"{metric['prompt_tokens']:.6f}",
                    f"{metric['completion_tokens']:.6f}",
                    f"{metric['total_tokens']:.6f}",
                    f"{metric['input_cost_usd']:.6f}",
                    f"{metric['output_cost_usd']:.6f}",
                    f"{metric['total_cost_usd']:.6f}",
                    f"{metric['llm_time_sec']:.6f}",
                    f"{metric['eval_time_sec']:.6f}",
                    f"{metric['total_time_sec']:.6f}",
                ]
            )

    with (out_dir / "selected_runs.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                label: {
                    "run_dir": str(run_paths[label]),
                    "primary_model": model_ids.get(label, ""),
                }
                for label in labels
            },
            f,
            indent=2,
        )

    print(f"Saved outputs to: {out_dir}")
    print(f"  - {out_dir / 'figure_tokens.png'}")
    print(f"  - {out_dir / 'figure_cost_usd.png'}")
    print(f"  - {out_dir / 'figure_runtime.png'}")
    print(f"  - {csv_path}")


if __name__ == "__main__":
    main()
