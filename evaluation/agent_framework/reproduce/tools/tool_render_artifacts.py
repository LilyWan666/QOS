#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

from repro_toolkit import (
    assert_action_allowed,
    append_history,
    blocked_action_payload,
    get_fsm_state,
    load_state,
    next_actions,
    set_fsm_state,
    write_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, help="Path to shared toolchain state.json")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def png_info(path: Path) -> dict[str, Any]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return {"exists": False, "valid_png": False, "reason": str(exc)}
    if len(data) < 33 or not data.startswith(b"\x89PNG\r\n\x1a\n") or data[12:16] != b"IHDR":
        return {"exists": True, "valid_png": False, "size_bytes": len(data)}
    width, height = struct.unpack(">II", data[16:24])
    return {
        "exists": True,
        "valid_png": True,
        "size_bytes": len(data),
        "width": int(width),
        "height": int(height),
    }


def method_label(method: str) -> str:
    return {
        "no_multiprogramming": "No M/P",
        "baseline_multiprogramming": "Baseline M/P",
        "qos_multiprogramming": "QOS M/P",
    }.get(method, method.replace("_", " ").title())


def util_label(value: Any) -> str:
    try:
        return f"{int(round(float(value) * 100))}%"
    except (TypeError, ValueError):
        return str(value)


def utilization_labels(metrics: dict[str, Any], values: list[Any]) -> list[str]:
    selected = metrics.get("threshold_selected_qubit_sequence") or []
    qpu_qubits = int(metrics.get("qpu_qubits") or 27)
    labels: list[str] = []
    for idx, value in enumerate(values):
        label = util_label(value)
        if idx < len(selected):
            try:
                qubits = int(selected[idx])
                pct = int(round((100.0 * qubits / max(qpu_qubits, 1)) / 5.0) * 5)
                label = f"{pct}% ({qubits}q)"
            except (TypeError, ValueError):
                label = f"{label} ({selected[idx]}q)"
        labels.append(label)
    return labels


def render_grouped_metric(
    metrics: dict[str, Any],
    metric_name: str,
    ylabel: str,
    title: str,
    output_path: Path,
    *,
    methods: list[str] | None = None,
    value_scale: float = 1.0,
    y_max: float | None = 1.08,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    results = metrics.get("results") or []
    methods = methods or ["no_multiprogramming", "baseline_multiprogramming", "qos_multiprogramming"]
    utilizations = [item.get("utilization") for item in results if isinstance(item, dict)]
    labels = utilization_labels(metrics, utilizations)
    x = np.arange(len(labels))
    width = min(0.32, 0.72 / max(len(methods), 1))

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    method_colors = {
        "no_multiprogramming": "#8CBCEB",
        "baseline_multiprogramming": "#F2A65A",
        "qos_multiprogramming": "#7BD88F",
    }
    for idx, method in enumerate(methods):
        values: list[float] = []
        for item in results:
            method_payload = ((item.get("methods") or {}).get(method) or {}) if isinstance(item, dict) else {}
            try:
                values.append(float(method_payload.get(metric_name, 0.0)) * value_scale)
            except (TypeError, ValueError):
                values.append(0.0)
        offset = (idx - (len(methods) - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=method_label(method),
            color=method_colors.get(method, "#9AA4B2"),
            edgecolor="#27313C",
            linewidth=0.6,
        )

    ax.set_title(title, pad=12)
    ax.set_xlabel("Target utilization")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if y_max is None:
        y_max = max(1.0, max((patch.get_height() for patch in ax.patches), default=1.0) * 1.10)
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", color="#D8DEE9", linewidth=0.8, alpha=0.75)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def render_relative_fidelity(metrics: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [row for row in (metrics.get("relative_fidelity_by_application") or []) if isinstance(row, dict)]
    thresholds = list(dict.fromkeys(row.get("threshold") for row in rows))
    thresholds = sorted(thresholds, key=lambda item: float(item))
    applications = list(dict.fromkeys(str(row.get("application", "")) for row in rows if row.get("application")))
    threshold_labels = utilization_labels(metrics, thresholds)
    threshold_values: list[list[float]] = []
    for threshold in thresholds:
        values: list[float] = []
        for app in applications:
            match = None
            for row in rows:
                try:
                    if str(row.get("application")) == app and abs(float(row.get("threshold")) - float(threshold)) < 1e-9:
                        match = float(row.get("relative_fidelity", 0.0))
                        break
                except (TypeError, ValueError):
                    pass
            values.append(float(match) if match is not None else 0.0)
        threshold_values.append(values)

    fig, ax = plt.subplots(figsize=(12.5, 5.0), dpi=180)
    x = np.arange(len(thresholds))
    width = min(0.075, 0.78 / max(len(applications), 1))
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(applications), 1)))
    for app_idx, app in enumerate(applications):
        values = [threshold_values[t_idx][app_idx] for t_idx in range(len(thresholds))]
        offset = (app_idx - (len(applications) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=app, color=colors[app_idx], edgecolor="#27313C", linewidth=0.4)

    ax.set_title("Figure 11(c): Relative fidelity by application", pad=12)
    ax.set_xlabel("Utilization target")
    ax.set_ylabel("Rel. Fidelity")
    ax.set_xticks(x)
    ax.set_xticklabels(threshold_labels)
    all_values = [value for values in threshold_values for value in values]
    ax.set_ylim(0, max(1.2, max(all_values or [1.0]) * 1.12))
    ax.grid(axis="y", color="#D8DEE9", linewidth=0.8, alpha=0.75)
    ax.legend(frameon=False, loc="upper right", ncol=3, fontsize=8, title="Application")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def render_qos_fig11(metrics_path: Path, artifact_dir: Path) -> dict[str, Any]:
    metrics = load_json(metrics_path)
    metrics["qpu_qubits"] = int(metrics.get("qpu_qubits") or 27)
    if metrics.get("figure_id") != "original_pipeline_raw_metrics":
        return {"rendered": False, "reason": "metrics are not original_pipeline_raw_metrics"}
    if not isinstance(metrics.get("results"), list) or not isinstance(metrics.get("relative_fidelity_by_application"), list):
        return {"rendered": False, "reason": "metrics missing results or relative_fidelity_by_application"}

    figure_dir = artifact_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = {
        "figure_11a": figure_dir / "figure_11a.png",
        "figure_11b": figure_dir / "figure_11b.png",
        "figure_11c": figure_dir / "figure_11c.png",
    }
    render_grouped_metric(metrics, "fidelity", "Fidelity", "Figure 11(a): Fidelity by method", figure_paths["figure_11a"])
    render_grouped_metric(
        metrics,
        "effective_utilization",
        "Effective utilization [%]",
        "Figure 11(b): Effective utilization",
        figure_paths["figure_11b"],
        methods=["baseline_multiprogramming", "qos_multiprogramming"],
        value_scale=100.0,
        y_max=100.0,
    )
    render_relative_fidelity(metrics, figure_paths["figure_11c"])

    output_files = dict(metrics.get("output_files") or {})
    figure_info: dict[str, Any] = {}
    for figure_id, path in figure_paths.items():
        info = png_info(path)
        output_files[figure_id] = str(path)
        output_files[f"{figure_id}_exists"] = bool(info.get("exists"))
        output_files[f"{figure_id}_valid_png"] = bool(info.get("valid_png"))
        output_files[f"{figure_id}_size_bytes"] = int(info.get("size_bytes") or 0)
        figure_info[figure_id] = {"path": str(path), **info}

    metrics["output_files"] = output_files
    metrics.setdefault("rendering", {})
    metrics["rendering"].update(
        {
            "tool": "repro_render_artifacts",
            "mode": "original_pipeline_raw_metrics",
            "figure_dir": str(figure_dir),
            "figures": {
                "figure_11a": {
                    "methods": ["no_multiprogramming", "baseline_multiprogramming", "qos_multiprogramming"],
                    "x_groups": utilization_labels(metrics, [item.get("utilization") for item in metrics.get("results", []) if isinstance(item, dict)]),
                    "value_scale": 1.0,
                    "y_axis": "Fidelity",
                },
                "figure_11b": {
                    "methods": ["baseline_multiprogramming", "qos_multiprogramming"],
                    "x_groups": utilization_labels(metrics, [item.get("utilization") for item in metrics.get("results", []) if isinstance(item, dict)]),
                    "value_scale": 100.0,
                    "y_axis": "Effective utilization [%]",
                },
                "figure_11c": {
                    "methods": ["qos_multiprogramming"],
                    "x_groups": utilization_labels(metrics, sorted(list(dict.fromkeys(row.get("threshold") for row in metrics.get("relative_fidelity_by_application", []) if isinstance(row, dict))), key=lambda item: float(item))),
                    "applications": list(dict.fromkeys(str(row.get("application", "")) for row in metrics.get("relative_fidelity_by_application", []) if isinstance(row, dict) and row.get("application"))),
                    "aggregation": "none_per_application_by_utilization_target",
                    "grouping": "utilization_target_outer_application_inner",
                    "bars_per_group": len(list(dict.fromkeys(str(row.get("application", "")) for row in metrics.get("relative_fidelity_by_application", []) if isinstance(row, dict) and row.get("application")))),
                    "bar_count": len(metrics.get("relative_fidelity_by_application", []) or []),
                    "value_scale": 1.0,
                    "y_axis": "Relative fidelity",
                },
            },
        }
    )
    write_json(metrics_path, metrics)
    return {"rendered": True, "figure_dir": str(figure_dir), "figures": figure_info, "metrics_path": str(metrics_path)}


def main() -> int:
    args = parse_args()
    state_path = Path(args.state).resolve()
    state = load_state(state_path)
    is_allowed, allowed_actions = assert_action_allowed(state, "render_artifacts")
    if not is_allowed:
        payload = blocked_action_payload(
            tool="repro_render_artifacts",
            action="render_artifacts",
            state=state,
            next_allowed_actions=allowed_actions,
        )
        append_history(state, "repro_render_artifacts", payload)
        state["last_step"] = "repro_render_artifacts"
        state["last_status"] = "blocked"
        write_state(state_path, state)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    run_root = Path(str(state.get("run_root", "."))).resolve()
    artifact_dir = run_root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifacts = (((state.get("last_manifest") or {}).get("artifacts")) or {})
    metrics_path_raw = artifacts.get("metrics")
    render_result: dict[str, Any] = {"rendered": False, "reason": "missing metrics artifact"}
    if metrics_path_raw:
        render_result = render_qos_fig11(Path(str(metrics_path_raw)).resolve(), artifact_dir)
    pre_render_state = get_fsm_state(state)
    pre_render_status = str(state.get("last_status", ""))
    render_outcome = "success" if bool(render_result.get("rendered")) or pre_render_state == "SUCCESS" else "failed"

    summary = {
        "recipe_name": state.get("recipe_name"),
        "last_status": pre_render_status,
        "last_step": state.get("last_step"),
        "pre_render_state": pre_render_state,
        "pre_render_status": pre_render_status,
        "render_outcome": render_outcome,
        "render_result": render_result,
        "fix_budget": state.get("fix_budget"),
        "history_count": len(state.get("history", [])) if isinstance(state.get("history"), list) else 0,
        "last_manifest": state.get("last_manifest"),
        "last_verdict_path": state.get("last_verdict_path"),
    }
    summary_path = artifact_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = {
        "tool": "repro_render_artifacts",
        "status": "success",
        "artifact_dir": str(artifact_dir),
        "summary_path": str(summary_path),
        "summary": summary,
        "render_result": render_result,
        "render_outcome": render_outcome,
        "pre_render_state": pre_render_state,
        "pre_render_status": pre_render_status,
    }
    state["last_step"] = "repro_render_artifacts"
    state["last_status"] = "artifacts_rendered"
    state["last_artifact_summary_path"] = str(summary_path)
    state["render_outcome"] = render_outcome
    set_fsm_state(state, "ARTIFACTS_RENDERED")
    payload["fsm_state"] = get_fsm_state(state)
    payload["next_allowed_actions"] = next_actions(state)
    append_history(state, "repro_render_artifacts", payload)
    write_state(state_path, state)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
