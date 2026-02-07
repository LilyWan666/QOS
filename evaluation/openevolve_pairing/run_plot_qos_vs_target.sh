#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG_FILE="${1:-$SCRIPT_DIR/plot_qos_vs_target.env}"

if [[ ! -f "$CFG_FILE" ]]; then
  echo "[ERR] Config file not found: $CFG_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CFG_FILE"

: "${PYTHON_BIN:=python}"
: "${CSV:?CSV is required in config}"
: "${UTIL:?UTIL is required in config}"
: "${SHOTS:?SHOTS is required in config}"
: "${TOP_K:=24}"
: "${UTIL_BIN_WIDTH:=0.02}"
: "${ORIG_TARGET:=}"
: "${NEW_TARGET:=}"
: "${OUTPUT_DIR:=}"
: "${DEFAULT_OUTPUT_DIR:=/work/nvme/betu/lily/QOS/temp/qos_seed_vs_target}"
: "${OUT:=}"
: "${OUT_CSV:=}"
: "${BAR_OUT:=}"
: "${BAR_OUT_CSV:=}"
: "${SUMMARY_OUT_CSV:=}"
: "${RESTRICT_TO_CSV:=1}"
: "${DRY_RUN:=0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERR] PYTHON_BIN not executable: $PYTHON_BIN" >&2
  exit 1
fi

if ! "$PYTHON_BIN" - <<'PY'
import importlib
mods = ["numpy", "matplotlib", "qiskit"]
for m in mods:
    importlib.import_module(m)
PY
then
  echo "[ERR] Missing python deps (need numpy/matplotlib/qiskit) in: $PYTHON_BIN" >&2
  exit 1
fi

csv_base="$(basename "$CSV")"
csv_base="${csv_base%.csv}"
run_tag="util${UTIL}_shots${SHOTS}"

if [[ "$csv_base" == *"$run_tag"* ]]; then
  name_base="$csv_base"
else
  name_base="${csv_base}_${run_tag}"
fi

if [[ -n "$NEW_TARGET" ]]; then
  target_dir="$(dirname "$NEW_TARGET")"
  target_label="$(basename "$NEW_TARGET")"
  target_label="${target_label%.py}"
else
  target_dir="$(dirname "$CSV")"
  target_label="new"
fi

target_label_safe="$(printf '%s' "$target_label" | tr -cs '[:alnum:]_.-' '_')"
target_label_safe="${target_label_safe#_}"
target_label_safe="${target_label_safe%_}"
if [[ -z "$target_label_safe" ]]; then
  target_label_safe="new"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$target_dir"
fi
mkdir -p "$OUTPUT_DIR"

if [[ -z "$OUT" ]]; then
  OUT="$OUTPUT_DIR/${name_base}_qos_vs_${target_label_safe}_rankonly.png"
fi
if [[ -z "$OUT_CSV" ]]; then
  OUT_CSV="$OUTPUT_DIR/${name_base}_qos_vs_${target_label_safe}_rankonly.csv"
fi
if [[ -z "$BAR_OUT" ]]; then
  BAR_OUT="$OUTPUT_DIR/${name_base}_qos_vs_${target_label_safe}_rankonly_utilbin_mean_fid.png"
fi
if [[ -z "$BAR_OUT_CSV" ]]; then
  BAR_OUT_CSV="$OUTPUT_DIR/${name_base}_qos_vs_${target_label_safe}_rankonly_utilbin_mean_fid.csv"
fi
if [[ -z "$SUMMARY_OUT_CSV" ]]; then
  SUMMARY_OUT_CSV="$OUTPUT_DIR/${name_base}_qos_vs_${target_label_safe}_rankonly_avg_rank_summary.csv"
fi

CMD=(
  "$PYTHON_BIN"
  "$SCRIPT_DIR/plot_qos_vs_evolved_from_csv.py"
  "--csv" "$CSV"
  "--util" "$UTIL"
  "--shots" "$SHOTS"
  "--top-k" "$TOP_K"
  "--util-bin-width" "$UTIL_BIN_WIDTH"
)

if [[ -n "$ORIG_TARGET" ]]; then
  CMD+=("--orig-target" "$ORIG_TARGET")
fi
if [[ -n "$NEW_TARGET" ]]; then
  CMD+=("--new-target" "$NEW_TARGET")
fi
if [[ -n "$OUT" ]]; then
  CMD+=("--out" "$OUT")
fi
if [[ -n "$OUT_CSV" ]]; then
  CMD+=("--out-csv" "$OUT_CSV")
fi
if [[ -n "$BAR_OUT" ]]; then
  CMD+=("--bar-out" "$BAR_OUT")
fi
if [[ -n "$BAR_OUT_CSV" ]]; then
  CMD+=("--bar-out-csv" "$BAR_OUT_CSV")
fi
if [[ -n "$SUMMARY_OUT_CSV" ]]; then
  CMD+=("--summary-csv" "$SUMMARY_OUT_CSV")
fi
if [[ "$RESTRICT_TO_CSV" == "1" ]]; then
  CMD+=("--restrict-to-csv")
fi

echo "[INFO] Config: $CFG_FILE"
echo "[INFO] Output dir: $OUTPUT_DIR"
printf '[INFO] Command:'
printf ' %q' "${CMD[@]}"
echo

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[INFO] DRY_RUN=1, command not executed."
  exit 0
fi

"${CMD[@]}"
