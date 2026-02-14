#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG_FILE="${1:-$SCRIPT_DIR/plot_qos_vs_target_physical.env}"

if [[ ! -f "$CFG_FILE" ]]; then
  echo "[ERR] Config file not found: $CFG_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CFG_FILE"

: "${PYTHON_BIN:=python}"
: "${UTILS:=30 60 88}"
: "${SHOTS:=1000}"
: "${TOP_K:=24}"
: "${TOP_K_RATIO:=0}"
# Strict mode: always run Top-10% and Top-20% into ckpt10/ckpt20.
: "${TOP_K_PERCENT_STEPS:=10,20}"
: "${UTIL_BIN_WIDTH:=0.02}"
: "${ORIG_TARGET:=}"
: "${NEW_TARGET:=}"
: "${CHECKPOINT_STEPS:=}"
: "${CHECKPOINTS_ROOT:=}"
: "${RUN_BASE_TARGET:=0}"
: "${OUTPUT_DIR:=}"
: "${DEFAULT_OUTPUT_DIR:=/work/nvme/betu/lily/QOS/temp/qos_seed_vs_target_physical}"
: "${BAR_OUT:=}"
: "${BAR_OUT_CSV:=}"
: "${SUMMARY_OUT_CSV:=}"
: "${RESTRICT_TO_CSV:=0}"
: "${DRY_RUN:=0}"

# Base: /work/nvme/betu/lily/QOS/ibm_quantum/jobs
: "${PHYSICAL_JOBS_ROOT:=/work/nvme/betu/lily/QOS/ibm_quantum/jobs}"
# Template under PHYSICAL_JOBS_ROOT, must contain {UTIL}; default uses util30/60/88 runs.
: "${PHYSICAL_FIDELITY_SUBDIR_TEMPLATE:=util{UTIL}/fidelity}"

# Always disable evaluator-side CSV pruning for plotting runs.
# Plot selection should be controlled only by Top-K / Top-K ratio.
export OE_RESTRICT_TO_PAIR_CSV=0

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

if [[ -n "$NEW_TARGET" ]]; then
  target_dir="$(dirname "$NEW_TARGET")"
  target_label="$(basename "$NEW_TARGET")"
  target_label="${target_label%.py}"
else
  target_dir="$SCRIPT_DIR"
  target_label="new"
fi

target_label_safe="$(printf '%s' "$target_label" | tr -cs '[:alnum:]_.-' '_')"
target_label_safe="${target_label_safe#_}"
target_label_safe="${target_label_safe%_}"
if [[ -z "$target_label_safe" ]]; then
  target_label_safe="new"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  # For physical comparisons, default to the best_program directory.
  OUTPUT_DIR="$target_dir"
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"

echo "[INFO] Config: $CFG_FILE"
echo "[INFO] Output dir: $OUTPUT_DIR"
echo "[INFO] Utils: $UTILS"
echo "[INFO] Physical jobs root: $PHYSICAL_JOBS_ROOT"
echo "[INFO] Checkpoint steps: ${CHECKPOINT_STEPS:-<none>}"
echo "[INFO] Run base target: $RUN_BASE_TARGET"
echo "[INFO] Top-K percent steps: $TOP_K_PERCENT_STEPS"

declare -a RUN_TARGET_PATHS=()
declare -a RUN_TARGET_LABELS=()
declare -a RUN_OUTPUT_DIRS=()
declare -a RUN_TOP_KS=()
declare -a RUN_TOP_K_RATIOS=()

add_run_target() {
  local target_path="$1"
  local target_label_local="$2"
  local target_output_dir="$3"
  local target_top_k="${4:-$TOP_K}"
  local target_top_k_ratio="${5:-$TOP_K_RATIO}"
  RUN_TARGET_PATHS+=("$target_path")
  RUN_TARGET_LABELS+=("$target_label_local")
  RUN_OUTPUT_DIRS+=("$target_output_dir")
  RUN_TOP_KS+=("$target_top_k")
  RUN_TOP_K_RATIOS+=("$target_top_k_ratio")
}

if [[ -z "$NEW_TARGET" ]]; then
  echo "[ERR] NEW_TARGET is required." >&2
  exit 1
fi

if [[ "$CHECKPOINT_STEPS" != "" ]]; then
  echo "[ERR] CHECKPOINT_STEPS is disabled. This script only supports Top10%/Top20% mode." >&2
  exit 1
fi

if [[ "$RUN_BASE_TARGET" != "0" ]]; then
  echo "[ERR] RUN_BASE_TARGET is disabled. Outputs are always ckpt10/ckpt20 only." >&2
  exit 1
fi

if [[ "$TOP_K_RATIO" != "0" && "$TOP_K_RATIO" != "0.0" ]]; then
  echo "[ERR] TOP_K_RATIO is disabled. Ratios are fixed to 10% and 20%." >&2
  exit 1
fi

# Accept only 10,20 (order-insensitive).
pct_tokens="${TOP_K_PERCENT_STEPS//,/ }"
pct_tokens="${pct_tokens//:/ }"
pct_tokens="${pct_tokens//;/ }"
declare -A pct_seen=()
for pct in $pct_tokens; do
  pct_num="${pct//[^0-9]/}"
  if [[ -n "$pct_num" ]]; then
    pct_seen["$pct_num"]=1
  fi
done
if [[ "${#pct_seen[@]}" -ne 2 || -z "${pct_seen[10]+x}" || -z "${pct_seen[20]+x}" ]]; then
  echo "[ERR] TOP_K_PERCENT_STEPS must be exactly 10,20 (order-insensitive)." >&2
  exit 1
fi

# Strict default: always produce ckpt10 and ckpt20 folders.
for pct_num in 10 20; do
  pct_ratio="$(awk "BEGIN { printf \"%.6f\", ${pct_num} / 100.0 }")"
  add_run_target "$NEW_TARGET" "ckpt${pct_num}" "$OUTPUT_DIR/ckpt${pct_num}" "$TOP_K" "$pct_ratio"
done

for idx in "${!RUN_TARGET_PATHS[@]}"; do
  RUN_NEW_TARGET="${RUN_TARGET_PATHS[$idx]}"
  RUN_TARGET_LABEL="${RUN_TARGET_LABELS[$idx]}"
  RUN_OUTPUT_DIR="${RUN_OUTPUT_DIRS[$idx]}"
  RUN_TOP_K="${RUN_TOP_KS[$idx]}"
  RUN_TOP_K_RATIO="${RUN_TOP_K_RATIOS[$idx]}"
  mkdir -p "$RUN_OUTPUT_DIR"

  echo
  echo "[INFO] ===== target=${RUN_TARGET_LABEL} ====="
  echo "[INFO] target path: $RUN_NEW_TARGET"
  echo "[INFO] target output dir: $RUN_OUTPUT_DIR"
  echo "[INFO] selection: top_k=${RUN_TOP_K} top_k_ratio=${RUN_TOP_K_RATIO}"

  for UTIL in $UTILS; do
    run_tag="util${UTIL}_shots${SHOTS}"
    phys_csv="$RUN_OUTPUT_DIR/pair_metrics_${run_tag}_physical.csv"

    fidelity_subdir="${PHYSICAL_FIDELITY_SUBDIR_TEMPLATE//\{UTIL\}/$UTIL}"
    if [[ "$fidelity_subdir" = /* ]]; then
      fidelity_dir="$fidelity_subdir"
    else
      fidelity_dir="$PHYSICAL_JOBS_ROOT/$fidelity_subdir"
    fi

    out_png="$RUN_OUTPUT_DIR/pair_metrics_${run_tag}_qos_vs_${RUN_TARGET_LABEL}_physical_rankonly.png"
    out_csv="$RUN_OUTPUT_DIR/pair_metrics_${run_tag}_qos_vs_${RUN_TARGET_LABEL}_physical_rankonly.csv"
    bar_png="$RUN_OUTPUT_DIR/pair_metrics_${run_tag}_qos_vs_${RUN_TARGET_LABEL}_physical_rankonly_utilbin_mean_fid.png"
    bar_csv="$RUN_OUTPUT_DIR/pair_metrics_${run_tag}_qos_vs_${RUN_TARGET_LABEL}_physical_rankonly_utilbin_mean_fid.csv"
    summary_csv="$RUN_OUTPUT_DIR/pair_metrics_${run_tag}_qos_vs_${RUN_TARGET_LABEL}_physical_rankonly_avg_rank_summary.csv"

    BUILD_CMD=(
      "$PYTHON_BIN"
      "$SCRIPT_DIR/build_physical_pair_csv.py"
      "--fidelity-dir" "$fidelity_dir"
      "--out-csv" "$phys_csv"
    )

    PLOT_CMD=(
      "$PYTHON_BIN"
      "$SCRIPT_DIR/plot_qos_vs_evolved_from_csv.py"
      "--csv" "$phys_csv"
      "--util" "$UTIL"
      "--shots" "$SHOTS"
      "--top-k" "$RUN_TOP_K"
      "--util-bin-width" "$UTIL_BIN_WIDTH"
      "--out" "$out_png"
      "--out-csv" "$out_csv"
      "--bar-out" "$bar_png"
      "--bar-out-csv" "$bar_csv"
      "--summary-csv" "$summary_csv"
      "--title-tag" "PHYSICAL"
    )

    if [[ -n "$ORIG_TARGET" ]]; then
      PLOT_CMD+=("--orig-target" "$ORIG_TARGET")
    fi
    if [[ -n "$RUN_NEW_TARGET" ]]; then
      PLOT_CMD+=("--new-target" "$RUN_NEW_TARGET")
    fi
    if [[ -n "$RUN_TOP_K_RATIO" && "$RUN_TOP_K_RATIO" != "0" && "$RUN_TOP_K_RATIO" != "0.0" ]]; then
      PLOT_CMD+=("--top-k-ratio" "$RUN_TOP_K_RATIO")
    fi
    if [[ "$RESTRICT_TO_CSV" == "1" ]]; then
      PLOT_CMD+=("--restrict-to-csv")
    fi

    printf '\n[INFO][target=%s][util=%s] Build CSV:' "$RUN_TARGET_LABEL" "$UTIL"
    printf ' %q' "${BUILD_CMD[@]}"
    echo
    printf '[INFO][target=%s][util=%s] Plot:' "$RUN_TARGET_LABEL" "$UTIL"
    printf ' %q' "${PLOT_CMD[@]}"
    echo

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[INFO][target=$RUN_TARGET_LABEL][util=$UTIL] DRY_RUN=1, skip execution."
      continue
    fi

    "${BUILD_CMD[@]}"
    "${PLOT_CMD[@]}"
  done
done
