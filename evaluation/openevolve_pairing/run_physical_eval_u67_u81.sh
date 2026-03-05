#!/usr/bin/env bash
# Run physical (shots 8192) CSV evaluation for util67 and util81 best programs.
# Uses existing physical fidelity data under PHYSICAL_JOBS_ROOT (util30/60/88).
# Output CSVs: *_physical_rankonly_avg_rank_summary.csv, *_physical_rankonly_utilbin_mean_fid.csv
# in OUTPUT_DIR/ckpt10 and OUTPUT_DIR/ckpt20 per env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[INFO] === Util67 best: physical eval (UTILS=30 60 88, SHOTS=8196) ==="
bash run_plot_qos_vs_target_physical.sh plot_physical_u67_best.env

echo "[INFO] === Util81 best: physical eval ==="
source plot_physical_u81_best.env
if [[ ! -f "$NEW_TARGET" ]] || [[ "$NEW_TARGET" == *"REPLACE_WITH"* ]]; then
  echo "[WARN] Util81 best program not set or missing. Edit plot_physical_u81_best.env: set NEW_TARGET to best_program.py path (e.g. openevolve_output/Qwen2.5-14B-Instruct_iter200_*_util81_shots1000_proxy_depth_ratio_*/best/best_program.py), then run:"
  echo "  bash run_plot_qos_vs_target_physical.sh plot_physical_u81_best.env"
  exit 0
fi
bash run_plot_qos_vs_target_physical.sh plot_physical_u81_best.env

echo "[INFO] Done. Check phys_eval_u67 and phys_eval_u81 under _proxy_depthratio_singleutil_physical_306088_8196_20260220/ for *_physical_rankonly_*.csv"
