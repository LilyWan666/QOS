#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing"
SLURM_SCRIPT="${ROOT_DIR}/run_vllm_openevolve_qwen3_14b_awq.slurm"
PHYS_CSV_DIR="${ROOT_DIR}/openevolve_output/_compare_u60_ckpt480_physical"
CSV_TEMPLATE='pair_metrics_util{util}_shots{shots}_physical.csv'

if [[ ! -f "${SLURM_SCRIPT}" ]]; then
  echo "[ERROR] Missing slurm script: ${SLURM_SCRIPT}" >&2
  exit 2
fi
if [[ ! -d "${PHYS_CSV_DIR}" ]]; then
  echo "[ERROR] Missing physical CSV dir: ${PHYS_CSV_DIR}" >&2
  exit 2
fi

ts="$(date +%Y%m%d_%H%M%S)"
job_name="qwen25_fid_phys_i200"
run_tag="avg306088_fidelity_physical8196_iter200_${ts}"

echo "[INFO] submitting ${job_name}"
echo "[INFO] physical csv dir: ${PHYS_CSV_DIR}"

sbatch \
  --job-name="${job_name}" \
  --export=ALL,OE_LLM_PROVIDER=qwen,OE_EVAL_OBJECTIVE=avg_rank,OE_EVAL_UTILS=30:60:88,OE_EVAL_SHOTS=8196,OE_MULTI_UTIL_AGG=mean,OE_RESTRICT_TO_PAIR_CSV=1,OE_PARETO_SECOND_METRIC=fidelity,OE_PAIR_METRICS_DIR="${PHYS_CSV_DIR}",OE_PAIR_METRICS_TEMPLATE="${CSV_TEMPLATE}",OE_MAX_ITERATIONS=200,OE_RUN_TAG="${run_tag}" \
  "${SLURM_SCRIPT}"

