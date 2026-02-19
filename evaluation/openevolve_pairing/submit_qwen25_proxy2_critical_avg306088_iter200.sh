#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing"
SLURM_SCRIPT="${ROOT_DIR}/run_vllm_openevolve_qwen3_14b_awq.slurm"

if [[ ! -f "${SLURM_SCRIPT}" ]]; then
  echo "[ERROR] Missing slurm script: ${SLURM_SCRIPT}" >&2
  exit 2
fi

FEATURES=(
  "critical_depth_avg"
  "critical_depth_2"
)

for feature in "${FEATURES[@]}"; do
  ts="$(date +%Y%m%d_%H%M%S)"
  feature_tag="$(printf '%s' "${feature}" | tr -cs '[:alnum:]_-' '_')"
  job_name="qwen25_pxy_${feature_tag}_i200"
  run_tag="avg306088_proxy_${feature_tag}_iter200_${ts}"
  echo "[INFO] submitting ${job_name} feature=${feature}"
  sbatch \
    --job-name="${job_name}" \
    --export=ALL,OE_LLM_PROVIDER=qwen,OE_EVAL_UTILS=30:60:88,OE_MULTI_UTIL_AGG=mean,OE_PARETO_SECOND_METRIC=proxy,OE_PROXY_FEATURE="${feature}",OE_MAX_ITERATIONS=200,OE_RUN_TAG="${run_tag}" \
    "${SLURM_SCRIPT}"
done
