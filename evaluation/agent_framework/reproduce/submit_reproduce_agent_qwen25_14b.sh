#!/bin/bash
set -euo pipefail

REPO_ROOT="/work/nvme/betu/lily/QOS-agent"
SLURM_SCRIPT="${REPO_ROOT}/evaluation/agent_framework/reproduce/run_vllm_reproduce_agent_qwen25_14b.slurm"
DEFAULT_RECIPE="${REPO_ROOT}/evaluation/agent_framework/reproduce/examples/qos_run_process_qernels_smoke.json"

recipe_arg="${1:-${DEFAULT_RECIPE}}"
if [[ "${recipe_arg}" != /* ]]; then
  recipe_arg="${REPO_ROOT}/${recipe_arg}"
fi

if [[ ! -f "${recipe_arg}" ]]; then
  echo "[ERROR] recipe not found: ${recipe_arg}" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/temp/agent_framework/reproduce_agent/slurm_logs"

echo "[INFO] submitting reproduce tool-loop job"
echo "[INFO] recipe=${recipe_arg}"

sbatch --export=ALL,REPRO_RECIPE="${recipe_arg}" "${SLURM_SCRIPT}"
