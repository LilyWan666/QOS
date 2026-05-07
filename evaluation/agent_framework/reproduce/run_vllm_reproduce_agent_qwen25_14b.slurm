#!/bin/bash
#SBATCH --job-name=repro_qwen25_14b
#SBATCH --output=/work/nvme/betu/lily/QOS-agent/temp/agent_framework/reproduce_agent/slurm_logs/%x.%j.out
#SBATCH --error=/work/nvme/betu/lily/QOS-agent/temp/agent_framework/reproduce_agent/slurm_logs/%x.%j.err
#SBATCH --partition=ghx4
#SBATCH --account=bgbx-dtai-gh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

REPO_ROOT="/work/nvme/betu/lily/QOS-agent"
REPRO_DIR="${REPO_ROOT}/evaluation/agent_framework/reproduce"
RUST_ROOT="${REPO_ROOT}/claw-code/rust"
MODEL_DIR="/projects/betu/llm_models/Qwen2.5-14B-Instruct"
VLLM_IMG="/work/nvme/betu/openrtlset2/env/apptainer_images/vllm_gh200_v0.11.0.sif"
WAIT_SCRIPT="/work/nvme/betu/openrtlset2/dataset_gen/llm_inferences/src/gh200-utils/wait_for_vllm.py"

: "${REPRO_CONDA_ROOT:=/work/nvme/becn/lily/miniconda3}"
: "${REPRO_CONDA_ENV:=qos_fig11}"
: "${REPRO_PYTHON_BIN:=${REPRO_CONDA_ROOT}/envs/${REPRO_CONDA_ENV}/bin/python}"
: "${REPRO_ISOLATE_CONTROLLER_ENV:=1}"
: "${REPRO_CONTROLLER_BASE_PYTHON:=python3}"
: "${REPRO_RECIPE:=${REPRO_DIR}/examples/qos_run_process_qernels_smoke.json}"
: "${REPRO_OUTPUT_ROOT:=${REPO_ROOT}/temp/agent_framework/reproduce_agent/runs}"
: "${REPRO_AGENT_MODEL:=Qwen2.5-14B-Instruct}"
: "${REPRO_AGENT_API_KEY:=token-abc123}"
: "${REPRO_VLLM_HOST:=127.0.0.1}"
: "${REPRO_VLLM_PORT:=8000}"
: "${REPRO_MAX_AGENT_STEPS:=8}"
: "${REPRO_AGENT_EXIT_POLICY:=lenient}"
: "${REPRO_ALLOW_RUN_ON_PREFLIGHT_FAILURE:=0}"
: "${REPRO_GPU_MEMORY_UTILIZATION:=0.85}"
: "${REPRO_MAX_MODEL_LEN:=32768}"

REPRO_AGENT_API_BASE="http://${REPRO_VLLM_HOST}:${REPRO_VLLM_PORT}/v1"

mkdir -p "${REPO_ROOT}/temp/agent_framework/reproduce_agent/slurm_logs"

port_in_use() {
  local port="$1"
  ss -ltn "( sport = :${port} )" | grep -q ":${port}"
}

pick_free_port() {
  local start="$1"
  local end="$2"
  local p
  for ((p=start; p<=end; p++)); do
    if ! port_in_use "${p}"; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

if port_in_use "${REPRO_VLLM_PORT}"; then
  echo "[WARN] requested port ${REPRO_VLLM_PORT} already in use; searching fallback port"
  fallback_port="$(pick_free_port 18000 18999 || true)"
  if [[ -z "${fallback_port:-}" ]]; then
    echo "[ERROR] failed to find a free fallback port in [18000,18999]"
    exit 1
  fi
  REPRO_VLLM_PORT="${fallback_port}"
fi
REPRO_AGENT_API_BASE="http://${REPRO_VLLM_HOST}:${REPRO_VLLM_PORT}/v1"

if [[ "${REPRO_ISOLATE_CONTROLLER_ENV}" == "1" ]]; then
  CONTROLLER_ENV_ROOT="${REPO_ROOT}/temp/agent_framework/reproduce_agent/controller_envs/${SLURM_JOB_ID}"
  CONTROLLER_VENV_DIR="${CONTROLLER_ENV_ROOT}/venv"
  rm -rf "${CONTROLLER_ENV_ROOT}"
  mkdir -p "${CONTROLLER_ENV_ROOT}"
  BASE_PY="$(command -v "${REPRO_CONTROLLER_BASE_PYTHON}" || true)"
  if [[ -z "${BASE_PY}" ]]; then
    BASE_PY="$(command -v python3 || true)"
  fi
  if [[ -z "${BASE_PY}" ]]; then
    echo "[ERROR] unable to find base python for isolated controller env"
    exit 1
  fi
  "${BASE_PY}" -m venv "${CONTROLLER_VENV_DIR}"
  REPRO_PYTHON_BIN="${CONTROLLER_VENV_DIR}/bin/python"
else
  source "${REPRO_CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${REPRO_CONDA_ENV}"
  if [[ ! -x "${REPRO_PYTHON_BIN}" ]]; then
    REPRO_PYTHON_BIN="$(command -v python)"
  fi
fi

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "${VLLM_PID}" >/dev/null 2>&1 || true
    wait "${VLLM_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

cd "${REPO_ROOT}"

echo "[INFO] repo_root=${REPO_ROOT}"
echo "[INFO] recipe=${REPRO_RECIPE}"
echo "[INFO] model=${REPRO_AGENT_MODEL}"
echo "[INFO] api_base=${REPRO_AGENT_API_BASE}"
echo "[INFO] python=${REPRO_PYTHON_BIN}"
echo "[INFO] isolate_controller_env=${REPRO_ISOLATE_CONTROLLER_ENV}"

apptainer run --nv --bind /projects,/work,/u "${VLLM_IMG}" \
  vllm serve "${MODEL_DIR}" \
  --served-model-name "${REPRO_AGENT_MODEL}" \
  --host 0.0.0.0 \
  --port "${REPRO_VLLM_PORT}" \
  --api-key "${REPRO_AGENT_API_KEY}" \
  --dtype auto \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${REPRO_GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${REPRO_MAX_MODEL_LEN}" \
  > "${REPO_ROOT}/temp/agent_framework/reproduce_agent/slurm_logs/vllm_reproduce_agent.${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

python3 -u "${WAIT_SCRIPT}" --vllm_host "${REPRO_VLLM_HOST}" --vllm_port "${REPRO_VLLM_PORT}"

cd "${RUST_ROOT}"
cmd=(
  cargo run -q -p rusty-claude-cli -- reproduce --tool-loop
  --recipe "${REPRO_RECIPE}"
  --output-root "${REPRO_OUTPUT_ROOT}"
  --python-executable "${REPRO_PYTHON_BIN}"
  --max-agent-steps "${REPRO_MAX_AGENT_STEPS}"
  --exit-policy "${REPRO_AGENT_EXIT_POLICY}"
  --model "${REPRO_AGENT_MODEL}"
  --api-base "${REPRO_AGENT_API_BASE}"
  --api-key "${REPRO_AGENT_API_KEY}"
)

echo "[INFO] launching reproduce tool-loop"
printf '[INFO] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
