#!/bin/bash
#SBATCH --job-name=gemini_evolve_multi
#SBATCH --output=/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing/logs/%x.%j.out
#SBATCH --error=/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing/logs/%x.%j.err
#SBATCH --partition=ghx4
#SBATCH --account=bgbx-dtai-gh
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail

# Gemini mode defaults (can override via sbatch --export=ALL,...)
export OE_LLM_PROVIDER="${OE_LLM_PROVIDER:-gemini}"
export GEMINI_KEYS_DIR="${GEMINI_KEYS_DIR:-/work/nvme/betu/lily/QOS/ibm_quantum/keys}"
export GEMINI_KEY_GLOB="${GEMINI_KEY_GLOB:-gemini-*.key}"
export GEMINI_KEY_COUNT="${GEMINI_KEY_COUNT:-5}"
export GEMINI_DAILY_LIMIT="${GEMINI_DAILY_LIMIT:-0}"
export GEMINI_ITERS_PER_KEY="${GEMINI_ITERS_PER_KEY:-20}"
export GEMINI_MAX_QUERIES_PER_MIN="${GEMINI_MAX_QUERIES_PER_MIN:-5}"
export GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.0-flash}"
export GEMINI_API_BASE="${GEMINI_API_BASE:-https://generativelanguage.googleapis.com/v1beta/openai}"
export GEMINI_TEMPERATURE="${GEMINI_TEMPERATURE:-0.7}"
export GEMINI_TOP_P="${GEMINI_TOP_P:-0.9}"
export GEMINI_REASONING_EFFORT="${GEMINI_REASONING_EFFORT:-low}"
export GEMINI_INCLUDE_ARTIFACTS="${GEMINI_INCLUDE_ARTIFACTS:-1}"
export GEMINI_MAX_ARTIFACT_BYTES="${GEMINI_MAX_ARTIFACT_BYTES:-196608}"

echo "[INFO] CPU Gemini wrapper started"
echo "[INFO] provider=$OE_LLM_PROVIDER keys_dir=$GEMINI_KEYS_DIR key_count=$GEMINI_KEY_COUNT iters_per_key=$GEMINI_ITERS_PER_KEY"
echo "[INFO] quota daily/key=$GEMINI_DAILY_LIMIT qpm/key=$GEMINI_MAX_QUERIES_PER_MIN model=$GEMINI_MODEL"
echo "[INFO] Gemini sampling temperature=$GEMINI_TEMPERATURE top_p=$GEMINI_TOP_P"
echo "[INFO] Gemini reasoning_effort=$GEMINI_REASONING_EFFORT"
echo "[INFO] Gemini include_artifacts=$GEMINI_INCLUDE_ARTIFACTS max_artifact_bytes=$GEMINI_MAX_ARTIFACT_BYTES"

# Reuse the unified runner script (it will branch to Gemini mode and won't start vLLM).
bash /work/nvme/betu/lily/QOS/evaluation/openevolve_pairing/run_vllm_openevolve_qwen3_14b_awq.slurm
