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
export GEMINI_DAILY_LIMIT="${GEMINI_DAILY_LIMIT:-20}"
export GEMINI_ITERS_PER_KEY="${GEMINI_ITERS_PER_KEY:-20}"
export GEMINI_MAX_QUERIES_PER_MIN="${GEMINI_MAX_QUERIES_PER_MIN:-5}"
export GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.0-flash}"
export GEMINI_API_BASE="${GEMINI_API_BASE:-https://generativelanguage.googleapis.com/v1beta/openai}"

echo "[INFO] CPU Gemini wrapper started"
echo "[INFO] provider=$OE_LLM_PROVIDER keys_dir=$GEMINI_KEYS_DIR key_count=$GEMINI_KEY_COUNT iters_per_key=$GEMINI_ITERS_PER_KEY"
echo "[INFO] quota daily/key=$GEMINI_DAILY_LIMIT qpm/key=$GEMINI_MAX_QUERIES_PER_MIN model=$GEMINI_MODEL"

# Reuse the unified runner script (it will branch to Gemini mode and won't start vLLM).
bash /work/nvme/betu/lily/QOS/evaluation/openevolve_pairing/run_vllm_openevolve_qwen3_14b_awq.slurm

