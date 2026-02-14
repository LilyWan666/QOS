#!/bin/bash
set -euo pipefail
export OE_LLM_PROVIDER="${OE_LLM_PROVIDER:-gemini}"
export GEMINI_KEY_COUNT="${GEMINI_KEY_COUNT:-5}"
export GEMINI_ITERS_PER_KEY="${GEMINI_ITERS_PER_KEY:-20}"
export GEMINI_DAILY_LIMIT="${GEMINI_DAILY_LIMIT:-20}"
export GEMINI_MAX_QUERIES_PER_MIN="${GEMINI_MAX_QUERIES_PER_MIN:-5}"
export GEMINI_PREFLIGHT_CHECK="${GEMINI_PREFLIGHT_CHECK:-1}"
export OE_CONDA_ROOT="${OE_CONDA_ROOT:-/work/nvme/becn/lily/miniconda3}"
export OE_CONDA_ENV="${OE_CONDA_ENV:-qos_fig11}"
export OE_PYTHON_BIN="${OE_PYTHON_BIN:-}"
export OE_EVAL_OBJECTIVE="${OE_EVAL_OBJECTIVE:-avg_rank}"
export OE_EVAL_UTILS="${OE_EVAL_UTILS:-30,60,88}"
export OE_EVAL_SHOTS="${OE_EVAL_SHOTS:-1000}"
export OE_RESTRICT_TO_PAIR_CSV="${OE_RESTRICT_TO_PAIR_CSV:-1}"
export OE_TOP_K_RATIO="${OE_TOP_K_RATIO:-}"
export OE_MULTI_UTIL_AGG="${OE_MULTI_UTIL_AGG:-mean}"
exec bash /work/nvme/betu/lily/QOS/evaluation/openevolve_pairing/run_vllm_openevolve_qwen3_14b_awq.slurm
