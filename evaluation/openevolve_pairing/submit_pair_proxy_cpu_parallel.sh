#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing"
SLURM_SCRIPT="${ROOT_DIR}/run_export_pair_proxy_cpu.slurm"

: "${UTILS:=30 60 88}"
: "${SHOTS_LIST:=1024 8196}"
: "${PARTITION:=cpu}"
: "${ACCOUNT:=bgbx-delta-cpu}"
: "${CPUS:=2}"
: "${MEM:=128G}"
: "${TIME_LIMIT:=36:00:00}"
: "${PROXY_FEATURE:=depth_ratio}"
: "${FLUSH_EVERY:=1}"
: "${SAMPLE_K:=0}"
: "${SAMPLE_SEED:=42}"
: "${RECOMPUTE:=0}"

if [[ ! -f "${SLURM_SCRIPT}" ]]; then
  echo "[ERROR] Missing slurm script: ${SLURM_SCRIPT}" >&2
  exit 2
fi

ts="$(date +%Y%m%d_%H%M%S)"

echo "[INFO] submitting pair-proxy CPU jobs in parallel"
echo "[INFO] partition=${PARTITION} account=${ACCOUNT}"
echo "[INFO] cpus=${CPUS} mem=${MEM} time=${TIME_LIMIT} proxy=${PROXY_FEATURE}"
echo "[INFO] utils=${UTILS}"
echo "[INFO] shots=${SHOTS_LIST}"

for util in ${UTILS}; do
  for shots in ${SHOTS_LIST}; do
    job_name="proxy_${PROXY_FEATURE}_u${util}_s${shots}"
    run_label="u${util}_s${shots}_${PROXY_FEATURE}_${ts}"
    out_base="${ROOT_DIR}/openevolve_output/_pair_proxy_${run_label}"

    submit_out="$(
      sbatch \
        --job-name="${job_name}" \
        --partition="${PARTITION}" \
        --account="${ACCOUNT}" \
        --cpus-per-task="${CPUS}" \
        --mem="${MEM}" \
        --time="${TIME_LIMIT}" \
        --export=ALL,UTILS="${util}",SHOTS_LIST="${shots}",PROXY_FEATURE="${PROXY_FEATURE}",FLUSH_EVERY="${FLUSH_EVERY}",SAMPLE_K="${SAMPLE_K}",SAMPLE_SEED="${SAMPLE_SEED}",RECOMPUTE="${RECOMPUTE}",RUN_LABEL="${run_label}",OUT_BASE="${out_base}" \
        "${SLURM_SCRIPT}"
    )"
    echo "[INFO] ${submit_out}"
  done
done
