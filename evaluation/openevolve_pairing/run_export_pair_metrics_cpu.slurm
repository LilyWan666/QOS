#!/bin/bash
#SBATCH --job-name=pair_export_cpu
#SBATCH --output=/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing/logs/%x.%j.out
#SBATCH --error=/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing/logs/%x.%j.err
#SBATCH --partition=cpu
#SBATCH --account=bgbx-delta-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=36:00:00

set -euo pipefail

ROOT_DIR="/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing"
EXPORT_PY="${ROOT_DIR}/export_pair_metrics.py"
SUMMARIZE_PY="${ROOT_DIR}/summarize_pair_metrics.py"
LOG_DIR="${ROOT_DIR}/logs"

: "${OE_CONDA_ROOT:=/u/lily/miniconda3}"
: "${OE_CONDA_ENV:=qos_fig11}"
: "${OE_PYTHON_BIN:=${OE_CONDA_ROOT}/envs/${OE_CONDA_ENV}/bin/python}"
: "${UTILS:=30 60 88}"
: "${SHOTS_LIST:=1024 8196}"
: "${WORKERS:=1}"
: "${FLUSH_EVERY:=10}"
: "${SAMPLE_K:=0}"
: "${SAMPLE_SEED:=42}"
: "${RECOMPUTE:=0}"
: "${RUN_LABEL:=cpu_bench}"
: "${OUT_BASE:=${ROOT_DIR}/openevolve_output/_pair_metrics_${RUN_LABEL}_${SLURM_JOB_ID}}"

mkdir -p "${LOG_DIR}" "${OUT_BASE}"

export PYTHONPATH="/work/nvme/betu/lily/QOS"

if [[ -f "/u/lily/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "/u/lily/miniconda3/etc/profile.d/conda.sh"
  conda activate "${OE_CONDA_ENV}"
elif [[ -f "${OE_CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${OE_CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${OE_CONDA_ENV}"
fi

PYTHON_BIN="${OE_PYTHON_BIN}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

PAIR_OUT_DIR="${OUT_BASE}/pairing_metadata"
SUMMARY_CSV="${OUT_BASE}/pair_export_resource_summary.csv"
SUMMARY_JSONL="${OUT_BASE}/pair_export_resource_summary.jsonl"
RUN_META_TXT="${OUT_BASE}/job_context.txt"
SACCT_CSV="${OUT_BASE}/sacct_job_summary.csv"

cat > "${RUN_META_TXT}" <<EOF
job_id=${SLURM_JOB_ID:-}
job_name=${SLURM_JOB_NAME:-}
node_list=${SLURM_JOB_NODELIST:-}
submit_dir=${SLURM_SUBMIT_DIR:-}
hostname=$(hostname)
python_bin=${PYTHON_BIN}
cpus_per_task=${SLURM_CPUS_PER_TASK:-}
mem_per_node=${SLURM_MEM_PER_NODE:-}
mem_per_cpu=${SLURM_MEM_PER_CPU:-}
utils=${UTILS}
shots_list=${SHOTS_LIST}
workers=${WORKERS}
sample_k=${SAMPLE_K}
sample_seed=${SAMPLE_SEED}
out_base=${OUT_BASE}
EOF

echo "job_id,job_name,hostname,node_list,util,shots,workers,sample_k,sample_seed,req_cpus,req_mem_mb,start_iso,end_iso,elapsed_sec,user_cpu_sec,sys_cpu_sec,cpu_pct,max_rss_kb,major_page_faults,minor_page_faults,vol_ctx_switches,invol_ctx_switches,exit_code,num_pairs,pairs_per_sec,out_csv_bytes,out_csv,per_pair_jsonl,summary_jsonl,stdout_log,stderr_log,time_tsv" > "${SUMMARY_CSV}"
: > "${SUMMARY_JSONL}"

echo "[INFO] job_id=${SLURM_JOB_ID:-na} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-na} mem_node_mb=${SLURM_MEM_PER_NODE:-na} mem_cpu_mb=${SLURM_MEM_PER_CPU:-na}"
echo "[INFO] utils=${UTILS} shots=${SHOTS_LIST} workers=${WORKERS} out_base=${OUT_BASE}"
echo "[INFO] python_bin=${PYTHON_BIN}"

failures=0
for util in ${UTILS}; do
  for shots in ${SHOTS_LIST}; do
    run_tag="u${util}_s${shots}"
    start_iso="$(date --iso-8601=seconds)"
    stdout_log="${OUT_BASE}/${run_tag}.stdout.log"
    stderr_log="${OUT_BASE}/${run_tag}.stderr.log"
    time_tsv="${OUT_BASE}/${run_tag}.time.tsv"
    per_pair_jsonl=""
    summary_jsonl=""
    if [[ "${SAMPLE_K}" -gt 0 ]]; then
      out_csv="${PAIR_OUT_DIR}/pair_metrics_util${util}_shots${shots}_sample${SAMPLE_K}_seed${SAMPLE_SEED}.csv"
    else
      out_csv="${PAIR_OUT_DIR}/pair_metrics_util${util}_shots${shots}.csv"
    fi
    per_pair_jsonl="${out_csv%.csv}.jsonl"
    summary_jsonl="${out_csv%.csv}_summary.jsonl"

    cmd=(
      "${PYTHON_BIN}" -u "${EXPORT_PY}"
      --util "${util}"
      --shots "${shots}"
      --workers "${WORKERS}"
      --flush-every "${FLUSH_EVERY}"
      --format csv
      --out-dir "${PAIR_OUT_DIR}"
    )
    if [[ "${SAMPLE_K}" -gt 0 ]]; then
      cmd+=(--sample-k "${SAMPLE_K}" --sample-seed "${SAMPLE_SEED}")
    fi
    if [[ "${RECOMPUTE}" == "1" ]]; then
      cmd+=(--recompute)
    fi

    echo "[INFO] start ${run_tag} at ${start_iso}"
    echo "[INFO] command: ${cmd[*]}"

    set +e
    /usr/bin/time \
      -f '%e\t%U\t%S\t%P\t%M\t%F\t%R\t%w\t%c\t%x' \
      -o "${time_tsv}" \
      "${cmd[@]}" \
      >"${stdout_log}" \
      2>"${stderr_log}"
    exit_code=$?
    set -e

    end_iso="$(date --iso-8601=seconds)"
    if [[ -s "${time_tsv}" ]]; then
      IFS=$'\t' read -r elapsed_sec user_cpu_sec sys_cpu_sec cpu_pct max_rss_kb major_pf minor_pf vol_cs invol_cs time_exit_code < "${time_tsv}"
    else
      elapsed_sec=""
      user_cpu_sec=""
      sys_cpu_sec=""
      cpu_pct=""
      max_rss_kb=""
      major_pf=""
      minor_pf=""
      vol_cs=""
      invol_cs=""
      time_exit_code="${exit_code}"
    fi

    if [[ "${exit_code}" -ne 0 ]]; then
      failures=$((failures + 1))
      echo "[WARN] ${run_tag} failed with exit_code=${exit_code}"
    else
      echo "[INFO] ${run_tag} finished: elapsed=${elapsed_sec}s max_rss_kb=${max_rss_kb} cpu_pct=${cpu_pct}"
    fi

    num_pairs=""
    pairs_per_sec=""
    out_csv_bytes=""
    if [[ -f "${out_csv}" ]]; then
      out_csv_bytes="$(stat -c '%s' "${out_csv}")"
      line_count="$(wc -l < "${out_csv}")"
      if [[ "${line_count}" -gt 0 ]]; then
        num_pairs=$((line_count - 1))
      else
        num_pairs=0
      fi
      if [[ -n "${elapsed_sec}" && "${elapsed_sec}" != "0" && -n "${num_pairs}" ]]; then
        pairs_per_sec="$(awk -v n="${num_pairs}" -v t="${elapsed_sec}" 'BEGIN { if (t > 0) printf "%.6f", n / t; }')"
      fi
    fi

    if [[ "${exit_code}" -eq 0 && -f "${out_csv}" ]]; then
      "${PYTHON_BIN}" "${SUMMARIZE_PY}" \
        --csv "${out_csv}" \
        --per-pair-jsonl "${per_pair_jsonl}" \
        --summary-jsonl "${summary_jsonl}" \
        --util "${util}" \
        --shots "${shots}" \
        --workers "${WORKERS}" \
        --sample-k "${SAMPLE_K}" \
        --sample-seed "${SAMPLE_SEED}" \
        --req-cpus "${SLURM_CPUS_PER_TASK:-}" \
        --req-mem-mb "${SLURM_MEM_PER_NODE:-${SLURM_MEM_PER_CPU:-}}" \
        --start-iso "${start_iso}" \
        --end-iso "${end_iso}" \
        --elapsed-sec "${elapsed_sec}" \
        --user-cpu-sec "${user_cpu_sec}" \
        --sys-cpu-sec "${sys_cpu_sec}" \
        --cpu-pct "${cpu_pct}" \
        --max-rss-kb "${max_rss_kb}" \
        --major-page-faults "${major_pf}" \
        --minor-page-faults "${minor_pf}" \
        --vol-ctx-switches "${vol_cs}" \
        --invol-ctx-switches "${invol_cs}" \
        --exit-code "${time_exit_code}" \
        --pairs-per-sec "${pairs_per_sec}" \
        --out-csv-bytes "${out_csv_bytes}" \
        --out-csv "${out_csv}" \
        --stdout-log "${stdout_log}" \
        --stderr-log "${stderr_log}" \
        --time-tsv "${time_tsv}"
    fi

    req_mem_mb="${SLURM_MEM_PER_NODE:-${SLURM_MEM_PER_CPU:-}}"
    printf '%s\n' \
      "${SLURM_JOB_ID:-},${SLURM_JOB_NAME:-},$(hostname),${SLURM_JOB_NODELIST:-},${util},${shots},${WORKERS},${SAMPLE_K},${SAMPLE_SEED},${SLURM_CPUS_PER_TASK:-},${req_mem_mb},${start_iso},${end_iso},${elapsed_sec},${user_cpu_sec},${sys_cpu_sec},${cpu_pct},${max_rss_kb},${major_pf},${minor_pf},${vol_cs},${invol_cs},${time_exit_code},${num_pairs},${pairs_per_sec},${out_csv_bytes},${out_csv},${per_pair_jsonl},${summary_jsonl},${stdout_log},${stderr_log},${time_tsv}" \
      >> "${SUMMARY_CSV}"

    JOB_ID="${SLURM_JOB_ID:-}" \
    JOB_NAME="${SLURM_JOB_NAME:-}" \
    HOSTNAME_VAL="$(hostname)" \
    NODE_LIST="${SLURM_JOB_NODELIST:-}" \
    UTIL_VAL="${util}" \
    SHOTS_VAL="${shots}" \
    WORKERS_VAL="${WORKERS}" \
    SAMPLE_K_VAL="${SAMPLE_K}" \
    SAMPLE_SEED_VAL="${SAMPLE_SEED}" \
    REQ_CPUS_VAL="${SLURM_CPUS_PER_TASK:-}" \
    REQ_MEM_MB_VAL="${req_mem_mb}" \
    START_ISO_VAL="${start_iso}" \
    END_ISO_VAL="${end_iso}" \
    ELAPSED_SEC_VAL="${elapsed_sec}" \
    USER_CPU_SEC_VAL="${user_cpu_sec}" \
    SYS_CPU_SEC_VAL="${sys_cpu_sec}" \
    CPU_PCT_VAL="${cpu_pct}" \
    MAX_RSS_KB_VAL="${max_rss_kb}" \
    MAJOR_PF_VAL="${major_pf}" \
    MINOR_PF_VAL="${minor_pf}" \
    VOL_CS_VAL="${vol_cs}" \
    INVOL_CS_VAL="${invol_cs}" \
    EXIT_CODE_VAL="${time_exit_code}" \
    NUM_PAIRS_VAL="${num_pairs}" \
    PAIRS_PER_SEC_VAL="${pairs_per_sec}" \
    OUT_CSV_BYTES_VAL="${out_csv_bytes}" \
    OUT_CSV_VAL="${out_csv}" \
    PER_PAIR_JSONL_VAL="${per_pair_jsonl}" \
    SUMMARY_JSONL_VAL="${summary_jsonl}" \
    STDOUT_LOG_VAL="${stdout_log}" \
    STDERR_LOG_VAL="${stderr_log}" \
    TIME_TSV_VAL="${time_tsv}" \
    "${PYTHON_BIN}" - <<'PY' >> "${SUMMARY_JSONL}"
import json
import os

def maybe_num(key):
    raw = os.environ.get(key, "")
    if raw == "":
        return None
    try:
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)
    except Exception:
        return raw

record = {
    "job_id": os.environ.get("JOB_ID", ""),
    "job_name": os.environ.get("JOB_NAME", ""),
    "hostname": os.environ.get("HOSTNAME_VAL", ""),
    "node_list": os.environ.get("NODE_LIST", ""),
    "util": maybe_num("UTIL_VAL"),
    "shots": maybe_num("SHOTS_VAL"),
    "workers": maybe_num("WORKERS_VAL"),
    "sample_k": maybe_num("SAMPLE_K_VAL"),
    "sample_seed": maybe_num("SAMPLE_SEED_VAL"),
    "req_cpus": maybe_num("REQ_CPUS_VAL"),
    "req_mem_mb": maybe_num("REQ_MEM_MB_VAL"),
    "start_iso": os.environ.get("START_ISO_VAL", ""),
    "end_iso": os.environ.get("END_ISO_VAL", ""),
    "elapsed_sec": maybe_num("ELAPSED_SEC_VAL"),
    "user_cpu_sec": maybe_num("USER_CPU_SEC_VAL"),
    "sys_cpu_sec": maybe_num("SYS_CPU_SEC_VAL"),
    "cpu_pct": os.environ.get("CPU_PCT_VAL", ""),
    "max_rss_kb": maybe_num("MAX_RSS_KB_VAL"),
    "major_page_faults": maybe_num("MAJOR_PF_VAL"),
    "minor_page_faults": maybe_num("MINOR_PF_VAL"),
    "vol_ctx_switches": maybe_num("VOL_CS_VAL"),
    "invol_ctx_switches": maybe_num("INVOL_CS_VAL"),
    "exit_code": maybe_num("EXIT_CODE_VAL"),
    "num_pairs": maybe_num("NUM_PAIRS_VAL"),
    "pairs_per_sec": maybe_num("PAIRS_PER_SEC_VAL"),
    "out_csv_bytes": maybe_num("OUT_CSV_BYTES_VAL"),
    "out_csv": os.environ.get("OUT_CSV_VAL", ""),
    "per_pair_jsonl": os.environ.get("PER_PAIR_JSONL_VAL", ""),
    "summary_jsonl": os.environ.get("SUMMARY_JSONL_VAL", ""),
    "stdout_log": os.environ.get("STDOUT_LOG_VAL", ""),
    "stderr_log": os.environ.get("STDERR_LOG_VAL", ""),
    "time_tsv": os.environ.get("TIME_TSV_VAL", ""),
}
print(json.dumps(record, ensure_ascii=True))
PY
  done
done

if command -v sacct >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  for _ in 1 2 3 4 5; do
    if sacct -j "${SLURM_JOB_ID}" -X -P --noheader >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done

  sacct \
    -j "${SLURM_JOB_ID}" \
    -X \
    -P \
    --units=K \
    --format=JobIDRaw,JobName,Partition,AllocCPUS,ReqMem,Elapsed,TotalCPU,CPUTime,MaxRSS,AveRSS,State,ExitCode,NodeList \
    > "${SACCT_CSV}" || true
  echo "[INFO] wrote sacct summary to ${SACCT_CSV}"
fi

echo "[INFO] per-run summary: ${SUMMARY_CSV}"
echo "[INFO] per-run jsonl: ${SUMMARY_JSONL}"
echo "[INFO] pair csv output dir: ${PAIR_OUT_DIR}"

if [[ "${failures}" -ne 0 ]]; then
  echo "[ERROR] ${failures} run(s) failed"
  exit 1
fi

echo "[OK] all runs completed"
