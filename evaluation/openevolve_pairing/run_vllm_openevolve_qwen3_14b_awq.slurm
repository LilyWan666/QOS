#!/bin/bash
#SBATCH --job-name=vllm_qwen2_5_14b_inst
#SBATCH --output=/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing/logs/%x.%j.out
#SBATCH --error=/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing/logs/%x.%j.err
#SBATCH --partition=ghx4
#SBATCH --account=bgbx-dtai-gh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail

MODEL_DIR="/projects/betu/llm_models/Qwen2.5-14B-Instruct"
VLLM_IMG="/work/nvme/betu/openrtlset2/env/apptainer_images/vllm_gh200_v0.11.0.sif"
WAIT_SCRIPT="/work/nvme/betu/openrtlset2/dataset_gen/llm_inferences/src/gh200-utils/wait_for_vllm.py"
EVO_CFG="/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing/openevolve_gpt.yaml"
TARGET_PY="/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing/target.py"
EVAL_PY="/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing/evaluator.py"

export OPENAI_API_KEY="token-abc123"
export PYTHONPATH="/work/nvme/betu/lily/QOS"
export ENABLE_ARTIFACTS="true"

source /work/nvme/becn/lily/miniconda3/etc/profile.d/conda.sh
conda activate qos_fig11

mkdir -p /work/nvme/betu/lily/QOS/evaluation/openevolve_pairing/logs
cd /work/nvme/betu/lily/QOS/evaluation/openevolve_pairing

# Build per-run output directory and config.
MODEL_NAME="Qwen2.5-14B-Instruct"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
ITERATIONS="$(/work/nvme/becn/lily/miniconda3/envs/qos_fig11/bin/python -c "import yaml; cfg_path='/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing/openevolve_gpt.yaml'; cfg=yaml.safe_load(open(cfg_path,'r',encoding='utf-8')); print(cfg.get('max_iterations',0))")"
OUT_DIR="/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing/openevolve_output/${MODEL_NAME}_iter${ITERATIONS}_${RUN_TS}"
mkdir -p "$OUT_DIR"

OUT_CFG="${OUT_DIR}/openevolve_gpt.yaml"
export OUT_DIR
export OUT_CFG
/work/nvme/becn/lily/miniconda3/envs/qos_fig11/bin/python - <<'PY'
import os
import yaml
from pathlib import Path

cfg_path = Path('/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing/openevolve_gpt.yaml')
cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))

cfg.setdefault('prompt', {})

out_dir = Path(os.environ['OUT_DIR'])
out_cfg = Path(os.environ['OUT_CFG'])
(out_dir / 'logs').mkdir(parents=True, exist_ok=True)
cfg['log_dir'] = str(out_dir / 'logs')
cfg.setdefault('database', {})['db_path'] = str(out_dir / 'db.sqlite')
cfg['database']['artifacts_base_path'] = str(out_dir / 'artifacts')
cfg['database']['max_snapshot_artifacts'] = None
cfg.setdefault('evolution_trace', {})['enabled'] = True
cfg['evolution_trace']['include_prompts'] = True
cfg['evolution_trace']['output_path'] = str(out_dir / 'evolution_trace.jsonl')

out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
PY

apptainer run --nv --bind /projects,/work,/u "$VLLM_IMG" \
  vllm serve "$MODEL_DIR" \
  --served-model-name Qwen2.5-14B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --api-key "$OPENAI_API_KEY" \
  --dtype auto \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 > vllm_serve.log 2>&1 &
VLLM_PID=$!

python3 -u "$WAIT_SCRIPT" --vllm_host 127.0.0.1 --vllm_port 8000

openevolve-run -c "$OUT_CFG" -o "$OUT_DIR" "$TARGET_PY" "$EVAL_PY"

kill "$VLLM_PID"
wait "$VLLM_PID" || true
