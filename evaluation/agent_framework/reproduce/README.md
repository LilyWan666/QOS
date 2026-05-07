# Agent Reproduce

`agent_framework/reproduce` is a repo-agnostic reproduction framework for agent workflows.

Its job is to provide one stable contract:

- load a recipe
- run preflight checks
- execute the recipe
- parse metrics
- classify success/failure
- write standardized artifacts for downstream steps

This framework is intentionally generic. Repo-specific logic belongs in recipes
and optional harnesses, not in the core runner.

## Core Runner

```bash
cd /work/nvme/betu/lily/QOS-agent
python evaluation/agent_framework/reproduce/run_reproduce.py \
  --recipe evaluation/agent_framework/reproduce/recipes/python_import_smoke.sample.json
```

Preflight only:

```bash
python evaluation/agent_framework/reproduce/run_reproduce.py \
  --recipe evaluation/agent_framework/reproduce/recipes/python_import_smoke.sample.json \
  --preflight-only
```

With an explicit Python interpreter:

```bash
python evaluation/agent_framework/reproduce/run_reproduce.py \
  --recipe evaluation/agent_framework/reproduce/recipes/python_import_smoke.sample.json \
  --python-executable /path/to/python
```

## Figure Claim Verification

Recipes can optionally include a `verification` block with a `figure_contract`.
When enabled, the runner evaluates claim checks against parsed metrics and writes
`figure_verdict.json`.

Standalone verifier:

```bash
cd /work/nvme/betu/lily/QOS-agent
python evaluation/agent_framework/reproduce/figure/verify_claim.py \
  --contract evaluation/agent_framework/reproduce/examples/figure_contracts/qos_fig11_smoke.contract.json \
  --metrics temp/agent_framework/reproduce_agent/runs/<run_id>/attempts/attempt_1/metrics.json \
  --print-json
```

## Figure Inventory (Full-Figure Planning)

Generate a paper+repo inventory manifest to drive all-figure recipe authoring:

```bash
cd /work/nvme/betu/lily/QOS-agent
python evaluation/agent_framework/reproduce/tools/tool_figure_inventory.py \
  --repo-root /work/nvme/betu/lily/QOS-agent \
  --paper paper/qos.pdf \
  --output temp/agent_framework/reproduce/figure_inventory/qos_figures_manifest.json
```

The manifest groups each paper figure mention (`figX`) with candidate repo paths
(contracts, python runners, scripts) and assigns a heuristic `inventory_status`.

Generate per-figure recipes from that manifest:

```bash
cd /work/nvme/betu/lily/QOS-agent
python evaluation/agent_framework/reproduce/tools/tool_generate_recipes_from_inventory.py \
  --manifest temp/agent_framework/reproduce/figure_inventory/qos_figures_manifest.json \
  --output-dir temp/agent_framework/reproduce/figure_recipes
```

Batch execute generated recipes:

```bash
cd /work/nvme/betu/lily/QOS-agent
python evaluation/agent_framework/reproduce/tools/tool_batch_reproduce_from_index.py \
  --index temp/agent_framework/reproduce/figure_recipes/recipes.index.json \
  --repo-root /work/nvme/betu/lily/QOS-agent \
  --mode core \
  --continue-on-failure
```

Build figure-to-code mapping (with motivation/result split):

```bash
cd /work/nvme/betu/lily/QOS-agent
python evaluation/agent_framework/reproduce/tools/tool_figure_code_map.py \
  --manifest temp/agent_framework/reproduce/figure_inventory/qos_figures_manifest.json \
  --output temp/agent_framework/reproduce/figure_inventory/qos_figure_code_map.json
```

Probe dependency/version hints from repo files:

```bash
cd /work/nvme/betu/lily/QOS-agent
python evaluation/agent_framework/reproduce/tools/tool_dependency_probe.py \
  --repo-root /work/nvme/betu/lily/QOS-agent \
  --output temp/agent_framework/reproduce/deps/dependency_probe.json
```

Normalize a batch report into required/coverage KPIs:

```bash
cd /work/nvme/betu/lily/QOS-agent
python evaluation/agent_framework/reproduce/tools/tool_report_normalize.py \
  --batch-summary temp/agent_framework/reproduce/figure_batch_runs/<batch_id>/batch_summary.json \
  --figure-code-map temp/agent_framework/reproduce/figure_inventory/qos_figure_code_map.json \
  --output temp/agent_framework/reproduce/figure_batch_runs/<batch_id>/normalized_report.json
```

## Environment Diagnosis

```bash
python evaluation/agent_framework/reproduce/env/diagnose_repro_env.py \
  --recipe evaluation/agent_framework/reproduce/recipes/python_import_smoke.sample.json
```

## LLM Reproduce Agent

Use the OpenClaw native tool-loop entrypoint for all LLM-driven reproduce runs:

```bash
cd /work/nvme/betu/lily/QOS-agent/claw-code/rust
cargo run -q -p rusty-claude-cli -- reproduce --tool-loop \
  --recipe /work/nvme/betu/lily/QOS-agent/evaluation/agent_framework/reproduce/examples/qos_run_process_qernels_smoke.json \
  --output-root /work/nvme/betu/lily/QOS-agent/temp/agent_framework/reproduce_agent/runs \
  --model Qwen2.5-14B-Instruct \
  --api-base http://localhost:8000/v1 \
  --api-key EMPTY \
  --exit-policy strict
```

Legacy note: direct `python .../run_reproduce_agent.py` is now a compatibility shim
that redirects to native tool-loop by default. Use
`REPRO_LEGACY_RUNNER_MODE=legacy` only if you intentionally need the old python path.

The tool-loop creates an isolated workspace copy for each run under:

- `temp/agent_framework/reproduce_agent/runs/<run_id>/workspace/repo`

This keeps the source repo clean. The tool-loop can modify the copied workspace,
retry inside it, and leave the original checkout unchanged.

## Qwen on Slurm

To run the native OpenClaw tool-loop on a GPU node with local vLLM, use:

```bash
cd /work/nvme/betu/lily/QOS-agent
bash evaluation/agent_framework/reproduce/submit_reproduce_agent_qwen25_14b.sh
```

This submits:

- `evaluation/agent_framework/reproduce/run_vllm_reproduce_agent_qwen25_14b.slurm`

Default recipe:

- `evaluation/agent_framework/reproduce/examples/qos_run_process_qernels_smoke.json`

The `run_process_qernels_smoke` harness supports `--mode auto` by default:
it uses an external script only when provided and present; otherwise it falls
back to an internal smoke path so deleted legacy `test/` scripts do not block runs.

Override the recipe or max agent steps:

```bash
cd /work/nvme/betu/lily/QOS-agent
REPRO_MAX_AGENT_STEPS=6 \
REPRO_AGENT_EXIT_POLICY=strict \
bash evaluation/agent_framework/reproduce/submit_reproduce_agent_qwen25_14b.sh \
  evaluation/agent_framework/reproduce/examples/qos_import_smoke.json
```

Logs are written under:

- `temp/agent_framework/reproduce_agent/slurm_logs/`

## Output

Each run creates a timestamped directory under `temp/agent_framework/reproduce/runs/` with:

- `recipe.snapshot.json`
- `preflight.json`
- `run.json`
- `stdout.log`
- `stderr.log`
- `metrics.json`
- `status.json`
- `manifest.json`
- `diagnosis.json`
- `figure_verdict.json` when recipe `verification.enabled=true`

Agent-driven runs write to `temp/agent_framework/reproduce_agent/runs/` and include:

- `recipe.snapshot.json`
- `recipe.generated.snapshot.json` when `recipe_authoring.enabled=true`
- `status.json`
- `manifest.json`
- `paper/paper_context.json` when the recipe declares paper documents
- `recipe_authoring/manifest.json` when recipe authoring is enabled
- `attempts/`
- `agent_steps/`

## Structure

- `run_reproduce.py` — generic runner
- `agent_tool_runtime.py` — reproduce tool contract + runtime dispatcher
- `env/` — generic environment diagnosis helpers
- `harnesses/` — generic helper entrypoints
- `figure/` — claim verification scripts
- `recipes/` — generic sample recipes
- `examples/` — repo-specific example recipes

QOS-specific example recipes now include:

- `examples/qos_import_smoke.json`
- `examples/qos_run_process_qernels_smoke.json`

Recipes can optionally declare paper inputs:

- `paper.documents[]` — local PDF/text files to ground the reproduce agent
- `paper.focus_terms[]` — keywords the extractor should emphasize

When present, the reproduce agent extracts a compact `paper_context.json`
artifact and includes it in the decision and repair prompts.

Recipes can also enable recipe authoring:

- `recipe_authoring.enabled=true`

When enabled, the agent first proposes a generated recipe JSON grounded in
paper context and repo entrypoint hints, validates it, and then runs reproduce
with that generated recipe.

## Design Rule

The framework should not assume:

- a specific repo layout
- a specific package name
- a specific benchmark suite
- a specific paper

Those belong in recipes.
