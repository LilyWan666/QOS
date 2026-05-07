# Figure Contract Verification

`verify_claim.py` evaluates a figure contract against a metrics JSON artifact.

## Usage

```bash
python evaluation/agent_framework/reproduce/figure/verify_claim.py \
  --contract <contract.json> \
  --metrics <metrics.json> \
  --output <figure_verdict.json>
```

Exit code:

- `0`: claim checks passed
- `2`: claim checks failed

## Contract Schema (Minimal)

```json
{
  "figure_id": "fig11_smoke",
  "paper_claim": "Short natural-language claim",
  "expected_trend": "Optional trend statement",
  "tolerance": {},
  "checks": [
    {"id": "exit_code_zero", "type": "equals", "path": "returncode", "value": 0}
  ],
  "pass_rule": "all"
}
```

`path` supports nested keys and list indices, e.g.:

- `checked_modules[0].module`
- `metrics.summary.score`

## Supported `check.type`

- `exists`
- `equals`
- `bool_equals`
- `in_set`
- `numeric_gte`
- `numeric_lte`
- `numeric_between`
- `approx_equals`
- `monotonic_non_decreasing`
- `monotonic_non_increasing`

## `pass_rule`

- `"all"`: all checks must pass
- `"any"`: at least one check must pass
- `{ "min_pass": N }`: at least `N` checks must pass
