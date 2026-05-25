# QOS Fig. 11 Reproduction Flow Notes

This note summarizes the current Fig. 11 reproduction semantics used by
`qos_fig11_full_agent_autorun.json`.

## Scope

- The run is simulation-based and offline.
- The noise model is the IBM preset depolarizing/readout model with
  `REPRO_NOISE_QPU=ibm_marrakesh` and `REPRO_NOISE_2Q_MODE=layered`.
- The timeout-bounded scaled run maps the paper utilization targets to
  `8q,12q,16q` on a 27-qubit backend. The full `24q` target is not run in this
  recipe because it exceeds the current timeout budget.

## Pair Selection

For each utilization target, the runner generates all candidate circuit pairs
whose qubit counts satisfy:

```text
q1 + q2 == selected_qubits
```

Baseline and QOS use separate selected pair sets.

- Baseline currently uses deterministic random sampling from the candidate set.
- QOS uses `coverage_aware_topk`:
  - score every candidate with repo multiprogrammer metrics,
  - first select pairs that cover all benchmark applications,
  - then fill by score until `REPRO_FIG11_PAIRS_PER_UTIL` is reached,
  - if the requested top-k is too small to cover all applications, expand it.

The selection provenance is written under
`selected_pair_sets_by_threshold` and
`metric_provenance.relative_fidelity`.

## Fig. 11(c)

Each per-application relative-fidelity bar must come from selected QOS pair
simulation records that include that application.

The runner does not fall back to a utilization-level average for missing
applications. Missing coverage is a failure.

Expected provenance:

```json
{
  "selection_mode": "coverage_aware_topk",
  "per_application_source": "selected_qos_pair_records",
  "no_application_fallback": true,
  "application_coverage_complete": true
}
```

## Last Coverage Smoke

`retry60` validated the coverage-aware path:

```text
temp/agent_framework/reproduce_tool_loop/local_full_scaled_total_8_12_16_retry60/
```

The successful attempt was `attempt_0012_run_once`, with all 42 contract checks
passing and all three Fig. 11 PNGs generated.

Note: the recipe currently sets `REPRO_AER_SHOTS=1024` and
`REPRO_PAIR_JOINT_AER_SHOTS=1024` in its own environment block. Shell-provided
values do not override those recipe values in the current tool flow.
