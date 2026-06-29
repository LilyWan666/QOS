# QOS Fig. 11 Reproduction Flow Notes

This note summarizes the current Fig. 11 reproduction semantics used by
`qos_fig11_full_agent_autorun.json`.

## Scope

- The run is simulation-based and offline.
- The noise model is the IBM preset depolarizing/readout model with
  `REPRO_NOISE_QPU=ibm_marrakesh` and `REPRO_NOISE_2Q_MODE=layered`.
- The current strict run maps the paper utilization targets back to
  `8q,16q,24q` on a 27-qubit backend. The `24q` target must be attempted by the
  simulation branch; if it is too slow, the run should fail or time out rather
  than pass through a heuristic fallback.
- The strict simulation budget is `900s`. A timeout is classified as
  `simulation_too_expensive` and should automatically route the tool loop to
  `openevolve_target_probe -> openevolve_param_probe -> openevolve_proxy_search
  -> openevolve_proxy_verify`. This proxy branch is an acceleration follow-up,
  not Fig. 11 strict success.
- Strict Fig. 11 simulation uses `REPRO_AER_SHOTS=8192`. The figure contract
  checks this value in top-level simulation provenance and in Fig. 11(a/b/c)
  metric derivations.
- Proxy or OpenEvolve outputs are not valid Fig. 11 strict-success inputs.
  If 8192-shot simulation is too slow, the flow should first find target
  functions, then run a separate `Evolve -> Verify` path for proxy/search
  artifacts.
- The OpenEvolve follow-up path has its own contract:
  `evaluation/agent_framework/reproduce/examples/figure_contracts/qos_openevolve_proxy_search.contract.json`.
  That contract requires the evolution implementation to come from
  `third_party/openevolve` and explicitly marks the result as
  `not_fig11_strict_success`.

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

Note: the recipe currently sets one simulation-shot knob, `REPRO_AER_SHOTS=8192`,
in its own environment block. Shell-provided values do not override recipe
values in the current tool flow.

## Slow Simulation Follow-up Flow

If strict simulation is too slow, do not downgrade Fig. 11 success. Follow this
order instead:

```text
Find target functions
  - Evolve: locate OpenEvolve candidate scoring / proxy search entry points
  - Verify: locate simulation-grounded verification and comparison entry points
Evolve
  - run proxy/evolution search from `third_party/openevolve` as a separate
    artifact path
Verify
  - compare evolved/proxy rankings against 8192-shot simulation provenance
```
