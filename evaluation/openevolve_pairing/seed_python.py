def get_matching_score(self, q1, q2, backend, weighted: bool = False, weights = []) -> float:
    """Replacement for Multiprogrammer.get_matching_score."""
    meta1 = q1.get_metadata()
    meta2 = q2.get_metadata()

    depth1 = meta1.get("depth", 0)
    depth2 = meta2.get("depth", 0)
    qubits1 = meta1.get("num_qubits", 0)
    qubits2 = meta2.get("num_qubits", 0)
    nonlocal1 = meta1.get("num_nonlocal_gates", 0)
    nonlocal2 = meta2.get("num_nonlocal_gates", 0)
    cnot1 = meta1.get("num_cnot_gates", 0)
    cnot2 = meta2.get("num_cnot_gates", 0)
    meas1 = meta1.get("num_measurements", 0)
    meas2 = meta2.get("num_measurements", 0)
    instr1 = meta1.get("number_instructions", 0)
    instr2 = meta2.get("number_instructions", 0)
    cc1 = meta1.get("num_connected_components", 0)
    cc2 = meta2.get("num_connected_components", 0)
    liveness1 = meta1.get("liveness", 0.0)
    liveness2 = meta2.get("liveness", 0.0)
    prog_comm1 = meta1.get("program_communication", 0.0)
    prog_comm2 = meta2.get("program_communication", 0.0)
    parallel1 = meta1.get("parallelism", 0.0)
    parallel2 = meta2.get("parallelism", 0.0)
    meas_ratio1 = meta1.get("measurement", 0.0)
    meas_ratio2 = meta2.get("measurement", 0.0)
    ent_ratio1 = meta1.get("entanglement_ratio", 0.0)
    ent_ratio2 = meta2.get("entanglement_ratio", 0.0)
    crit1 = meta1.get("critical_depth", 0.0)
    crit2 = meta2.get("critical_depth", 0.0)

    # Utilization (0-1). We treat utilization as necessary but not sufficient:
    # high fidelity is often dominated by circuit "activity"/contention.
    util_eff = self.effective_utilization(q1, q2, backend)
    util_eff_norm = util_eff / 100.0

    # Available comparison helpers (assumed 0-1, higher is better).
    entanglementDiff = self.entanglementComparison(q1, q2)
    measurementDiff = self.measurementComparison(q1, q2)
    parallelismDiff = self.parallelismComparison(q1, q2)
    depth_sim = self.depthComparison(q1, q2)

    # OE_BEGIN
    if weighted and sum(weights) > 0:
        return (
            weights[0] * util_eff +
            weights[1] * entanglementDiff +
            weights[2] * measurementDiff +
            weights[3] * parallelismDiff
        )
    # OE_END

    # ------------------------------
    # Fidelity proxy / risk model
    # ------------------------------
    # We cannot directly observe fidelity here, but we can approximate "error risk"
    # from structural/activity features. We prefer pairs that:
    #   * fill the device (utilization) AND
    #   * avoid pairing two highly-entangling / deep / communication-heavy programs
    #     that tend to depress fidelity.
    #
    # Normalizations are intentionally simple and scale-free.
    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    d1 = _safe_float(depth1)
    d2 = _safe_float(depth2)
    q1n = max(_safe_float(qubits1), 0.0)
    q2n = max(_safe_float(qubits2), 0.0)

    i1 = _safe_float(instr1)
    i2 = _safe_float(instr2)
    # Fallback when instruction counts are missing.
    if i1 <= 0.0:
        i1 = max(d1 * max(q1n, 1.0), 1.0)
    if i2 <= 0.0:
        i2 = max(d2 * max(q2n, 1.0), 1.0)

    # Densities capture "activity" better than raw counts.
    cnot_den1 = _safe_float(cnot1) / max(i1, 1.0)
    cnot_den2 = _safe_float(cnot2) / max(i2, 1.0)
    nonlocal_den1 = _safe_float(nonlocal1) / max(i1, 1.0)
    nonlocal_den2 = _safe_float(nonlocal2) / max(i2, 1.0)
    meas_den1 = _safe_float(meas1) / max(i1, 1.0)
    meas_den2 = _safe_float(meas2) / max(i2, 1.0)

    # Depth/critical depth: map to [0,1) smoothly with a fixed knee.
    depth_n1 = d1 / (d1 + 20.0) if d1 > 0.0 else 0.0
    depth_n2 = d2 / (d2 + 20.0) if d2 > 0.0 else 0.0
    cdepth1 = _safe_float(crit1)
    cdepth2 = _safe_float(crit2)
    cdepth_n1 = cdepth1 / (cdepth1 + 20.0) if cdepth1 > 0.0 else 0.0
    cdepth_n2 = cdepth2 / (cdepth2 + 20.0) if cdepth2 > 0.0 else 0.0

    # Map unbounded metadata to [0,1).
    live1 = _safe_float(liveness1)
    live2 = _safe_float(liveness2)
    live_n1 = live1 / (live1 + 1.0) if live1 > 0.0 else 0.0
    live_n2 = live2 / (live2 + 1.0) if live2 > 0.0 else 0.0

    pc1 = _safe_float(prog_comm1)
    pc2 = _safe_float(prog_comm2)
    pc_n1 = pc1 / (pc1 + 1.0) if pc1 > 0.0 else 0.0
    pc_n2 = pc2 / (pc2 + 1.0) if pc2 > 0.0 else 0.0

    # Entanglement / measurement ratios are already in [0,1] (usually).
    er1 = max(min(_safe_float(ent_ratio1), 1.0), 0.0)
    er2 = max(min(_safe_float(ent_ratio2), 1.0), 0.0)
    mr1 = max(min(_safe_float(meas_ratio1), 1.0), 0.0)
    mr2 = max(min(_safe_float(meas_ratio2), 1.0), 0.0)

    # "Activity" summarizes two-qubit interactions and nonlocal structure.
    act1 = max(cnot_den1, nonlocal_den1) + 0.5 * er1
    act2 = max(cnot_den2, nonlocal_den2) + 0.5 * er2

    # Individual risk: depth + activity + pressure + communication (+ small meas penalty).
    r1 = (
        0.30 * depth_n1 +
        0.20 * cdepth_n1 +
        0.25 * min(act1, 1.0) +
        0.15 * live_n1 +
        0.10 * pc_n1 +
        0.05 * min(meas_den1 + mr1, 1.0)
    )
    r2 = (
        0.30 * depth_n2 +
        0.20 * cdepth_n2 +
        0.25 * min(act2, 1.0) +
        0.15 * live_n2 +
        0.10 * pc_n2 +
        0.05 * min(meas_den2 + mr2, 1.0)
    )

    # Interaction risk: pairing two "busy" programs is especially harmful.
    interaction = (
        0.55 * min(act1 * act2, 1.0) +
        0.20 * min(er1 * er2, 1.0) +
        0.15 * min(live_n1 * live_n2, 1.0) +
        0.10 * min(pc_n1 * pc_n2, 1.0)
    )

    # Total risk (0..~1). Convert to a fidelity proxy in [0,1].
    risk = (r1 + r2) * 0.5
    risk = risk + 0.75 * interaction
    if risk < 0.0:
        risk = 0.0
    if risk > 1.0:
        risk = 1.0
    fidelity_proxy = 1.0 - risk

    # ------------------------------
    # Balance / complementarity
    # ------------------------------
    # Avoid pathological pairs (one tiny + one huge) that may underperform on either
    # utilization stability or fidelity due to extreme asymmetry.
    qubit_balance = min(q1n, q2n) / max(q1n, q2n, 1.0)
    depth_balance = min(d1, d2) / max(d1, d2, 1.0)
    balance = 0.5 * qubit_balance + 0.5 * depth_balance

    # Compatibility from provided comparison helpers (kept as a mild term).
    compat = (entanglementDiff + measurementDiff + parallelismDiff + depth_sim) / 4.0
    if compat < 0.0:
        compat = 0.0
    if compat > 1.0:
        compat = 1.0

    # Final score: prioritize fidelity proxy while still rewarding high utilization.
    # Exponents bias toward selecting Pareto-front pairs (high util AND high fidelity).
    score = (util_eff_norm ** 1.10) * (fidelity_proxy ** 2.20) * (0.70 + 0.30 * compat) * (0.85 + 0.15 * balance)

    # Keep within a stable numeric range.
    if score != score:  # NaN guard
        return 0.0
    if score < 0.0:
        return 0.0
    return float(score)