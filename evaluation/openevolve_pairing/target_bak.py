"""Evolution target: get_matching_score replacement."""
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

    depth_ratio = min(depth1, depth2) / max(depth1, depth2, 1)
    qubit_ratio = min(qubits1, qubits2) / max(qubits1, qubits2, 1)
    nonlocal_ratio = min(nonlocal1, nonlocal2) / max(nonlocal1, nonlocal2, 1)
    cnot_ratio = min(cnot1, cnot2) / max(cnot1, cnot2, 1)
    meas_ratio = min(meas1, meas2) / max(meas1, meas2, 1)
    instr_ratio = min(instr1, instr2) / max(instr1, instr2, 1)
    cc_ratio = min(cc1, cc2) / max(cc1, cc2, 1)
    crit_ratio = min(crit1, crit2) / max(crit1, crit2, 1e-9)
    liveness_avg = (liveness1 + liveness2) / 2.0
    prog_comm_avg = (prog_comm1 + prog_comm2) / 2.0
    parallel_avg = (parallel1 + parallel2) / 2.0
    meas_ratio_avg = (meas_ratio1 + meas_ratio2) / 2.0
    ent_ratio_avg = (ent_ratio1 + ent_ratio2) / 2.0
    depth_sim = self.depthComparison(q1, q2)

    util_eff = self.effective_utilization(q1, q2, backend)
    util_eff_norm = util_eff / 100.0
    entanglementDiff = self.entanglementComparison(q1, q2)
    measurementDiff = self.measurementComparison(q1, q2)
    parallelismDiff = self.parallelismComparison(q1, q2)

    # OE_BEGIN
    if weighted and sum(weights) > 0:
        return (
            weights[0] * util_eff +
            weights[1] * entanglementDiff +
            weights[2] * measurementDiff +
            weights[3] * parallelismDiff
        )
    # OE_END

    # Compute complementarity metrics for better pairing
    # Circuits with similar depth allow better scheduling
    depth_similarity = 1.0 - abs(depth1 - depth2) / max(depth1 + depth2, 1)

    # Parallelism complementarity: prefer pairs where combined parallelism is high
    parallel_complement = (parallel1 + parallel2) / 2.0

    # Liveness complementarity: lower combined liveness means less qubit pressure
    liveness_complement = 1.0 - (liveness1 + liveness2) / 2.0

    # Entanglement balance: similar entanglement ratios pair better
    ent_balance = 1.0 - abs(ent_ratio1 - ent_ratio2)

    # Critical depth similarity for better scheduling alignment
    crit_similarity = 1.0 - abs(crit1 - crit2) / max(crit1 + crit2, 1e-9)

    # Nonlocal gate balance: similar nonlocal gate counts reduce interference
    nonlocal_balance = nonlocal_ratio

    # Connected components: fewer combined components is better
    cc_complement = 1.0 / max(cc1 + cc2, 1)

    # Instruction balance for resource utilization
    instr_balance = instr_ratio

    # Weighted combination emphasizing utilization and complementarity
    score = (
        0.35 * util_eff_norm +
        0.15 * entanglementDiff +
        0.10 * measurementDiff +
        0.12 * parallelismDiff +
        0.08 * depth_similarity +
        0.05 * parallel_complement +
        0.05 * liveness_complement +
        0.04 * ent_balance +
        0.03 * crit_similarity +
        0.02 * nonlocal_balance +
        0.01 * instr_balance
    )

    return score