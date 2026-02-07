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

    # Complementarity metrics for better pairing
    depth_similarity = 1.0 - abs(depth1 - depth2) / max(depth1 + depth2, 1)
    parallel_complement = (parallel1 + parallel2) / 2.0
    liveness_complement = 1.0 - (liveness1 + liveness2) / 2.0
    ent_balance = 1.0 - abs(ent_ratio1 - ent_ratio2)
    crit_similarity = 1.0 - abs(crit1 - crit2) / max(crit1 + crit2, 1e-9)
    nonlocal_balance = nonlocal_ratio
    cc_complement = 1.0 / max(cc1 + cc2, 1)
    instr_balance = instr_ratio
    
    # Adjusting weights to give more emphasis to critical metrics
    score = (
        0.8 * util_eff_norm +  # Increase weight on effective utilization
        0.1 * entanglementDiff +  # Maintain weight on entanglement comparison
        0.05 * measurementDiff +  # Maintain weight on measurement comparison
        0.05 * parallelismDiff +  # Slightly increase weight on parallelism comparison
        0.03 * depth_similarity +  # Increase weight on depth similarity
        0.02 * parallel_complement +  # Slightly increase weight on parallelism complementarity
        0.01 * liveness_complement +  # Decrease weight on liveness complementarity
        0.01 * ent_balance +  # Decrease weight on entanglement balance
        0.02 * crit_similarity +  # Decrease weight on critical depth similarity
        0.01 * nonlocal_balance +  # Decrease weight on nonlocal gate balance
        0.04 * instr_balance  # Increase weight on instruction balance
    )

    # Normalize and adjust score further based on effective utilization and parallelism
    score *= (util_eff_norm + parallel_avg) / 2.0

    # Additional adjustment based on effective utilization
    score += 0.1 * util_eff_norm

    # Further adjustments to emphasize key metrics
    score += 0.05 * qubit_ratio  # Add weight on qubit ratio
    score -= 0.05 * prog_comm_avg  # Penalize high program communication

    # Bonus points for high effective utilization
    if util_eff >= 70:
        score *= 1.15

    # Bonus for circuits with balanced parallelism and low program communication
    if parallel_avg >= 0.6 and prog_comm_avg <= 0.3:
        score *= 1.1

    # Bonus for circuits with similar depth
    if depth_ratio >= 0.8:
        score *= 1.1

    # Bonus for circuits with fewer connected components
    if cc_ratio >= 0.7:
        score *= 1.1

    # Bonus for circuits with similar qubit usage
    if qubit_ratio >= 0.8:
        score *= 1.05

    # Additional bonuses for circuits with complementary properties
    if parallel_avg >= 0.6 and liveness_avg <= 0.4:
        score *= 1.1

    # Bonus for circuits with similar nonlocal gate usage
    if nonlocal_ratio >= 0.8:
        score *= 1.05

    # Bonus for circuits with similar number of connected components
    if cc_ratio >= 0.8:
        score *= 1.05

    # Additional heuristic to improve score
    score += 0.02 * (crit_ratio + ent_ratio_avg)  # Encourage circuits with similar critical and entanglement ratios

    # Additional bonus for high utilization and low communication
    if util_eff_norm >= 0.9 and prog_comm_avg < 0.3:
        score *= 1.2

    # Encourage pairing with low liveness and high parallelism
    if liveness_avg <= 0.4 and parallel_avg >= 0.6:
        score *= 1.1

    # Additional bonus for circuits with balanced parallelism and low liveness
    if parallel_avg >= 0.6 and liveness_avg <= 0.4:
        score *= 1.1

    # Bonus for circuits with similar nonlocal gate usage
    if nonlocal_ratio >= 0.8:
        score *= 1.05

    # Bonus for circuits with similar number of connected components
    if cc_ratio >= 0.8:
        score *= 1.05

    return score