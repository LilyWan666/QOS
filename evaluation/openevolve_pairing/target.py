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
    ent_ratio1 = meta1.get("entanglement_ratio", 0.0)
    ent_ratio2 = meta2.get("entanglement_ratio", 0.0)
    crit1 = meta1.get("critical_depth", 0.0)
    crit2 = meta2.get("critical_depth", 0.0)

    util_eff = self.effective_utilization(q1, q2, backend)
    util_eff_norm = util_eff / 100.0
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

    # 核心指标计算
    total_qubits = qubits1 + qubits2
    total_nonlocal = nonlocal1 + nonlocal2
    total_cnot = cnot1 + cnot2
    max_depth = max(depth1, depth2, 1)
    min_depth = min(depth1, depth2, 1)
    
    # 两比特门密度 - 主要的fidelity杀手
    two_q_density = total_nonlocal / max(total_qubits, 1)
    
    # 深度平衡
    depth_balance = min_depth / max_depth
    
    # 纠缠比例平均值
    ent_avg = (ent_ratio1 + ent_ratio2) / 2.0
    
    # 程序通信 - 越高越容易crosstalk
    prog_comm_avg = (prog_comm1 + prog_comm2) / 2.0
    
    # 电路复杂度估计 (用于预测fidelity)
    # 高utilization + 高复杂度 = 低fidelity
    complexity = (
        0.4 * two_q_density / 3.0 +  # normalize by typical max ~3
        0.3 * ent_avg +
        0.2 * prog_comm_avg +
        0.1 * (1.0 - depth_balance)
    )
    complexity = min(complexity, 1.0)
    
    # 关键：utilization越高，对复杂度的惩罚越重
    # 这样可以避免选中高util但低fidelity的点
    if util_eff_norm > 0.55:
        # 高utilization区域，强烈惩罚复杂度
        util_complexity_penalty = complexity * (util_eff_norm - 0.45) * 2.0
    else:
        util_complexity_penalty = 0.0
    
    # 质量分数 - 预测fidelity的能力
    quality_score = (
        0.30 * entanglementDiff +
        0.25 * measurementDiff +
        0.20 * parallelismDiff +
        0.15 * depth_sim +
        0.10 * depth_balance
    )
    
    # 简单电路奖励
    simplicity_bonus = (1.0 - complexity) * 0.3
    
    # 最终分数：质量为主，减去utilization-复杂度的联合惩罚
    score = quality_score + simplicity_bonus - util_complexity_penalty
    
    return score