
参考paper”/mnt/nfs/shared/home/wan25/Document/QOS/paper/qos.pdf”,复现fig11（Multiprogramming结果），需要生成一个python scripts。先从repo目录/mnt/nfs/shared/home/wan25/Document/QOS/evaluation/benchmarks下创建qernel_dict，只用以下9种benchmark的4～24qubits版本做成一个集合作为它的workload：
BENCHMARK_MAPPING = {
    'QAOA-R3': 'qaoa_r3', 'BV': 'bv', 'GHZ': 'ghz', 'HS-1': 'hamsim_1',
    'QAOA-P1': 'qaoa_pl1', 'QSVM': 'qsvm', 'TL-1': 'twolocal_1',
    'VQE-1': 'vqe_1', 'W-STATE': 'wstate'
}
qernel_dict的format如下所示：
            qernel_dict (Dict[Qernel, List[Tuple[List[int], str, float]]]): 
                A dictionary where keys are Qernel objects and values are lists of tuples. 
                Each tuple contains:
                    - A list of integers representing the layout.
                    - A string representing the backend.
                    - A float representing the estimated fidelity.假设第三个float是1。

我们就assume一个backend：IBM_BACKEND = FakeKolkataV2()，是27qubits的qpu。

我要implement paper里的三种方法，一个是no m/p，一个是baseline，一个是qos，paper中有详细的说明。

在figure里，utilization的含义是对于单个电路，如果是utility=30%，那我们只跑qpu上30%的qubits，也就是8，60%是16，88%是24。

对于no m/p 他是跑一个大电路，但是baseline和qos是把两个电路组合起来，然后跑组合之后的电路，这两个电路是从benchmarks folder下从两个for循环选出来的。
对于baseline，用文件夹/mnt/nfs/shared/home/wan25/Document/QOS/Baseline_Multiprogramming 中定义的function直接调用
对于qos:
根据/mnt/nfs/shared/home/wan25/Document/QOS/qos/estimator/README.md，跑run即可跑出physical QPU的layout，然后用layout创建qernel_dict，再跑multiprogramming.py中的process_qernels函数，再用simulator算fidelity(图a）, 和relative fidelity （ 图c）。图b的effective utilization和relative fidelity的公式在paper里，注意effective utilization 和utilization是俩parameters。

把你写的python脚本放进path“/mnt/nfs/shared/home/wan25/Document/QOS/test”，然后生成的fig也放进去，尽量让结果贴近fig11。

requirement:
1. simulation用qiskit自带的aer：from qiskit_aer import AerSimulator
	from qiskit_aer.noise import NoiseModel, depolarizing_error
2. 有些import的path可能不对，但他就是在这个repo里，尽量reuse这个repo里的existing code，不要自己凭空捏造。
3. Effective utilization (图b的纵坐标）的定义在paper的section 7.1
4. Aer的noise model定义可以参考：def _noise_model(p1: float, p2: float):
    AerSimulator, NoiseModel, depolarizing_error = _import_aer()
    noise = NoiseModel()
    err1 = depolarizing_error(p1, 1)
    err2 = depolarizing_error(p2, 2)
    one_q = ["x", "y", "z", "h", "rx", "ry", "rz", "sx", "id", "s", "sdg"]
    two_q = ["cx", "cz", "swap", "rzz", "cp", "ecr"]
    noise.add_all_qubit_quantum_error(err1, one_q)
    noise.add_all_qubit_quantum_error(err2, two_q)
    return noise
5. 每一个bundle最多只能有两个workloads， 我们只考虑两个multiprogram的情况， 如multiprogrammer.py中的process_qernels函数
6. baseline的multiprogramming有一个bug：
            high_utility_neighbors = [n for n in neighbors if utility[n] > alpha]
            low_error_nodes = [n for n in neighbors + [node] if utility[n] < beta]
    这个地方会陷入死循环
7. 多加一些中间的print，让我能看到code运行的进度