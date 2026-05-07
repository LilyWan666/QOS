#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StubQPU:
    name: str
    num_qubits: int


class SimpleCircuit:
    def __init__(self, name: str, num_qubits: int, ops: dict[str, int]) -> None:
        self.name = name
        self.num_qubits = num_qubits
        self._ops = ops

    def count_ops(self) -> dict[str, int]:
        return dict(self._ops)

    def depth(self) -> int:
        return max(1, sum(self._ops.values()))


class StubQernel:
    def __init__(self, circuit: SimpleCircuit, metadata: dict[str, float], label: str) -> None:
        self._circuit = circuit
        self._metadata = metadata
        self.label = label

    def get_circuit(self) -> SimpleCircuit:
        return self._circuit

    def get_metadata(self) -> dict[str, float]:
        return self._metadata

    def num_qubits(self) -> int:
        return self._circuit.num_qubits

    def depth(self) -> int:
        return self._circuit.depth()

    def __hash__(self) -> int:
        return hash(self.label)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StubQernel) and self.label == other.label

def circuit_metadata(circuit: SimpleCircuit) -> dict[str, float]:
    ops = circuit.count_ops()
    total_ops = float(sum(ops.values()) or 1)
    depth = float(circuit.depth() or 0)
    num_qubits = float(max(circuit.num_qubits, 1))
    two_qubit_ops = float(sum(count for name, count in ops.items() if name in {"cx", "cz", "ecr"}))
    measure_ops = float(ops.get("measure", 0))
    parallelism = max(0.0, min(1.0, 1.0 - (depth / max(total_ops, 1.0))))
    return {
        "depth": depth,
        "entanglement_ratio": max(0.0, min(1.0, two_qubit_ops / total_ops)),
        "measurement": max(0.0, min(1.0, measure_ops / total_ops)),
        "parallelism": parallelism,
        "num_qubits": num_qubits,
    }


def _load_simple_qasm(path: Path) -> SimpleCircuit:
    ops: dict[str, int] = {}
    num_qubits = 1
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("qreg ") and "[" in line and "]" in line:
            try:
                num_qubits = max(num_qubits, int(line.split("[", 1)[1].split("]", 1)[0]))
            except ValueError:
                pass
            continue
        if line.startswith(("OPENQASM", "include", "qreg", "creg", "barrier")):
            continue
        op = line.split(None, 1)[0].split("(", 1)[0].strip().rstrip(";")
        if op:
            ops[op] = ops.get(op, 0) + 1
    return SimpleCircuit(name=path.stem, num_qubits=num_qubits, ops=ops or {"id": 1})


def load_fixed_circuits(repo_root: Path) -> list[tuple[str, SimpleCircuit]]:
    relative = [
        "evaluation/benchmarks/bv/8.qasm",
        "evaluation/benchmarks/ghz/8.qasm",
    ]
    out: list[tuple[str, SimpleCircuit]] = []
    for rel in relative:
        path = repo_root / rel
        qc = _load_simple_qasm(path)
        out.append((rel, qc))
    return out


def install_simulation_stubs() -> None:
    """Keep fig11 smoke reproduction offline: no IBMProvider/token/QPU access."""
    class SimulationEngine:
        pass

    class SimulationQernel:
        pass

    class SimulationQPU:
        def __init__(self) -> None:
            self.id = -1
            self.provider = "simulation"
            self.name = "SimulationQPU"
            self.alias = "simulation"
            self.args = {}
            self.local_queue = []

    qiskit_module = types.ModuleType("qiskit")
    qiskit_module.__path__ = []
    providers_module = types.ModuleType("qiskit.providers")
    fake_provider_module = types.ModuleType("qiskit.providers.fake_provider")
    converters_module = types.ModuleType("qiskit.converters")
    compiler_module = types.ModuleType("qiskit.compiler")
    qiskit_module.QuantumCircuit = SimpleCircuit
    qiskit_module.dagcircuit = types.SimpleNamespace()
    qiskit_module.transpile = lambda circuits, *args, **kwargs: circuits
    converters_module.circuit_to_dag = lambda circuit: circuit
    compiler_module.transpile = lambda circuits, *args, **kwargs: circuits
    qiskit_module.providers = providers_module
    providers_module.fake_provider = fake_provider_module
    sys.modules.setdefault("qiskit", qiskit_module)
    sys.modules.setdefault("qiskit.providers", providers_module)
    sys.modules.setdefault("qiskit.providers.fake_provider", fake_provider_module)
    sys.modules.setdefault("qiskit.converters", converters_module)
    sys.modules.setdefault("qiskit.compiler", compiler_module)

    qos_types_module = types.ModuleType("qos.types")
    qos_types_types_module = types.ModuleType("qos.types.types")
    qos_types_types_module.Engine = SimulationEngine
    qos_types_types_module.Qernel = SimulationQernel
    qos_types_types_module.QPU = SimulationQPU
    qos_types_module.types = qos_types_types_module
    sys.modules.setdefault("qos.types", qos_types_module)
    sys.modules.setdefault("qos.types.types", qos_types_types_module)

    qos_database_module = types.ModuleType("qos.database")
    sys.modules.setdefault("qos.database", qos_database_module)

    time_estimator_module = types.ModuleType("qos.time_estimator")
    basic_estimator_module = types.ModuleType("qos.time_estimator.basic_estimator")

    class SimulationCircuitEstimator:
        pass

    basic_estimator_module.CircuitEstimator = SimulationCircuitEstimator
    time_estimator_module.basic_estimator = basic_estimator_module
    sys.modules.setdefault("qos.time_estimator", time_estimator_module)
    sys.modules.setdefault("qos.time_estimator.basic_estimator", basic_estimator_module)

    mapomatic_module = types.ModuleType("mapomatic")
    mapomatic_layouts_module = types.ModuleType("mapomatic.layouts")
    mapomatic_module.layouts = mapomatic_layouts_module
    sys.modules.setdefault("mapomatic", mapomatic_module)
    sys.modules.setdefault("mapomatic.layouts", mapomatic_layouts_module)

    qos_backends_module = types.ModuleType("qos.backends")
    qos_backends_types_module = types.ModuleType("qos.backends.types")
    qos_backends_types_module.QPU = SimulationQPU
    qos_backends_module.types = qos_backends_types_module
    sys.modules.setdefault("qos.backends", qos_backends_module)
    sys.modules.setdefault("qos.backends.types", qos_backends_types_module)

    estimator_module = types.ModuleType("qos.estimator.estimator")

    class SimulationEstimator:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    estimator_module.Estimator = SimulationEstimator
    sys.modules.setdefault("qos.estimator.estimator", estimator_module)


def select_pair_offline(mp, qernel_dict: dict, threshold: float):
    def matching_score(q1: StubQernel, q2: StubQernel, backend: StubQPU) -> float:
        metadata_1 = q1.get_metadata()
        metadata_2 = q2.get_metadata()
        spatial = (q1.num_qubits() + q2.num_qubits()) / max(float(backend.num_qubits), 1.0)
        entanglement = (1 - metadata_1["entanglement_ratio"]) * (1 - metadata_2["entanglement_ratio"])
        measurement = (1 - metadata_1["measurement"]) * (1 - metadata_2["measurement"])
        parallelism = (1 - metadata_1["parallelism"]) * (1 - metadata_2["parallelism"])
        return float((spatial + entanglement + measurement + parallelism) / 4)

    results = []
    for q1, q1_data in qernel_dict.items():
        for q2, q2_data in qernel_dict.items():
            if q1 == q2:
                continue
            for layout1, backend1, _ in q1_data:
                for layout2, backend2, _ in q2_data:
                    if backend1 != backend2:
                        continue
                    spatial_util = (q1.num_qubits() + q2.num_qubits()) / max(float(backend1.num_qubits), 1.0)
                    score = matching_score(q1, q2, backend1)
                    if score > threshold:
                        results.append((q1, q2, layout1, layout2, score, spatial_util, backend1))
    if not results:
        return None
    results.sort(key=lambda item: (item[4], item[5]), reverse=True)
    return results[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    install_simulation_stubs()
    from qos.multiprogrammer.multiprogrammer import Multiprogrammer

    loaded = load_fixed_circuits(repo_root)
    backend = StubQPU(name="FakeMarrakeshV2", num_qubits=127)

    qernels: list[StubQernel] = []
    for idx, (path, qc) in enumerate(loaded):
        qernels.append(StubQernel(circuit=qc, metadata=circuit_metadata(qc), label=f"q{idx}:{path}"))

    qernel_dict = {}
    cursor = 0
    for q in qernels:
        width = q.get_circuit().num_qubits
        layout = list(range(cursor, cursor + width))
        cursor += width
        qernel_dict[q] = [(layout, backend, 1.0)]

    mp = Multiprogrammer()
    selected = select_pair_offline(mp, qernel_dict=qernel_dict, threshold=args.threshold)

    print(f"Built {len(qernels)} qernels in external mode")
    print(json.dumps({"backend": backend.name, "threshold": args.threshold}, sort_keys=True))

    if selected is None:
        print("No pair selected.")
        return 0

    q1, q2, layout1, layout2, matching_score, spatial_util, _backend = selected
    print(
        "Selected pair: "
        f"{q1.label} + {q2.label} "
        f"matching_score={float(matching_score):.6f} "
        f"spatial_util={float(spatial_util):.6f}"
    )
    print(
        json.dumps(
            {
                "layout_1": layout1,
                "layout_2": layout2,
                "q1_depth": q1.get_circuit().depth(),
                "q2_depth": q2.get_circuit().depth(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
