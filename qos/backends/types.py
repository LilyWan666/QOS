"""QPU type definition for QOS backends."""

from typing import Any, Dict, List, Optional


class QPU:
    """Represents a Quantum Processing Unit."""

    id: int
    provider: str
    name: str
    alias: str
    local_queue: List[tuple]
    max_qubits: int
    args: Dict[str, Any]

    def __init__(self) -> None:
        self.id = 0
        self.provider = ""
        self.name = ""
        self.alias = ""
        self.local_queue = []
        self.max_qubits = 0
        self.args = {}
