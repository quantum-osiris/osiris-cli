"""Sovereign backend — classical pure-NumPy statevector execution via SovereignExecutor."""

from __future__ import annotations

import random
import time
from typing import Any, Set

from qbyte_system.sovereign_executor import Circuit as SovereignCircuit
from qbyte_system.sovereign_executor import SovereignExecutor

from ..dna_ir import IROpType, QuantumCircuitIR
from ..dna_runtime import ExecutionResult
from .base import BackendAdapter
from .registry import BackendRegistry

MAX_STATEVECTOR_QUBITS = 16
