"""
SovereignExecutor — Pure-NumPy Classical Quantum Circuit Simulator
==================================================================

Classical statevector simulator for arbitrary small-width quantum circuits
without external SDK dependencies.

Note on physical scope:
    SovereignExecutor is a classical NumPy statevector simulator executing on CPU.
    It is NOT an independent physical quantum processor or hardware substrate.
    Heuristic non-unitary phase-conjugate "healing" is disabled by default
    (`enable_healing=False`) so that unitary circuits remain strictly unitary.

Basis-order convention (Big-Endian):
    For an n-qubit circuit, computational basis state |b_0 b_1 ... b_{n-1}> maps
    to flat statevector index:
        idx = sum_{q=0}^{n-1} b_q * 2^(n - 1 - q)
    so qubit q in {0..n-1} corresponds to tensor axis q and integer bit shift
    (n - 1 - q). Use `result.get_statevector(endian="little")` or
    `executor.get_statevector(endian="little")` for Qiskit-compatible
    little-endian ordering |b_{n-1} ... b_1 b_0>.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

try:
    from .gates import (
        helix, cleave, twist, fold, splice, bond, phase_flip,
        rotate_y, controlled_z, swap, iswap, controlled_phase,
        xx_rotation, yy_rotation, zz_rotation,
        t_gate, t_dg, s_gate, s_dg, toffoli, fredkin,
    )
    from .phase_conjugate import PhaseConjugateEngine
    from .ccce_runtime import create_runtime
except ImportError:
    from gates import (
        helix, cleave, twist, fold, splice, bond, phase_flip,
        rotate_y, controlled_z, swap, iswap, controlled_phase,
        xx_rotation, yy_rotation, zz_rotation,
        t_gate, t_dg, s_gate, s_dg, toffoli, fredkin,
    )
    from phase_conjugate import PhaseConjugateEngine
    from ccce_runtime import create_runtime

# Physical Constants
LAMBDA_PHI = 2.176435e-8
PHI_THRESHOLD = 0.7734
GAMMA_FIXED = 0.092


class ExecutionMode(Enum):
    """Quantum execution modes."""
    STATEVECTOR = "statevector"  # Full state vector simulation
    TENSOR = "tensor"            # Tensor network (future)
    SAMPLING = "sampling"        # Monte Carlo sampling
    NATIVE = "native"            # Native hardware (future)


@dataclass
class GateOp:
    """Represents a quantum gate operation."""
    name: str
    qubits: List[int]
    params: List[float] = field(default_factory=list)
    controls: List[int] = field(default_factory=list)


@dataclass
class Circuit:
    """Quantum circuit representation."""
    n_qubits: int
    operations: List[GateOp] = field(default_factory=list)
    name: str = "unnamed"

    def add(self, name: str, qubits: List[int],
            params: Optional[List[float]] = None,
            controls: Optional[List[int]] = None):
        """Add a gate operation."""
        self.operations.append(GateOp(
            name=name,
            qubits=qubits,
            params=params or [],
            controls=controls or []
        ))
        return self

    # DNA-encoded gate shortcuts
    def helix(self, qubit: int):
        """Add Hadamard (DNA helix unwinding)."""
        return self.add('helix', [qubit])

    def cleave(self, qubit: int):
        """Add Pauli-X (DNA cleavage)."""
        return self.add('cleave', [qubit])

    def twist(self, qubit: int, theta: float):
        """Add RZ rotation (DNA twist)."""
        return self.add('twist', [qubit], [theta])

    def fold(self, qubit: int, theta: float):
        """Add RY rotation (protein folding)."""
        return self.add('fold', [qubit], [theta])

    def splice(self, qubit: int, theta: float):
        """Add RX rotation (RNA splicing)."""
        return self.add('splice', [qubit], [theta])

    def bond(self, control: int, target: int):
        """Add CNOT (hydrogen bonding)."""
        return self.add('bond', [target], controls=[control])

    def phase(self, qubit: int):
        """Add Pauli-Z (phase flip)."""
        return self.add('phase', [qubit])

    def cz(self, q1: int, q2: int):
        """Add CZ gate."""
        return self.add('cz', [q1, q2])

    def swap_qubits(self, q1: int, q2: int):
        """Add SWAP gate."""
        return self.add('swap', [q1, q2])

    def toffoli_gate(self, c1: int, c2: int, target: int):
        """Add Toffoli (CCNOT) gate."""
        return self.add('toffoli', [target], controls=[c1, c2])

    def fredkin_gate(self, control: int, q1: int, q2: int):
        """Add Fredkin (CSWAP) gate."""
        return self.add('fredkin', [q1, q2], controls=[control])

    def y(self, qubit: int):
        """Add Pauli-Y gate."""
        return self.add('y', [qubit])

    def s(self, qubit: int):
        """Add S gate."""
        return self.add('s', [qubit])

    def sdg(self, qubit: int):
        """Add S† gate."""
        return self.add('sdg', [qubit])

    def t(self, qubit: int):
        """Add T gate."""
        return self.add('t', [qubit])

    def tdg(self, qubit: int):
        """Add T† gate."""
        return self.add('tdg', [qubit])

    def cphase(self, control: int, target: int, phi: float):
        """Add controlled-phase CP(phi) gate."""
        return self.add('cphase', [control, target], [phi])

    def rxx(self, q1: int, q2: int, theta: float):
        """Add RXX(theta) gate."""
        return self.add('rxx', [q1, q2], [theta])

    def ryy(self, q1: int, q2: int, theta: float):
        """Add RYY(theta) gate."""
        return self.add('ryy', [q1, q2], [theta])

    def rzz(self, q1: int, q2: int, theta: float):
        """Add RZZ(theta) gate."""
        return self.add('rzz', [q1, q2], [theta])

    # Standard aliases
    h = helix
    x = cleave
    z = phase
    rx = splice
    ry = fold
    rz = twist
    cx = bond
    ccx = toffoli_gate
    ccnot = toffoli_gate
    cswap = fredkin_gate

    def measure(self, qubit: int, classical_bit: Optional[int] = None):
        """Add measurement operation."""
        cbit = qubit if classical_bit is None else classical_bit
        return self.add('measure', [qubit], [float(cbit)])

    def barrier(self):
        """Add barrier (for visualization)."""
        return self.add('barrier', [])

    def depth(self) -> int:
        """Compute circuit depth (layers)."""
        if not self.operations:
            return 0

        # Track last operation time for each qubit
        qubit_depth = [0] * self.n_qubits
        for op in self.operations:
            if op.name in ('barrier', 'measure'):
                continue
            involved = op.qubits + op.controls
            max_depth = max(qubit_depth[q] for q in involved) if involved else 0
            for q in involved:
                qubit_depth[q] = max_depth + 1

        return max(qubit_depth) if qubit_depth else 0

    def gate_count(self) -> int:
        """Count number of gates (excluding barriers/measures)."""
        return sum(1 for op in self.operations
                   if op.name not in ('barrier', 'measure'))

    def __repr__(self) -> str:
        return (f"Circuit('{self.name}', qubits={self.n_qubits}, "
                f"gates={self.gate_count()}, depth={self.depth()})")


@dataclass
class ExecutionResult:
    """Result of circuit execution."""
    counts: Dict[str, int]           # Measurement counts
    statevector: Optional[np.ndarray] = None  # Final state (if available)
    probabilities: Optional[np.ndarray] = None  # Probability distribution
    metrics: Optional[Dict[str, float]] = None  # CCCE metrics
    execution_time: float = 0.0
    shots: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def most_likely(self) -> str:
        """Return most likely measurement outcome."""
        if not self.counts:
            return ""
        return max(self.counts, key=self.counts.get)

    def get_counts(self, endian: str = "big") -> Dict[str, int]:
        """Return measurement counts in 'big' (|q_0..q_{n-1}>) or 'little' (|q_{n-1}..q_0>) order."""
        if endian == "big":
            return dict(self.counts)
        if endian == "little":
            return {k[::-1]: v for k, v in self.counts.items()}
        raise ValueError(f"endian must be 'big' or 'little', got {endian!r}")

    def get_statevector(self, endian: str = "big") -> Optional[np.ndarray]:
        """Return final statevector in 'big' (|q_0..q_{n-1}>) or 'little' (|q_{n-1}..q_0>) order."""
        if self.statevector is None:
            return None
        if endian == "big":
            return self.statevector.copy()
        if endian == "little":
            n_q = int(round(np.log2(len(self.statevector))))
            tensor_state = self.statevector.reshape([2] * n_q)
            return np.transpose(tensor_state, list(range(n_q - 1, -1, -1))).reshape(len(self.statevector)).copy()
        raise ValueError(f"endian must be 'big' or 'little', got {endian!r}")

    def expectation(self, observable: str = "Z") -> float:
        """Compute expectation value for simple observables."""
        if not self.counts:
            return 0.0

        total = sum(self.counts.values())
        exp_val = 0.0

        for bitstring, count in self.counts.items():
            # For Z observable, +1 for |0⟩, -1 for |1⟩
            if observable == "Z":
                parity = sum(int(b) for b in bitstring) % 2
                exp_val += (-1) ** parity * count / total

        return exp_val


class SovereignExecutor:
    """
    Pure-NumPy Classical Statevector Circuit Executor.

    Executes small-width quantum circuits deterministically (statevector) or via
    Monte-Carlo shot sampling on CPU without external vendor SDK dependencies.
    Uses a uniform big-endian computational basis convention |q_0 q_1 ... q_{n-1}>.
    """

    def __init__(self, n_qubits: int = 8,
                 mode: ExecutionMode = ExecutionMode.STATEVECTOR,
                 enable_healing: bool = False,
                 enable_ccce: bool = False):
        """
        Initialize Sovereign Executor.

        Args:
            n_qubits: Maximum number of qubits supported by this executor instance
            mode: Execution mode
            enable_healing: Enable heuristic non-unitary phase-conjugate healing
                            (default False so unitary circuits remain strictly unitary)
            enable_ccce: Enable heuristic CCCE metric tracking (default False)
        """
        self.n_qubits = n_qubits
        self._active_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.mode = mode
        self.enable_healing = enable_healing
        self.enable_ccce = enable_ccce

        # Initialize state to |0...0⟩
        self._state = np.zeros(self.dim, dtype=np.complex128)
        self._state[0] = 1.0

        # Healing engine
        if enable_healing:
            self._healing_engine = PhaseConjugateEngine()
        else:
            self._healing_engine = None

        # CCCE runtime
        if enable_ccce:
            self._ccce = create_runtime("SovereignExecutor")
        else:
            self._ccce = None

        # Statistics
        self._execution_count = 0
        self._total_gates = 0
        self._genesis = time.time()

        # Gate dispatch table
        self._gate_dispatch = self._build_gate_dispatch()

    def _build_gate_dispatch(self) -> Dict[str, Callable]:
        """Build dispatch table for gate operations."""
        return {
            'helix': lambda op: self._apply_single(helix(), op.qubits[0]),
            'h': lambda op: self._apply_single(helix(), op.qubits[0]),
            'cleave': lambda op: self._apply_single(cleave(), op.qubits[0]),
            'x': lambda op: self._apply_single(cleave(), op.qubits[0]),
            'y': lambda op: self._apply_single(rotate_y(), op.qubits[0]),
            'phase': lambda op: self._apply_single(phase_flip(), op.qubits[0]),
            'z': lambda op: self._apply_single(phase_flip(), op.qubits[0]),
            'twist': lambda op: self._apply_single(twist(op.params[0]), op.qubits[0]),
            'rz': lambda op: self._apply_single(twist(op.params[0]), op.qubits[0]),
            'fold': lambda op: self._apply_single(fold(op.params[0]), op.qubits[0]),
            'ry': lambda op: self._apply_single(fold(op.params[0]), op.qubits[0]),
            'splice': lambda op: self._apply_single(splice(op.params[0]), op.qubits[0]),
            'rx': lambda op: self._apply_single(splice(op.params[0]), op.qubits[0]),
            't': lambda op: self._apply_single(t_gate(), op.qubits[0]),
            'tdg': lambda op: self._apply_single(t_dg(), op.qubits[0]),
            's': lambda op: self._apply_single(s_gate(), op.qubits[0]),
            'sdg': lambda op: self._apply_single(s_dg(), op.qubits[0]),
            'bond': lambda op: self._apply_cnot(op.controls[0], op.qubits[0]),
            'cx': lambda op: self._apply_cnot(
                op.controls[0] if op.controls else op.qubits[0],
                op.qubits[0] if op.controls else op.qubits[1],
            ),
            'cz': lambda op: self._apply_two(controlled_z(), op.qubits[0], op.qubits[1]),
            'swap': lambda op: self._apply_two(swap(), op.qubits[0], op.qubits[1]),
            'iswap': lambda op: self._apply_two(iswap(), op.qubits[0], op.qubits[1]),
            'cphase': lambda op: self._apply_two(controlled_phase(op.params[0]), op.qubits[0], op.qubits[1]),
            'cp': lambda op: self._apply_two(controlled_phase(op.params[0]), op.qubits[0], op.qubits[1]),
            'rxx': lambda op: self._apply_two(xx_rotation(op.params[0]), op.qubits[0], op.qubits[1]),
            'ryy': lambda op: self._apply_two(yy_rotation(op.params[0]), op.qubits[0], op.qubits[1]),
            'rzz': lambda op: self._apply_two(zz_rotation(op.params[0]), op.qubits[0], op.qubits[1]),
            'toffoli': lambda op: self._apply_toffoli(op.controls[0], op.controls[1], op.qubits[0]),
            'ccnot': lambda op: self._apply_toffoli(op.controls[0], op.controls[1], op.qubits[0]),
            'ccx': lambda op: self._apply_toffoli(op.controls[0], op.controls[1], op.qubits[0]),
            'fredkin': lambda op: self._apply_three(fredkin(), op.controls[0], op.qubits[0], op.qubits[1]),
            'cswap': lambda op: self._apply_three(fredkin(), op.controls[0], op.qubits[0], op.qubits[1]),
        }

    def reset(self, active_qubits: Optional[int] = None):
        """Reset executor state to |0...0⟩."""
        n_q = self.n_qubits if active_qubits is None else active_qubits
        if n_q < 0 or n_q > self.n_qubits:
            raise ValueError(f"active_qubits {n_q} out of range [0, {self.n_qubits}]")
        self._active_qubits = n_q
        self.dim = 2 ** n_q
        self._state = np.zeros(self.dim, dtype=np.complex128)
        self._state[0] = 1.0
        if self._ccce:
            self._ccce.reset()

    def _validate_qubits(self, *qubits: int):
        """Validate that qubit indices are in range [0, _active_qubits) and distinct."""
        for q in qubits:
            if not isinstance(q, (int, np.integer)) or q < 0 or q >= self._active_qubits:
                raise ValueError(f"Qubit {q} out of range [0, {self._active_qubits})")
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"Qubits must be distinct, got {qubits}")

    def _apply_k_qubit_gate(self, gate: np.ndarray, qubits: Tuple[int, ...]):
        """
        Apply a k-qubit unitary to arbitrary distinct wires using big-endian
        tensor-axis permutation (wire q <-> tensor axis q <-> bit shift n-1-q).
        """
        self._validate_qubits(*qubits)
        k = len(qubits)
        expected_dim = 1 << k
        shape = [2] * self._active_qubits
        state_tensor = self._state.reshape(shape)

        remaining = [ax for ax in range(self._active_qubits) if ax not in qubits]
        axes = remaining + list(qubits)

        state_tensor = np.transpose(state_tensor, axes)
        permuted_shape = state_tensor.shape

        state_tensor = state_tensor.reshape(-1, expected_dim) @ gate.T
        state_tensor = state_tensor.reshape(permuted_shape)

        inverse = [0] * self._active_qubits
        for idx, ax in enumerate(axes):
            inverse[ax] = idx
        state_tensor = np.transpose(state_tensor, inverse)

        self._state = state_tensor.reshape(self.dim)
        self._total_gates += 1

    def _apply_single(self, gate: np.ndarray, qubit: int):
        """Apply single-qubit gate."""
        self._apply_k_qubit_gate(gate, (qubit,))

    def _apply_cnot(self, control: int, target: int):
        """Apply CNOT gate."""
        self._apply_two(bond(), control, target)

    def _apply_two(self, gate: np.ndarray, q1: int, q2: int):
        """Apply two-qubit gate."""
        self._apply_k_qubit_gate(gate, (q1, q2))

    def _apply_three(self, gate: np.ndarray, q1: int, q2: int, q3: int):
        """Apply three-qubit gate."""
        self._apply_k_qubit_gate(gate, (q1, q2, q3))

    def _apply_toffoli(self, c1: int, c2: int, target: int):
        """
        Apply Toffoli (CCNOT) gate using the unified big-endian convention.

        Repairs two historical defects:
          1. Double-swap defect: iterating over all basis indices swapped (i, j)
             and subsequently swapped (j, i) back, acting as identity.
          2. Bit-mask vs tensor-axis mismatch: previous bit shifts treated qubit 0
             as LSB while 1Q/2Q tensor operations treated qubit 0 as MSB (axis 0).
        """
        self._apply_three(toffoli(), c1, c2, target)

    def execute_gate(self, op: GateOp):
        """Execute a single gate operation."""
        if op.name == 'barrier':
            return
        if op.name == 'measure':
            return  # Handle separately

        if op.name in self._gate_dispatch:
            self._gate_dispatch[op.name](op)
        else:
            raise ValueError(f"Unknown gate: {op.name}")

    def run(self, circuit: Circuit, shots: int = 1024,
            return_statevector: bool = False) -> ExecutionResult:
        """
        Execute a quantum circuit.

        Args:
            circuit: Circuit to execute
            shots: Number of measurement shots
            return_statevector: Include statevector in result

        Returns:
            ExecutionResult with counts, metrics, etc.
        """
        start_time = time.time()
        self._execution_count += 1

        # Validate circuit
        if circuit.n_qubits > self.n_qubits:
            raise ValueError(
                f"Circuit has {circuit.n_qubits} qubits but executor "
                f"has {self.n_qubits}"
            )

        # Reset state sized to circuit width
        self.reset(active_qubits=circuit.n_qubits)

        # Execute operations
        measurements = []
        for op in circuit.operations:
            if op.name == 'measure':
                measurements.append(op)
            else:
                self.execute_gate(op)

                # Check for healing after each gate
                if self._healing_engine and self._ccce:
                    self._ccce.update_from_quantum_state(self._state)
                    self._state, did_heal = self._ccce.check_and_heal(self._state)

        # Update CCCE metrics
        if self._ccce:
            self._ccce.update_from_quantum_state(self._state)

        # Compute normalized probabilities (guarding against O(1e-15) FP drift)
        probabilities = np.abs(self._state) ** 2
        p_sum = float(np.sum(probabilities))
        if p_sum <= 0.0:
            raise RuntimeError("Statevector norm collapsed to zero")
        probabilities = probabilities / p_sum

        # Sample measurements (respecting explicit measure(qubit, classical_bit) ops if present)
        counts = {}
        if shots > 0:
            samples = np.random.choice(self.dim, size=shots, p=probabilities)
            if measurements:
                cbit_map: Dict[int, int] = {}
                for mop in measurements:
                    q_idx = mop.qubits[0]
                    c_idx = int(round(mop.params[0])) if mop.params else q_idx
                    cbit_map[c_idx] = q_idx
                n_cbits = max(cbit_map.keys()) + 1
                for s in samples:
                    full_bits = format(int(s), f'0{circuit.n_qubits}b')
                    chars = ['0'] * n_cbits
                    for c_idx, q_idx in cbit_map.items():
                        chars[c_idx] = full_bits[q_idx]
                    bitstring = ''.join(chars)
                    counts[bitstring] = counts.get(bitstring, 0) + 1
            else:
                for s in samples:
                    bitstring = format(int(s), f'0{circuit.n_qubits}b')
                    counts[bitstring] = counts.get(bitstring, 0) + 1

        execution_time = time.time() - start_time

        # Build result
        result = ExecutionResult(
            counts=counts,
            probabilities=probabilities if return_statevector else None,
            statevector=self._state.copy() if return_statevector else None,
            execution_time=execution_time,
            shots=shots,
            metrics=self._ccce.state.to_dict() if self._ccce else None,
            metadata={
                'circuit_name': circuit.name,
                'n_qubits': circuit.n_qubits,
                'gate_count': circuit.gate_count(),
                'depth': circuit.depth(),
                'executor': 'SovereignExecutor',
                'mode': self.mode.value
            }
        )

        return result

    def run_vqe_iteration(self, circuit_builder: Callable[[np.ndarray], Circuit],
                          params: np.ndarray) -> float:
        """
        Run single VQE iteration.

        Args:
            circuit_builder: Function that takes params and returns circuit
            params: Current parameters

        Returns:
            Energy expectation value
        """
        circuit = circuit_builder(params)
        result = self.run(circuit, shots=0, return_statevector=True)

        # For VQE, we need an observable expectation
        # This is a simplified version using Z expectation
        probs = result.probabilities
        energy = 0.0
        for i, p in enumerate(probs):
            # Parity of bitstring gives +1/-1
            parity = bin(i).count('1') % 2
            energy += (-1) ** parity * p

        return energy

    # ═══════════════════════════════════════════════════════════════════════════
    # STATE ACCESS
    # ═══════════════════════════════════════════════════════════════════════════

    @property
    def state(self) -> np.ndarray:
        """Get current state vector (copy)."""
        return self._state.copy()

    def get_statevector(self, endian: str = "big") -> np.ndarray:
        """Get current statevector in 'big' (|q_0..q_{n-1}>) or 'little' (|q_{n-1}..q_0>) order."""
        if endian == "big":
            return self._state.copy()
        if endian == "little":
            tensor_state = self._state.reshape([2] * self._active_qubits)
            return np.transpose(tensor_state, list(range(self._active_qubits - 1, -1, -1))).reshape(self.dim).copy()
        raise ValueError(f"endian must be 'big' or 'little', got {endian!r}")

    @property
    def probabilities(self) -> np.ndarray:
        """Get probability distribution."""
        return np.abs(self._state) ** 2

    @property
    def metrics(self) -> Optional[Dict[str, Any]]:
        """Get current CCCE metrics."""
        if self._ccce:
            return self._ccce.state.to_dict()
        return None

    @property
    def is_conscious(self) -> bool:
        """Check if consciousness threshold is met."""
        if self._ccce:
            return self._ccce.is_conscious
        return False

    # ═══════════════════════════════════════════════════════════════════════════
    # TELEMETRY
    # ═══════════════════════════════════════════════════════════════════════════

    def telemetry(self) -> Dict[str, Any]:
        """Get executor telemetry."""
        return {
            'genesis': self._genesis,
            'timestamp': time.time(),
            'uptime': time.time() - self._genesis,
            'n_qubits': self.n_qubits,
            'mode': self.mode.value,
            'executions': self._execution_count,
            'total_gates': self._total_gates,
            'healing_enabled': self.enable_healing,
            'ccce_enabled': self.enable_ccce,
            'ccce': self._ccce.telemetry() if self._ccce else None,
            'healing': self._healing_engine.telemetry() if self._healing_engine else None
        }

    def __repr__(self) -> str:
        return (f"SovereignExecutor(qubits={self.n_qubits}, "
                f"mode={self.mode.value}, "
                f"executions={self._execution_count})")


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_executor(n_qubits: int = 8, **kwargs) -> SovereignExecutor:
    """Factory function for SovereignExecutor."""
    return SovereignExecutor(n_qubits=n_qubits, **kwargs)


def bell_pair_circuit() -> Circuit:
    """Create Bell pair circuit."""
    circuit = Circuit(2, name="BellPair")
    circuit.helix(0).bond(0, 1)
    return circuit


def ghz_circuit(n: int = 3) -> Circuit:
    """Create GHZ state circuit."""
    circuit = Circuit(n, name=f"GHZ_{n}")
    circuit.helix(0)
    for i in range(1, n):
        circuit.bond(0, i)
    return circuit


def qft_circuit(n: int) -> Circuit:
    """Create Quantum Fourier Transform circuit."""
    circuit = Circuit(n, name=f"QFT_{n}")

    for i in range(n):
        circuit.helix(i)
        for j in range(i + 1, n):
            angle = np.pi / (2 ** (j - i))
            circuit.add('cphase', [i, j], [angle])

    # Swap qubits
    for i in range(n // 2):
        circuit.swap_qubits(i, n - 1 - i)

    return circuit


__all__ = [
    'SovereignExecutor', 'ExecutionMode', 'ExecutionResult',
    'Circuit', 'GateOp',
    'create_executor', 'bell_pair_circuit', 'ghz_circuit', 'qft_circuit'
]
