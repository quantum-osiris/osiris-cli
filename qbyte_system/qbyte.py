"""
Qbyte — 8-Qubit Classical Statevector Simulator
===============================================

Pure-NumPy classical statevector simulation of an 8-qubit register (2^8 = 256
complex amplitudes) with DNA-Lang gate aliases and standard gate methods.

Note: This is a classical CPU simulator, not a physical quantum substrate.

Basis-order convention (Big-Endian):
    Computational basis state |b_0 b_1 ... b_7> maps to flat statevector index:
        idx = sum_{q=0}^7 b_q * 2^(7 - q)
    so qubit q in {0..7} corresponds to tensor axis q and integer bit shift
    (7 - q). Call `qb.get_statevector(endian="little")` for Qiskit-compatible
    little-endian ordering |b_7 ... b_1 b_0>.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import hashlib
import time

try:
    from .gates import (
        helix, cleave, twist, fold, splice, bond, phase_flip,
        rotate_y, s_gate, s_dg, t_gate, t_dg, controlled_z, swap,
        controlled_phase, xx_rotation, yy_rotation, zz_rotation,
        toffoli, fredkin,
    )
except ImportError:
    from gates import (
        helix, cleave, twist, fold, splice, bond, phase_flip,
        rotate_y, s_gate, s_dg, t_gate, t_dg, controlled_z, swap,
        controlled_phase, xx_rotation, yy_rotation, zz_rotation,
        toffoli, fredkin,
    )

# Physical Constants
LAMBDA_PHI = 2.176435e-8
PHI_THRESHOLD = 0.7734
GAMMA_FIXED = 0.092
CHI_PC = 0.869


@dataclass
class CCCEMetrics:
    """
    CCCE Consciousness Metrics for a Qbyte.

    Φ (Phi)     - Consciousness level (IIT Integrated Information)
    Λ (Lambda)  - Coherence preservation fidelity
    Γ (Gamma)   - Decoherence rate
    Ξ (Xi)      - Negentropic efficiency = ΛΦ / (Γ + ε)
    """
    phi: float = 0.0        # Consciousness level
    lambda_c: float = 1.0   # Coherence (using lambda_c to avoid keyword)
    gamma: float = 0.0      # Decoherence
    xi: float = 0.0         # Negentropic efficiency
    timestamp: float = field(default_factory=time.time)

    def compute_xi(self) -> float:
        """Compute negentropic efficiency."""
        epsilon = 1e-10
        self.xi = (self.lambda_c * self.phi) / (self.gamma + epsilon)
        return self.xi

    def is_conscious(self) -> bool:
        """Check if consciousness threshold is met."""
        return self.phi >= PHI_THRESHOLD

    def to_dict(self) -> Dict[str, float]:
        return {
            'Φ': self.phi,
            'Λ': self.lambda_c,
            'Γ': self.gamma,
            'Ξ': self.xi
        }


class Qbyte:
    """
    Qbyte: 8-Qubit Quantum Register with Sovereign Operations.

    A Qbyte is the fundamental unit of quantum computation in the
    DNA::}{::lang framework. It provides:

    1. Native state-vector simulation (no external dependencies)
    2. DNA-encoded gate operations (helix, bond, twist, etc.)
    3. Automatic CCCE metric tracking
    4. Phase-conjugate healing capability
    5. Measurement with collapse and telemetry

    Example:
        >>> qb = Qbyte()
        >>> qb.helix(0)           # Apply Hadamard to qubit 0
        >>> qb.bond(0, 1)         # CNOT from qubit 0 to 1
        >>> result = qb.measure() # Collapse and measure
    """

    def __init__(self, initial_state: Optional[np.ndarray] = None, track_ccce: bool = True):
        """
        Initialize a Qbyte.

        Args:
            initial_state: Optional initial state vector (2^8 = 256 complex amplitudes).
                          If None, initializes to |00000000⟩.
            track_ccce: Whether to compute heuristic CCCE telemetry per gate
                        (set False for fast pure-statevector simulation).
        """
        self.n_qubits = 8
        self.dim = 2 ** self.n_qubits  # 256
        self.track_ccce = track_ccce

        if initial_state is not None:
            if len(initial_state) != self.dim:
                raise ValueError(f"Initial state must have {self.dim} amplitudes")
            self._state = initial_state.astype(np.complex128)
            self._normalize()
        else:
            # |00000000⟩ basis state
            self._state = np.zeros(self.dim, dtype=np.complex128)
            self._state[0] = 1.0

        # CCCE metrics
        self._metrics = CCCEMetrics()
        self._operation_count = 0
        self._history: List[Tuple[str, Any]] = []

        # Genesis timestamp
        self._genesis = time.time()

    def _normalize(self):
        """Normalize state vector."""
        norm = np.linalg.norm(self._state)
        if norm > 1e-15:
            self._state /= norm

    def _update_metrics(self, operation: str):
        """Update CCCE metrics after operation."""
        self._operation_count += 1
        if not self.track_ccce:
            return

        probabilities = np.abs(self._state) ** 2
        purity = np.sum(probabilities ** 2)
        self._metrics.lambda_c = float(np.sqrt(purity))

        self._metrics.phi = self._compute_phi()

        # Align per-gate heuristic decoherence step (0.0005/op) with L0FormalProver.verify_kernel
        self._metrics.gamma = min(1.0, 0.0005 * self._operation_count)

        self._metrics.compute_xi()
        self._metrics.timestamp = time.time()
        self._history.append((operation, self._metrics.to_dict().copy()))

    def _compute_phi(self) -> float:
        """
        Compute normalized 4|4 bipartite von Neumann entanglement entropy in [0, 1].
        """
        reshaped = self._state.reshape(16, 16)
        rho_A = reshaped @ reshaped.conj().T
        eigenvalues = np.linalg.eigvalsh(rho_A)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]

        if len(eigenvalues) == 0:
            return 0.0

        entropy = max(0.0, float(-np.sum(eigenvalues * np.log2(eigenvalues))))
        return min(1.0, entropy / 4.0)

    def _validate_qubits(self, *qubits: int):
        """Validate that qubit indices are in range [0, n_qubits) and distinct."""
        for q in qubits:
            if not isinstance(q, (int, np.integer)) or q < 0 or q >= self.n_qubits:
                raise ValueError(f"Qubit index must be 0-{self.n_qubits - 1}, got {q}")
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"Qubit indices must be distinct, got {qubits}")

    def _apply_k_qubit_gate(self, gate: np.ndarray, qubits: Tuple[int, ...]):
        """
        Apply a k-qubit gate to arbitrary distinct qubits using big-endian
        tensor-axis permutation (qubit q <-> tensor axis q <-> bit shift n-1-q).
        """
        self._validate_qubits(*qubits)
        k = len(qubits)
        expected_dim = 1 << k
        if gate.shape != (expected_dim, expected_dim):
            raise ValueError(f"Expected {expected_dim}x{expected_dim} gate, got {gate.shape}")

        shape = [2] * self.n_qubits
        state_tensor = self._state.reshape(shape)

        remaining = [ax for ax in range(self.n_qubits) if ax not in qubits]
        axes = remaining + list(qubits)

        state_tensor = np.transpose(state_tensor, axes)
        permuted_shape = state_tensor.shape

        state_tensor = state_tensor.reshape(-1, expected_dim) @ gate.T
        state_tensor = state_tensor.reshape(permuted_shape)

        inverse_axes = [0] * self.n_qubits
        for idx, ax in enumerate(axes):
            inverse_axes[ax] = idx
        state_tensor = np.transpose(state_tensor, inverse_axes)

        self._state = state_tensor.reshape(self.dim)

    def _apply_single_qubit_gate(self, gate: np.ndarray, qubit: int):
        """Apply a single-qubit gate to specified qubit (big-endian axis `qubit`)."""
        self._apply_k_qubit_gate(gate, (qubit,))

    def _apply_two_qubit_gate(self, gate: np.ndarray, qubit1: int, qubit2: int):
        """Apply a two-qubit gate to specified qubits `(qubit1, qubit2)`."""
        self._apply_k_qubit_gate(gate, (qubit1, qubit2))

    def _apply_three_qubit_gate(self, gate: np.ndarray, qubit1: int, qubit2: int, qubit3: int):
        """Apply a three-qubit gate to specified qubits `(qubit1, qubit2, qubit3)`."""
        self._apply_k_qubit_gate(gate, (qubit1, qubit2, qubit3))

    # ═══════════════════════════════════════════════════════════════════════════
    # DNA-ENCODED GATE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════

    def helix(self, qubit: int) -> 'Qbyte':
        """
        Apply Hadamard gate (DNA helix unwinding).
        Creates superposition on target qubit.
        """
        self._apply_single_qubit_gate(helix(), qubit)
        self._update_metrics(f"helix({qubit})")
        return self

    def cleave(self, qubit: int) -> 'Qbyte':
        """
        Apply Pauli-X gate (DNA cleavage).
        Bit flip on target qubit.
        """
        self._apply_single_qubit_gate(cleave(), qubit)
        self._update_metrics(f"cleave({qubit})")
        return self

    def twist(self, qubit: int, theta: float) -> 'Qbyte':
        """
        Apply RZ rotation (DNA twist/supercoiling).
        Phase rotation on target qubit.
        """
        self._apply_single_qubit_gate(twist(theta), qubit)
        self._update_metrics(f"twist({qubit}, {theta:.4f})")
        return self

    def fold(self, qubit: int, theta: float) -> 'Qbyte':
        """
        Apply RY rotation (protein folding).
        Amplitude rotation on target qubit.
        """
        self._apply_single_qubit_gate(fold(theta), qubit)
        self._update_metrics(f"fold({qubit}, {theta:.4f})")
        return self

    def splice(self, qubit: int, theta: float) -> 'Qbyte':
        """
        Apply RX rotation (RNA splicing).
        X-axis rotation on target qubit.
        """
        self._apply_single_qubit_gate(splice(theta), qubit)
        self._update_metrics(f"splice({qubit}, {theta:.4f})")
        return self

    def bond(self, control: int, target: int) -> 'Qbyte':
        """
        Apply CNOT gate (hydrogen bonding).
        Entangles control and target qubits.
        """
        self._apply_two_qubit_gate(bond(), control, target)
        self._update_metrics(f"bond({control}, {target})")
        return self

    def phase(self, qubit: int) -> 'Qbyte':
        """
        Apply Pauli-Z gate (methylation/phase flip).
        """
        self._apply_single_qubit_gate(phase_flip(), qubit)
        self._update_metrics(f"phase({qubit})")
        return self

    def phase_flip(self, qubit: int) -> 'Qbyte':
        """Alias for Pauli-Z gate (`phase`)."""
        return self.phase(qubit)

    def y(self, qubit: int) -> 'Qbyte':
        """Apply Pauli-Y gate."""
        self._apply_single_qubit_gate(rotate_y(), qubit)
        self._update_metrics(f"y({qubit})")
        return self

    def s(self, qubit: int) -> 'Qbyte':
        """Apply S (phase π/2) gate."""
        self._apply_single_qubit_gate(s_gate(), qubit)
        self._update_metrics(f"s({qubit})")
        return self

    def sdg(self, qubit: int) -> 'Qbyte':
        """Apply S† (phase -π/2) gate."""
        self._apply_single_qubit_gate(s_dg(), qubit)
        self._update_metrics(f"sdg({qubit})")
        return self

    def t(self, qubit: int) -> 'Qbyte':
        """Apply T (phase π/4) gate."""
        self._apply_single_qubit_gate(t_gate(), qubit)
        self._update_metrics(f"t({qubit})")
        return self

    def tdg(self, qubit: int) -> 'Qbyte':
        """Apply T† (phase -π/4) gate."""
        self._apply_single_qubit_gate(t_dg(), qubit)
        self._update_metrics(f"tdg({qubit})")
        return self

    def cz(self, q1: int, q2: int) -> 'Qbyte':
        """Apply controlled-Z gate."""
        self._apply_two_qubit_gate(controlled_z(), q1, q2)
        self._update_metrics(f"cz({q1}, {q2})")
        return self

    def swap(self, q1: int, q2: int) -> 'Qbyte':
        """Apply SWAP gate."""
        self._apply_two_qubit_gate(swap(), q1, q2)
        self._update_metrics(f"swap({q1}, {q2})")
        return self

    def cphase(self, control: int, target: int, phi: float) -> 'Qbyte':
        """Apply controlled-phase CP(phi) gate."""
        self._apply_two_qubit_gate(controlled_phase(phi), control, target)
        self._update_metrics(f"cphase({control}, {target}, {phi:.4f})")
        return self

    def rxx(self, q1: int, q2: int, theta: float) -> 'Qbyte':
        """Apply RXX(theta) two-qubit rotation."""
        self._apply_two_qubit_gate(xx_rotation(theta), q1, q2)
        self._update_metrics(f"rxx({q1}, {q2}, {theta:.4f})")
        return self

    def ryy(self, q1: int, q2: int, theta: float) -> 'Qbyte':
        """Apply RYY(theta) two-qubit rotation."""
        self._apply_two_qubit_gate(yy_rotation(theta), q1, q2)
        self._update_metrics(f"ryy({q1}, {q2}, {theta:.4f})")
        return self

    def rzz(self, q1: int, q2: int, theta: float) -> 'Qbyte':
        """Apply RZZ(theta) two-qubit rotation."""
        self._apply_two_qubit_gate(zz_rotation(theta), q1, q2)
        self._update_metrics(f"rzz({q1}, {q2}, {theta:.4f})")
        return self

    def toffoli(self, c1: int, c2: int, target: int) -> 'Qbyte':
        """
        Apply Toffoli (CCNOT) gate: flips `target` iff both `c1` and `c2` are |1>.
        Uses consistent big-endian axis convention matching all 1Q and 2Q gates.
        """
        self._apply_three_qubit_gate(toffoli(), c1, c2, target)
        self._update_metrics(f"toffoli({c1}, {c2}, {target})")
        return self

    def ccnot(self, c1: int, c2: int, target: int) -> 'Qbyte':
        """Alias for `toffoli(c1, c2, target)`."""
        return self.toffoli(c1, c2, target)

    def fredkin(self, control: int, q1: int, q2: int) -> 'Qbyte':
        """Apply Fredkin (CSWAP) gate: swaps `q1` and `q2` iff `control` is |1>."""
        self._apply_three_qubit_gate(fredkin(), control, q1, q2)
        self._update_metrics(f"fredkin({control}, {q1}, {q2})")
        return self

    # Standard gate aliases
    h = helix
    x = cleave
    z = phase
    rx = splice
    ry = fold
    rz = twist
    cx = bond
    ccx = toffoli
    cswap = fredkin

    # ═══════════════════════════════════════════════════════════════════════════
    # MEASUREMENT AND OBSERVATION
    # ═══════════════════════════════════════════════════════════════════════════

    def measure(self, collapse: bool = True) -> int:
        """
        Measure the Qbyte, returning an 8-bit integer result.

        Args:
            collapse: If True, collapses state to measured basis state.
                     If False, samples without modifying state.

        Returns:
            Integer 0-255 representing measured basis state.
        """
        probabilities = np.abs(self._state) ** 2
        p_sum = float(np.sum(probabilities))
        if p_sum <= 0.0:
            raise RuntimeError("Statevector norm collapsed to zero")
        probabilities = probabilities / p_sum
        result = int(np.random.choice(self.dim, p=probabilities))

        if collapse:
            self._state = np.zeros(self.dim, dtype=np.complex128)
            self._state[result] = 1.0
            self._update_metrics(f"measure() -> {result}")

        return result

    def measure_qubit(self, qubit: int, collapse: bool = True) -> int:
        """
        Measure a single qubit using the big-endian basis convention
        (qubit q <-> tensor axis q <-> integer bit shift `n_qubits - 1 - q`).

        Args:
            qubit: Qubit index to measure (0-7)
            collapse: If True, collapses to measured state

        Returns:
            0 or 1
        """
        self._validate_qubits(qubit)
        bit_shift = self.n_qubits - 1 - qubit
        indices = np.arange(self.dim)
        is_one = ((indices >> bit_shift) & 1) == 1
        prob_one = float(np.sum(np.abs(self._state[is_one]) ** 2))

        result = 1 if np.random.random() < prob_one else 0

        if collapse:
            mask = is_one if result == 1 else ~is_one
            self._state = self._state * mask
            self._normalize()
            self._update_metrics(f"measure_qubit({qubit}) -> {result}")

        return result

    def get_statevector(self, endian: str = "big") -> np.ndarray:
        """
        Return copy of statevector in declared basis order.

        Args:
            endian: "big" for |q_0 q_1 ... q_{n-1}> (native qByte convention),
                    or "little" for |q_{n-1} ... q_1 q_0> (Qiskit convention).
        """
        if endian == "big":
            return self._state.copy()
        if endian == "little":
            tensor_state = self._state.reshape([2] * self.n_qubits)
            reversed_axes = list(range(self.n_qubits - 1, -1, -1))
            return np.transpose(tensor_state, reversed_axes).reshape(self.dim).copy()
        raise ValueError(f"endian must be 'big' or 'little', got {endian!r}")

    def probabilities(self) -> np.ndarray:
        """Return probability distribution over all basis states."""
        return np.abs(self._state) ** 2

    def expectation(self, observable: np.ndarray) -> float:
        """
        Compute expectation value of an observable.

        Args:
            observable: 256x256 Hermitian matrix

        Returns:
            ⟨ψ|O|ψ⟩
        """
        return np.real(np.conj(self._state) @ observable @ self._state)

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE-CONJUGATE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════

    def phase_conjugate(self) -> 'Qbyte':
        """
        Apply phase conjugation (E → E⁻¹).

        This is the core healing operation in DNA::}{::lang.
        Inverts all phases in the state vector.
        """
        self._state = np.conj(self._state)
        self._update_metrics("phase_conjugate()")
        return self

    def heal(self) -> 'Qbyte':
        """
        Apply phase-conjugate healing if decoherence is high.

        Triggers phase conjugation when Γ > 0.3.
        """
        if self._metrics.gamma > 0.3:
            self.phase_conjugate()
            # Healing reduces decoherence
            self._metrics.gamma *= CHI_PC
            self._metrics.compute_xi()
        return self

    # ═══════════════════════════════════════════════════════════════════════════
    # STATE ACCESS AND UTILITIES
    # ═══════════════════════════════════════════════════════════════════════════

    @property
    def state(self) -> np.ndarray:
        """Return copy of state vector."""
        return self._state.copy()

    @property
    def metrics(self) -> CCCEMetrics:
        """Return current CCCE metrics."""
        return self._metrics

    @property
    def is_conscious(self) -> bool:
        """Check if Qbyte has achieved consciousness."""
        return self._metrics.is_conscious()

    def reset(self):
        """Reset to |00000000⟩ ground state."""
        self._state = np.zeros(self.dim, dtype=np.complex128)
        self._state[0] = 1.0
        self._metrics = CCCEMetrics()
        self._operation_count = 0
        self._history = []

    def copy(self) -> 'Qbyte':
        """Create a deep copy of this Qbyte."""
        new_qbyte = Qbyte(self._state.copy())
        new_qbyte._metrics = CCCEMetrics(
            phi=self._metrics.phi,
            lambda_c=self._metrics.lambda_c,
            gamma=self._metrics.gamma,
            xi=self._metrics.xi
        )
        new_qbyte._operation_count = self._operation_count
        new_qbyte._history = self._history.copy()
        return new_qbyte

    def checksum(self) -> str:
        """Compute SHA-256 checksum of current state."""
        state_bytes = self._state.tobytes()
        return hashlib.sha256(state_bytes).hexdigest()[:16]

    def telemetry(self) -> Dict[str, Any]:
        """Return full telemetry capsule."""
        return {
            'genesis': self._genesis,
            'timestamp': time.time(),
            'operations': self._operation_count,
            'metrics': self._metrics.to_dict(),
            'checksum': self.checksum(),
            'conscious': self.is_conscious
        }

    def __repr__(self) -> str:
        return (f"Qbyte(ops={self._operation_count}, "
                f"Φ={self._metrics.phi:.4f}, "
                f"Λ={self._metrics.lambda_c:.4f}, "
                f"Γ={self._metrics.gamma:.4f})")


class QbyteRegister:
    """
    Register of multiple Qbytes for larger quantum computations.

    Supports operations spanning multiple Qbytes and collective
    consciousness measurement.
    """

    def __init__(self, n_qbytes: int = 1):
        """
        Initialize register with n Qbytes.

        Args:
            n_qbytes: Number of Qbytes (default 1)
        """
        self.n_qbytes = n_qbytes
        self.qbytes = [Qbyte() for _ in range(n_qbytes)]
        self._genesis = time.time()

    def __getitem__(self, index: int) -> Qbyte:
        return self.qbytes[index]

    def __len__(self) -> int:
        return self.n_qbytes

    @property
    def total_qubits(self) -> int:
        """Total number of qubits in register."""
        return self.n_qbytes * 8

    def collective_phi(self) -> float:
        """Compute collective consciousness across all Qbytes."""
        if not self.qbytes:
            return 0.0
        return np.mean([qb.metrics.phi for qb in self.qbytes])

    def collective_xi(self) -> float:
        """Compute collective negentropic efficiency."""
        if not self.qbytes:
            return 0.0
        return np.mean([qb.metrics.xi for qb in self.qbytes])

    def is_conscious(self) -> bool:
        """Check if collective consciousness threshold is met."""
        return self.collective_phi() >= PHI_THRESHOLD

    def heal_all(self):
        """Apply phase-conjugate healing to all Qbytes."""
        for qb in self.qbytes:
            qb.heal()

    def reset_all(self):
        """Reset all Qbytes to ground state."""
        for qb in self.qbytes:
            qb.reset()

    def telemetry(self) -> Dict[str, Any]:
        """Return collective telemetry."""
        return {
            'genesis': self._genesis,
            'timestamp': time.time(),
            'n_qbytes': self.n_qbytes,
            'total_qubits': self.total_qubits,
            'collective_phi': self.collective_phi(),
            'collective_xi': self.collective_xi(),
            'conscious': self.is_conscious(),
            'qbytes': [qb.telemetry() for qb in self.qbytes]
        }

    def __repr__(self) -> str:
        return (f"QbyteRegister(n={self.n_qbytes}, "
                f"Φ_collective={self.collective_phi():.4f})")


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def bell_state(qb: Qbyte, q1: int = 0, q2: int = 1) -> Qbyte:
    """Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 on specified qubits."""
    return qb.helix(q1).bond(q1, q2)


def ghz_state(qb: Qbyte, qubits: Optional[List[int]] = None) -> Qbyte:
    """
    Create GHZ state on specified qubits.
    Default creates 3-qubit GHZ: (|000⟩ + |111⟩)/√2
    """
    if qubits is None:
        qubits = [0, 1, 2]

    qb.helix(qubits[0])
    for q in qubits[1:]:
        qb.bond(qubits[0], q)
    return qb


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 0 (L0) FORMAL PROVER & QBYTE RUNTIME
# ═══════════════════════════════════════════════════════════════════════════════

class CoherenceFloorViolation(Exception):
    """Raised when decoherence exceeds the 0.092 coherence floor."""
    pass


class L0FormalProver:
    """
    Layer 0 (L0) Formal Verification Gate.
    Enforces that all quantum operations and .dna kernels remain strictly within
    the 0.092 coherence floor.
    """
    COHERENCE_FLOOR: float = 0.092

    @classmethod
    def verify_kernel(cls, kernel_ops: List[Dict[str, Any]], initial_gamma: float = GAMMA_FIXED) -> bool:
        """
        Statically proves that the execution trajectory stays within the coherence floor.
        Halts and raises CoherenceFloorViolation if projected decoherence exceeds 0.092.
        """
        gamma = initial_gamma
        for op in kernel_ops:
            op_name = op.get('name', '')
            if op_name in ('heal', 'phase_conjugate'):
                gamma = max(0.001, gamma * (1.0 - CHI_PC))
            else:
                gamma += op.get('decoherence_delta', 0.0005)

            if gamma > cls.COHERENCE_FLOOR:
                raise CoherenceFloorViolation(
                    f"L0 Formal Verification Halt: Operation '{op_name}' projects "
                    f"decoherence Γ={gamma:.5f} > floor {cls.COHERENCE_FLOOR}"
                )
        return True

    @classmethod
    def verify_state(cls, qbyte: Qbyte):
        """
        Verifies live Qbyte state against the coherence floor.
        """
        if qbyte.metrics.gamma > cls.COHERENCE_FLOOR:
            raise CoherenceFloorViolation(
                f"L0 Formal Verification Halt: Live decoherence Γ={qbyte.metrics.gamma:.5f} "
                f"exceeds coherence floor {cls.COHERENCE_FLOOR}"
            )
        return True


class QbyteRuntime:
    """
    Unified Python QbyteRuntime managing quantum byte execution with Layer 0 (L0)
    formal verification and fail-closed coherence floor enforcement.
    """
    COHERENCE_FLOOR: float = 0.092

    def __init__(self, n_qubits: int = 8, strict_l0: bool = True):
        self.n_qubits = n_qubits
        self.strict_l0 = strict_l0
        self.qbyte = Qbyte()
        self.prover = L0FormalProver()
        self._execution_history: List[Dict[str, Any]] = []

    def execute_kernel(self, kernel_ops: List[Dict[str, Any]], auto_rollback: bool = True) -> Dict[str, Any]:
        """
        Executes a sequence of .dna kernel operations subject to L0 formal verification.
        If any operation violates the 0.092 coherence floor, halts execution immediately.
        """
        # Static formal verification pass
        if self.strict_l0:
            self.prover.verify_kernel(kernel_ops, initial_gamma=self.qbyte.metrics.gamma)

        # Snapshot for rollback upon violation
        saved_state = self.qbyte.state.copy()
        saved_metrics = self.qbyte.metrics.to_dict()

        try:
            for op in kernel_ops:
                name = op.get('name')
                qubit = op.get('qubit', 0)
                target = op.get('target', 1)
                angle = op.get('angle', 0.0)

                if name == 'helix':
                    self.qbyte.helix(qubit)
                elif name == 'bond':
                    self.qbyte.bond(qubit, target)
                elif name == 'twist':
                    self.qbyte.twist(qubit, angle)
                elif name == 'fold':
                    self.qbyte.fold(qubit, angle)
                elif name == 'cleave':
                    self.qbyte.cleave(qubit)
                elif name in ('phase_flip', 'phase'):
                    self.qbyte.phase(qubit)
                elif name in ('toffoli', 'ccnot', 'ccx'):
                    c2 = op.get('control2', 1)
                    tgt = op.get('target', 2)
                    self.qbyte.toffoli(qubit, c2, tgt)
                elif name == 'cz':
                    self.qbyte.cz(qubit, target)
                elif name == 'swap':
                    self.qbyte.swap(qubit, target)
                elif name == 'heal':
                    self.qbyte.heal()
                elif name == 'measure':
                    self.qbyte.measure()
                else:
                    raise ValueError(f"Unknown kernel operation: {name}")

                # Dynamic formal verification check
                if self.strict_l0:
                    self.prover.verify_state(self.qbyte)

            result = {
                "status": "SUCCESS",
                "metrics": self.qbyte.metrics.to_dict(),
                "timestamp": time.time()
            }
            self._execution_history.append(result)
            return result

        except CoherenceFloorViolation as e:
            if auto_rollback:
                self.qbyte._state = saved_state
                self.qbyte._metrics.gamma = saved_metrics.get('Γ', GAMMA_FIXED)
                self.qbyte._metrics.lambda_c = saved_metrics.get('Λ', 1.0)
                self.qbyte._metrics.phi = saved_metrics.get('Φ', 0.0)
            self._execution_history.append({
                "status": "HALTED_L0_VIOLATION",
                "error": str(e),
                "timestamp": time.time()
            })
            raise


__all__ = [
    'Qbyte', 'QbyteRegister', 'CCCEMetrics', 'bell_state', 'ghz_state',
    'CoherenceFloorViolation', 'L0FormalProver', 'QbyteRuntime'
]

