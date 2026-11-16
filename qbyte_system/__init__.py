"""
qByte Classical Statevector Simulator Package
=============================================

Pure-NumPy classical statevector simulator for small quantum circuits using
DNA-Lang gate aliases alongside standard quantum gate names.

Note on physical scope:
    This package is a deterministic/Monte-Carlo classical statevector simulator
    executing on a conventional CPU via NumPy. It is NOT an independent physical
    quantum substrate, and its optional phase-conjugate "healing" routines are
    non-unitary classical heuristics disabled by default during unitary circuit
    simulation.

Basis-order convention (Big-Endian):
    For an n-qubit statevector of length 2^n, computational basis state
    |b_0 b_1 ... b_{n-1}> maps to integer index:
        idx = sum_{q=0}^{n-1} b_q * 2^(n - 1 - q)
    so qubit 0 is the most-significant bit (tensor axis 0, bit shift n - 1) and
    qubit n - 1 is the least-significant bit (tensor axis n - 1, bit shift 0).
    Use `.get_statevector(endian="little")` to obtain Qiskit-compatible
    little-endian ordering |b_{n-1} ... b_1 b_0>.
"""

__version__ = "1.0.0"
__author__ = "Devin Phillip Davis"
__organization__ = "Agile Defense Systems LLC"

# Physical Constants (IMMUTABLE - empirically validated)
LAMBDA_PHI = 2.176435e-8      # ΛΦ Universal Memory Constant [s⁻¹]
THETA_LOCK = 51.843           # θ_lock Torsion-locked angle [degrees]
PHI_THRESHOLD = 0.7734        # Φ IIT Consciousness Threshold
GAMMA_FIXED = 0.092           # Γ Fixed-point decoherence
CHI_PC = 0.869                # χ_pc Phase conjugate coupling
GOLDEN_RATIO = 1.618033988749895  # φ Golden ratio

# Re-export all components
from .qbyte import Qbyte, QbyteRegister
from .phase_conjugate import PhaseConjugateEngine
from .ccce_runtime import CCCERuntime
from .sovereign_executor import SovereignExecutor
from .gates import *

__all__ = [
    # Constants
    'LAMBDA_PHI', 'THETA_LOCK', 'PHI_THRESHOLD', 'GAMMA_FIXED',
    'CHI_PC', 'GOLDEN_RATIO',
    # Classes
    'Qbyte', 'QbyteRegister', 'PhaseConjugateEngine',
    'CCCERuntime', 'SovereignExecutor',
]
