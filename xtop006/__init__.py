"""OSIRIS-XTOP-006: Tunable Non-Clifford Scrambling & [[4,2,2]] Codespace QCCD Recovery on Quantinuum H2.

Modules:
  * `kraus_channel`: Exact analytical 7-qubit Hayden-Preskill / Landsman-Figgatt-Yoshida-Kitaev
    remnant Kraus contraction `(M_r(theta))_{xa}`, 4-Bell CPTP completeness, 1-parameter tunable
    non-Clifford 3-qubit scrambler `U(theta)`, and the Scrambling-vs-Decoherence Discriminator.
  * `qccd_circuits`: Offline `QuantinuumAPIOffline` (`H2-1`) `pytket` circuit generator and noisy
    QCCD simulator for Arm `P0` (`PHYSICAL_7Q`, 7Q) vs Arm `E1` (`ENCODED_422_HYBRID`, 12Q with
    verified `[[4,2,2]]` prep and flagged weight-4 `X`/`Z` syndrome checks).
"""

from .kraus_channel import (
    BELL_OUTCOMES,
    INPUT_STATES_6,
    THETA_GRID_DEFAULT,
    ChannelEvaluation,
    analytical_fidelity_6state,
    analytical_p_phi_plus,
    apply_bell_channel,
    compute_all_bell_kraus_channels,
    compute_remnant_kraus_operators,
    is_non_clifford_unitary,
    landsman_figgatt_scrambler,
    scrambling_vs_decoherence_discriminator,
    verify_cptp_completeness,
)

__all__ = [
    "BELL_OUTCOMES",
    "INPUT_STATES_6",
    "THETA_GRID_DEFAULT",
    "ChannelEvaluation",
    "analytical_fidelity_6state",
    "analytical_p_phi_plus",
    "apply_bell_channel",
    "compute_all_bell_kraus_channels",
    "compute_remnant_kraus_operators",
    "is_non_clifford_unitary",
    "landsman_figgatt_scrambler",
    "scrambling_vs_decoherence_discriminator",
    "verify_cptp_completeness",
]
