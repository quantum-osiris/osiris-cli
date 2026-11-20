"""Pytest entrypoint mirroring phase2/correlated_noise_01/test_correlated_noise_models.py."""

import pytest

pytest.importorskip("qiskit")

from phase2.correlated_noise_01.test_correlated_noise_models import *  # noqa: F401,F403,E402
