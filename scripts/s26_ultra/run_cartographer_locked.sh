#!/usr/bin/env bash
# =============================================================================
# OSIRIS Prime-Core-Locked Cartographer Wrapper (Galaxy S26 Ultra / ARM64)
# =============================================================================
# Dynamically detects the Prime CPU cluster via Linux cpufreq sysfs
# (cores 6,7 on Snapdragon 8 Elite 2+6 Oryon topology) instead of hardcoding
# a 4+4 mask, and runs the governed SHA-256 hash-chained AST cartographer.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET_DIR="${1:-${REPO_ROOT}}"
LEDGER_OUT="${2:-${HOME}/.osiris/semantic_ledger.jsonl}"

PRIME_CORES="$(PYTHONPATH="${REPO_ROOT}" python3 -c '
from osiris_cli.s26_cartographer import detect_prime_cores
print(",".join(str(c) for c in detect_prime_cores()))
')"

echo "================================================================="
echo " OSIRIS GOVERNED AST CARTOGRAPHER (PRIME CORES: ${PRIME_CORES})"
echo " Target: ${TARGET_DIR}"
echo " Ledger: ${LEDGER_OUT}"
echo "================================================================="

if command -v taskset >/dev/null 2>&1 && [[ -n "${PRIME_CORES}" ]]; then
    PYTHONPATH="${REPO_ROOT}" taskset -c "${PRIME_CORES}" \
        python3 -m osiris_cli.s26_cartographer scan "${TARGET_DIR}" --ledger "${LEDGER_OUT}" --lock-prime
else
    PYTHONPATH="${REPO_ROOT}" \
        python3 -m osiris_cli.s26_cartographer scan "${TARGET_DIR}" --ledger "${LEDGER_OUT}" --lock-prime
fi
