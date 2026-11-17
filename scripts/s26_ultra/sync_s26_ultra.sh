#!/usr/bin/env bash
# =============================================================================
# OSIRIS Samsung Galaxy S26 Ultra (Snapdragon 8 Elite / Oryon) Workspace Sync
# =============================================================================
# Navigates Android UID sandboxing and Scoped Storage noexec cleanly:
#   - Uses `adb forward tcp:8022 tcp:8022` + Termux OpenSSH (`sshd -p 8022`)
#   - Syncs directly into Termux private app storage (`~/src/osiris-cli`),
#     NEVER `/sdcard/` (FUSE noexec) and NEVER via `adb shell pkg` (UID 2000).
#   - Axiom (SYS-021): Cross-node transfer is a data operation, not an
#     authorization grant.
# =============================================================================
set -euo pipefail

SSH_PORT="${OSIRIS_TERMUX_SSH_PORT:-8022}"
LOCAL_REPO="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
REMOTE_DIR="${2:-~/src/osiris-cli}"

ADB_BIN="$(command -v adb || true)"
if [[ -z "$ADB_BIN" && -x "/mnt/c/Users/there/AppData/Local/Android/Sdk/platform-tools/adb.exe" ]]; then
    ADB_BIN="/mnt/c/Users/there/AppData/Local/Android/Sdk/platform-tools/adb.exe"
fi

echo "================================================================="
echo " OSIRIS S26 ULTRA WORKSPACE SYNCHRONIZATION (GOVERNED SSH/ADB)"
echo " Source: ${LOCAL_REPO}"
echo " Target: localhost:${SSH_PORT} -> ${REMOTE_DIR} (Termux private fs)"
echo "================================================================="

if [[ -n "$ADB_BIN" ]]; then
    echo "[1/3] Checking connected ADB devices and forwarding tcp:${SSH_PORT}..."
    "$ADB_BIN" devices | grep -E "[[:space:]]device$" >/dev/null || {
        echo "[!] No authorized ADB device detected. Pair Wireless Debugging or USB ADB first."
        exit 1
    }
    "$ADB_BIN" forward "tcp:${SSH_PORT}" "tcp:${SSH_PORT}"
else
    echo "[1/3] Local adb binary not in PATH; assuming tcp:${SSH_PORT} is already forwarded or using Bluetooth PAN."
fi

echo "[2/3] Verifying Termux OpenSSH listener on 127.0.0.1:${SSH_PORT}..."
ssh -p "${SSH_PORT}" -o BatchMode=yes -o ConnectTimeout=5 localhost "mkdir -p ${REMOTE_DIR}" || {
    echo "[!] Cannot reach Termux sshd on port ${SSH_PORT}."
    echo "    In Termux on your S26 Ultra, run: pkg install -y openssh rsync python && sshd -p ${SSH_PORT}"
    exit 1
}

echo "[3/3] Synchronizing repository via rsync (excluding .git worktree pointers, venvs, secrets)..."
rsync -avz --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '.pytest_cache' \
    --exclude 'venv*' \
    --exclude '.env*' \
    --exclude '*.pem' \
    --exclude '*.key' \
    --exclude 'results/' \
    -e "ssh -p ${SSH_PORT}" \
    "${LOCAL_REPO}/" "localhost:${REMOTE_DIR}/"

echo "[+] Synchronization complete. Run inside Termux on S26 Ultra:"
echo "    cd ${REMOTE_DIR} && bash scripts/s26_ultra/run_cartographer_locked.sh"
