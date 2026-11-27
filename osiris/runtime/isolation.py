"""OS resource-limit subprocess backend for untrusted commands.

bubblewrap and nsjail are not required and are not assumed to be installed.
This backend runs the command as a separate process with POSIX rlimits
(CPU, address space, file size, process count, open files). It never calls
``exec`` or ``import`` on the caller's code in this process.

rlimits are not a filesystem or network namespace. ``IS_SANDBOX`` stays false.
The AST firewall remains a static check in front of this boundary. If the
backend cannot prove the limits were applied, it refuses the command.
"""

from __future__ import annotations

import hashlib
import os
import resource
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, List, Optional

from osiris.runtime.contracts import RuntimeCapability
from osiris.runtime.worker import IsolationUnavailable, WorkerRequest, WorkerResult

IS_SANDBOX = False
BACKEND_NAME = "rlimit-subprocess"


def _set_limits(cpu_s: int, memory_mb: int, disk_mb: int, processes: int) -> None:
    memory = memory_mb * 1024 * 1024
    disk = disk_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (disk, disk))
    resource.setrlimit(resource.RLIMIT_NPROC, (processes, processes))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


class RlimitSubprocessBackend:
    """Run an allowlisted command in a child with rlimits applied."""

    def __init__(self, probe: Optional[Callable[[], bool]] = None) -> None:
        self.verified = False
        self.detail = "not verified"
        checker = probe if probe is not None else self._probe
        try:
            self.verified = bool(checker())
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            self.verified = False
            self.detail = f"isolation probe failed: {exc}"
            return
        self.detail = "rlimits confirmed in a child process" if self.verified else "rlimit probe did not confirm the limit"

    def _probe(self) -> bool:
        script = (
            "import resource,sys\n"
            "limit = resource.getrlimit(resource.RLIMIT_CPU)[0]\n"
            "sys.stdout.write(str(limit))\n"
        )

        def preexec() -> None:
            _set_limits(3, 512, 32, 16)

        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            preexec_fn=preexec,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONNOUSERSITE": "1"},
        )
        return completed.returncode == 0 and completed.stdout.strip() == "3"

    def __call__(self, request: WorkerRequest, capability: RuntimeCapability) -> WorkerResult:
        if not self.verified:
            raise IsolationUnavailable(f"no verified OS isolation backend ({self.detail})")
        if request.command != os.path.basename(request.command) or "/" in request.command:
            raise IsolationUnavailable("untrusted command must be a bare allowlisted name")
        executable = shutil.which(request.command)
        if not executable:
            raise IsolationUnavailable(f"{request.command} is not on PATH; refusing to run it in-process")
        argv: List[str] = [executable, *list(request.arguments)]
        if any("\x00" in part or "\n" in part for part in argv):
            raise IsolationUnavailable("command arguments contain a null or newline")
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"}
        for key in request.environment:
            if key in capability.allowed_environment:
                env[key] = request.environment[key]
        work = tempfile.mkdtemp(prefix="osiris-isolate-")
        if capability.allowed_write_root and os.path.isdir(capability.allowed_write_root):
            work = capability.allowed_write_root

        def preexec() -> None:
            _set_limits(
                capability.max_cpu_seconds,
                capability.max_memory_mb,
                capability.max_disk_mb,
                capability.max_processes,
            )
            os.setsid()

        completed = subprocess.run(
            argv,
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            timeout=capability.max_cpu_seconds + 2,
            check=False,
            preexec_fn=preexec,
        )
        output = completed.stdout
        if len(output.encode("utf-8")) > capability.max_output_bytes:
            output = output.encode("utf-8")[: capability.max_output_bytes].decode("utf-8", errors="ignore")
        status = "completed" if completed.returncode == 0 else "failed"
        return WorkerResult(
            request_id=request.request_id,
            status=status,
            output=output,
            output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
        )


def verified_backend() -> RlimitSubprocessBackend:
    """Return a backend whose probe succeeded, or raise. Never returns an in-process fallback."""
    backend = RlimitSubprocessBackend()
    if not backend.verified:
        raise IsolationUnavailable(backend.detail)
    return backend


def run_untrusted(request: WorkerRequest, capability: RuntimeCapability) -> WorkerResult:
    """Run ``request`` only inside a verified rlimit subprocess."""
    from osiris.runtime.worker import WorkerRunner

    return WorkerRunner(verified_backend()).run(request, capability)
