"""Enforceable execution sandbox for running an untrusted repository's test suite.

Backend: bubblewrap (``bwrap``) with fresh user, pid, net, ipc, uts and mount namespaces. The child
sees ONLY: the interpreter's prefix (read-only), a few loader files under /etc (read-only), a tiny
/dev and /proc, a private tmpfs /tmp, and the working directory (read-write). It does not see the
home directory, ~/.osiris, ssh keys, env files, or any other repository; it has no network
interface and a scrubbed environment. rlimits bound CPU, address space, processes, file size and
open files; a wall-clock timeout kills the whole process group.

Fail closed: if the requested isolation cannot be provided, ``run`` raises ``SandboxUnavailable``
and nothing executes. There is deliberately no "run unsandboxed" fallback here.

What this is not: a hardened container or a VM. It shares the host kernel (a kernel exploit
defeats it), CPU time is bounded but not cgroup-metered, and a process can still use disk under
the working directory up to the per-file and tmpfs limits. ``selftest()`` demonstrates each
property on the machine it runs on; nothing here is claimed on machines where it was not run.
"""

from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

_ETC_RO = ("ld.so.cache", "ld.so.conf", "ld.so.conf.d", "passwd", "group", "nsswitch.conf", "localtime")
_SYS_RO = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64")
SAFE_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "TERM")


class SandboxUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Limits:
    wall_s: float = 60.0
    cpu_s: int = 60
    mem_mb: int = 3072          # RLIMIT_AS
    procs: int = 256            # RLIMIT_NPROC (per real uid; the namespace has its own uid map)
    fsize_mb: int = 64          # RLIMIT_FSIZE per file
    nofile: int = 256
    out_bytes: int = 1_000_000  # captured output cap


@dataclass
class SandboxResult:
    returncode: int
    output: str
    timed_out: bool
    truncated: bool
    isolation: Dict[str, object] = field(default_factory=dict)


def _bwrap() -> Optional[str]:
    return shutil.which("bwrap")


def available() -> bool:
    """True only if a real namespace sandbox can start here (probed, not assumed)."""
    b = _bwrap()
    if not b:
        return False
    try:
        p = subprocess.run([b, "--unshare-all", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
                            "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "true"],
                           capture_output=True, timeout=10)
        return p.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def isolation_report() -> Dict[str, object]:
    ok = available()
    return {"backend": "bwrap" if ok else "none", "fs_isolated": ok, "net_isolated": ok, "pid_isolated": ok,
            "rlimits": True, "kernel_shared": True, "hardened_container": False}


def _interpreter_prefixes() -> List[str]:
    out = []
    for p in (os.path.dirname(os.path.dirname(os.path.realpath(sys.executable))),
              os.path.dirname(os.path.dirname(sys.executable)), sys.prefix, sys.base_prefix):
        rp = os.path.realpath(p)
        if rp not in out and rp not in ("/", "/usr") and os.path.isdir(rp):
            out.append(rp)
    return out


def _argv_bwrap(argv: Sequence[str], workdir: str, ro_extra: Sequence[str]) -> List[str]:
    wd = os.path.realpath(workdir)
    home = os.path.realpath(os.path.expanduser("~"))
    if wd in (home, "/") or home.startswith(wd + os.sep):
        raise SandboxUnavailable("refusing a working directory that contains the home directory")
    cmd = [_bwrap() or "bwrap", "--die-with-parent", "--new-session", "--unshare-all", "--clearenv",
           "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--hostname", "sandbox"]
    for p in _SYS_RO:
        if os.path.islink(p):
            cmd += ["--symlink", os.readlink(p), p]
        elif os.path.isdir(p):
            cmd += ["--ro-bind", p, p]
    cmd += ["--dir", "/etc"]
    for n in _ETC_RO:
        f = "/etc/" + n
        if os.path.exists(f):
            cmd += ["--ro-bind", os.path.realpath(f), f]
    for p in list(_interpreter_prefixes()) + [os.path.realpath(x) for x in ro_extra]:
        if p.startswith(home + os.sep) or not p.startswith(("/usr", "/lib", "/opt")):
            cmd += ["--ro-bind", p, p]          # e.g. a venv under the home directory: that path only
    cmd += ["--bind", wd, wd, "--chdir", wd]
    env = {"HOME": wd, "TMPDIR": "/tmp", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": wd,
           "PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin"}
    for k in SAFE_ENV_KEYS:
        if k in os.environ and k != "PATH":
            env[k] = os.environ[k]
    for k, v in env.items():
        cmd += ["--setenv", k, v]
    return cmd + ["--"] + list(argv)


def _preexec(lim: Limits):
    def fn():
        os.setsid()
        mb = 1024 * 1024
        for r, v in ((resource.RLIMIT_CPU, lim.cpu_s), (resource.RLIMIT_AS, lim.mem_mb * mb),
                     (resource.RLIMIT_FSIZE, lim.fsize_mb * mb), (resource.RLIMIT_NOFILE, lim.nofile),
                     (resource.RLIMIT_NPROC, lim.procs), (resource.RLIMIT_CORE, 0)):
            try:
                resource.setrlimit(r, (v, v))
            except (ValueError, OSError):
                pass
    return fn


def run(argv: Sequence[str], workdir: str, *, limits: Limits = Limits(),
        ro_paths: Sequence[str] = ()) -> SandboxResult:
    if not available():
        raise SandboxUnavailable("bubblewrap with user namespaces is not available; refusing to run unsandboxed")
    full = _argv_bwrap(argv, workdir, ro_paths)
    # RLIMIT_AS applies to bwrap too; leave it headroom, the child inherits the same ceiling.
    with tempfile.TemporaryFile() as out:
        p = subprocess.Popen(full, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             preexec_fn=_preexec(limits), env={"PATH": "/usr/bin:/bin"})
        timed_out = False
        try:
            p.wait(timeout=limits.wall_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(p.pid, sig)
                except ProcessLookupError:
                    break
                time.sleep(0.3)
            p.wait()
        out.seek(0)
        data = out.read(limits.out_bytes + 1)
    trunc = len(data) > limits.out_bytes
    return SandboxResult(124 if timed_out else p.returncode, data[:limits.out_bytes].decode("utf-8", "replace"),
                         timed_out, trunc, isolation_report())


def selftest(python: Optional[str] = None) -> Dict[str, bool]:
    """Demonstrate each guarantee on THIS machine. True = the attack was contained."""
    import socket
    py = python or sys.executable
    res: Dict[str, bool] = {}
    wd = tempfile.mkdtemp(prefix="osiris_sbx_")
    try:
        def probe(code: str, lim: Limits = Limits(wall_s=20)) -> SandboxResult:
            return run([py, "-I", "-c", code], wd, limits=lim)

        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        r = probe(f"import socket\ntry:\n s=socket.create_connection(('127.0.0.1',{port}),2);print('CONNECTED')\n"
                  f"except OSError as e: print('BLOCKED')")
        res["host_loopback_unreachable"] = "BLOCKED" in r.output
        r = probe("import socket\ntry:\n socket.create_connection(('1.1.1.1',53),2);print('CONNECTED')\n"
                  "except OSError: print('BLOCKED')")
        res["internet_unreachable"] = "BLOCKED" in r.output
        srv.close()

        home = os.path.expanduser("~")
        rh = os.path.realpath(home)
        allowed = sorted({os.path.relpath(q, rh).split(os.sep)[0] for q in _interpreter_prefixes()
                          if q.startswith(rh + os.sep)})
        r = probe(f"import os\nh={home!r}\nbad=[n for n in os.listdir(h) if n not in {allowed!r}] "
                  f"if os.path.isdir(h) else []\nprint('LEAK' if bad or os.path.exists('/mnt/c') else 'HIDDEN')")
        res["home_dir_not_visible"] = "HIDDEN" in r.output
        marker = os.path.join(tempfile.gettempdir(), "osiris_sbx_marker_%d" % os.getpid())
        r = probe(f"open({marker!r},'w').write('x')\nprint('WROTE')")
        res["tmp_writes_do_not_reach_host"] = not os.path.exists(marker)
        r = probe("try:\n open('/usr/osiris_x','w');print('WROTE')\nexcept OSError: print('RO')")
        res["system_dirs_read_only"] = "RO" in r.output
        os.environ["OSIRIS_SELFTEST_SECRET_TOKEN"] = "hunter2"
        r = probe("import os\nprint('LEAK' if any('SECRET' in k or 'TOKEN' in k for k in os.environ) else 'CLEAN')")
        res["environment_scrubbed"] = "CLEAN" in r.output
        os.environ.pop("OSIRIS_SELFTEST_SECRET_TOKEN", None)
        r = probe("b=bytearray(6*1024*1024*1024)\nprint('ALLOCATED')", Limits(wall_s=20, mem_mb=512))
        res["memory_limit_enforced"] = "ALLOCATED" not in r.output
        r = probe("import subprocess,sys\nn=0\nps=[]\ntry:\n for i in range(200):\n  ps.append(subprocess.Popen(['sleep','5']));n+=1\n"
                  "except OSError: pass\nprint('SPAWNED',n)\n[p.kill() for p in ps]", Limits(wall_s=20, procs=40))
        spawned = int(r.output.split("SPAWNED")[1].split()[0]) if "SPAWNED" in r.output else 0
        res["process_limit_enforced"] = spawned < 200
        t0 = time.monotonic()
        r = probe("import time\ntime.sleep(60)", Limits(wall_s=2))
        res["wall_timeout_kills"] = r.timed_out and time.monotonic() - t0 < 10
        r = probe("print('x'*5000000)", Limits(wall_s=20, out_bytes=1000))
        res["output_capped"] = r.truncated and len(r.output) <= 1000
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    return res


if __name__ == "__main__":
    import json
    print(json.dumps({"isolation": isolation_report(), "selftest": selftest() if available() else "unavailable"},
                     indent=2))
