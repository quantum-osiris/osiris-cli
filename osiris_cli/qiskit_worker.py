"""One long-lived thread for all Qiskit / Aer work in the IDE API, with a bounded backlog (F-34).

Why: with qiskit 2.6.2 and qiskit-aer 0.18.2 on Python 3.14, a second simulation started from a
*different* short-lived thread segfaults the whole process inside ``qiskit._accelerate``
(``get_standard_gate_name_mapping``, reached from ``AerSimulator.target`` during transpile).
``ThreadingHTTPServer`` opens one thread per request, so any two run requests could stop the
service. Running every Qiskit call on one persistent thread avoids it (reproduced and checked in
tests/test_qiskit_worker.py).

Policy:
* Calls run one at a time, in arrival order, on the thread named ``osiris-qiskit-0``.
* At most ``max_pending`` calls may be admitted (running + waiting). Beyond that, ``submit`` raises
  ``Busy`` at once; the API answers 503 ``SIMULATOR_BUSY``. Nothing queues without bound.
* A call is never cancelled once admitted: Aer cannot be interrupted safely, and a run that was
  admitted must still write its receipt. If the client gives up, the work finishes, its slot is
  released on completion, and the result is discarded with the request.
* Exceptions raised by the call (including ``ApiError``) are re-raised in the request thread, and
  the worker stays usable.
* ``shutdown`` waits for admitted work, so receipts of admitted runs are written before exit.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

DEFAULT_MAX_PENDING = 4


class Busy(Exception):
    def __init__(self, max_pending: int):
        super().__init__(f"{max_pending} simulator requests are already admitted; retry shortly")
        self.max_pending = max_pending


class QiskitWorker:
    def __init__(self, max_pending: Optional[int] = None):
        if max_pending is None:
            max_pending = int(os.environ.get("OSIRIS_QISKIT_MAX_PENDING", DEFAULT_MAX_PENDING))
        if max_pending < 1:
            raise ValueError("max_pending must be at least 1")
        self.max_pending = max_pending
        self._slots = threading.BoundedSemaphore(max_pending)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="osiris-qiskit")
        self._lock = threading.Lock()
        self._admitted = 0
        self._completed = 0
        self._rejected = 0

    def submit(self, fn: Callable[[], Any]) -> Any:
        """Run ``fn`` on the worker thread and return its result (or raise its exception)."""
        if not self._slots.acquire(blocking=False):
            with self._lock:
                self._rejected += 1
            raise Busy(self.max_pending)
        with self._lock:
            self._admitted += 1
        try:
            future = self._pool.submit(fn)
        except BaseException:
            self._release()
            raise
        future.add_done_callback(lambda _f: self._release())
        return future.result()

    def _release(self) -> None:
        with self._lock:
            self._completed += 1
        self._slots.release()

    def stats(self) -> dict:
        with self._lock:
            return {"max_pending": self.max_pending, "admitted": self._admitted, "completed": self._completed,
                    "in_flight": self._admitted - self._completed, "rejected_busy": self._rejected}

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
