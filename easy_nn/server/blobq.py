"""
The executor's queue of work.

A *blob* is one unit of work -- one sample. The client keeps roughly
``blob_size`` units sitting here and tops them up in batches of
``blob_size_prepare`` whenever the queue drops below that mark, so the local
side is preparing the next batch while the executor is still training on the
current one.

Credit is returned per consumed unit, not per message: the client's picture of
"how much work is over there" stays accurate no matter how the units were
grouped for transport. The stream is continuous -- epochs are bookkeeping the
executor derives from the step counter, they never chop it up.
"""

from __future__ import annotations

import queue
import threading
import time

WORK = "work"
END = "end"


class WorkQueue:
    def __init__(self, on_consume=None, eval_memory_limit=2 << 30):
        self._queue = queue.Queue()
        self._on_consume = on_consume or (lambda n: None)
        self._eval: list = []
        self._eval_bytes = 0
        self._eval_limit = eval_memory_limit
        self._eval_ready = threading.Event()
        self._failure = None

        #: Seconds the training loop has spent blocked waiting for work.
        #: The single number that says whether the network is the bottleneck.
        self.wait_seconds = 0.0
        self.units_received = 0
        self.units_consumed = 0

    # -- filled by the reader thread -------------------------------------
    def put(self, payload, units: int):
        self.units_received += units
        self._queue.put((WORK, payload, units))

    def put_end(self):
        self._queue.put((END, None, 0))

    def put_eval(self, blob, nbytes=0):
        self._eval_bytes += nbytes
        if self._eval_bytes > self._eval_limit:
            raise MemoryError(
                f"evaluation data exceeds {self._eval_limit >> 20} MiB. "
                "The executor holds the whole eval set in memory; use a "
                "smaller eval_data or raise eval_memory_limit."
            )
        self._eval.append(blob)

    def eval_done(self):
        self._eval_ready.set()

    def fail(self, exc):
        """Report a broken stream so a blocked consumer wakes up."""
        self._failure = exc
        self._eval_ready.set()
        self._queue.put((END, None, 0))

    # -- read by the training loop ---------------------------------------
    def stream(self):
        """Yield ``(payload, units)`` until the client says the work is done."""
        while True:
            started = time.perf_counter()
            kind, payload, units = self._queue.get()
            self.wait_seconds += time.perf_counter() - started

            if self._failure is not None:
                raise self._failure
            if kind == END:
                return
            yield payload, units

    def consumed(self, n: int = 1):
        """Hand credit back for ``n`` units the loop has finished with."""
        if n <= 0:
            return
        self.units_consumed += n
        self._on_consume(n)

    @property
    def depth(self) -> int:
        """Units received but not yet consumed."""
        return self.units_received - self.units_consumed

    def eval_blobs(self):
        self._eval_ready.wait()
        if self._failure is not None:
            raise self._failure
        return self._eval
