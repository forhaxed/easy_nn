"""
The executor's view of the data stream.

The client never sends more than it has credit for, so this queue holds at most
``blob_buffer`` undigested blobs.  Credit for a blob is returned only once the
training loop has come back for the next one -- that is, after every batch the
blob held has been consumed.  Backpressure therefore reflects real progress
rather than how fast the socket drains.
"""

from __future__ import annotations

import queue
import threading

BLOB = "blob"
EPOCH_END = "epoch_end"
END = "end"


class BlobFeed:
    def __init__(self, on_consume=None, eval_memory_limit=2 << 30):
        self._queue = queue.Queue()
        self._on_consume = on_consume or (lambda n: None)
        self._eval: list = []
        self._eval_bytes = 0
        self._eval_limit = eval_memory_limit
        self._eval_ready = threading.Event()
        self._failure = None

    # -- filled by the reader thread -------------------------------------
    def put_blob(self, blob):
        self._queue.put((BLOB, blob))

    def put_epoch_end(self, epoch):
        self._queue.put((EPOCH_END, epoch))

    def put_end(self):
        self._queue.put((END, None))

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
        self._queue.put((END, None))

    # -- read by the training loop ---------------------------------------
    def train_stream(self):
        while True:
            kind, payload = self._queue.get()
            if self._failure is not None:
                raise self._failure
            yield kind, payload
            if kind == BLOB:
                # Reached only once the consumer asks for the next item.
                self._on_consume(1)
            elif kind == END:
                return

    def eval_blobs(self):
        self._eval_ready.wait()
        if self._failure is not None:
            raise self._failure
        return self._eval
