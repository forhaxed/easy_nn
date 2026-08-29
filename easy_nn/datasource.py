"""
The local half of a job.

A ``DataSource`` never leaves your machine.  It owns the dataset, the disk and
the dataloader; the only thing that crosses the wire is whatever ``pack()``
returns.
"""

from __future__ import annotations


class DataSource:
    """Subclass this for the local side of a training job.

    Override ``setup`` to build your dataloader, ``stream`` to yield batches
    and ``pack`` to decide what actually travels.  ``pack`` receives a list of
    ``blob_size_prepare`` batches and its return value is serialized as one
    message;
    the executor turns it back into batches with ``Trainer.unpack``.
    """

    _ready = False

    def __init__(self, **options):
        self.options = options

    # -- lifecycle -------------------------------------------------------
    def prepare(self) -> None:
        """Called once by the client before streaming starts."""
        if not self._ready:
            self.setup()
            self._ready = True

    def setup(self) -> None:
        """Build the dataloader here.  Called once, locally."""

    # -- to override -----------------------------------------------------
    def __len__(self) -> int:
        """Number of samples.  Used only to work out the total step count."""
        raise NotImplementedError(
            f"{type(self).__name__} must define __len__ so the trainer can "
            "work out how many steps a run takes"
        )

    def stream(self):
        """Yield batches, in order, for one pass over the data."""
        raise NotImplementedError(f"{type(self).__name__} must define stream()")

    def pack(self, batches):
        """Turn a group of batches into the payload that crosses the wire.

        The default sends the batches untouched.  Override to cut the payload
        down -- drop columns the training step never reads, cast to fp16, or
        pre-compute anything cheap on the CPU.
        """
        return batches


class IterableSource(DataSource):
    """Wrap an existing iterable (a DataLoader, usually) as a DataSource."""

    def __init__(self, iterable, length=None, pack=None):
        super().__init__()
        self.iterable = iterable
        self._length = length
        self._pack = pack

    def __len__(self):
        if self._length is not None:
            return self._length
        dataset = getattr(self.iterable, "dataset", None)
        if dataset is not None:
            return len(dataset)
        return len(self.iterable)

    def stream(self):
        yield from self.iterable

    def pack(self, batches):
        return self._pack(batches) if self._pack is not None else batches
