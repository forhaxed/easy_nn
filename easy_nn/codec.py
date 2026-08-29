"""
Tensor-aware serialization.

One codec serves everything that crosses the wire: the trainer object, data
blobs and checkpoints.

The structure of an object is pickled with cloudpickle -- classes defined in
``__main__`` travel by value, which is what lets an executor run code it has
never seen.  Every ``torch.Tensor`` is pulled out of the pickle stream and sent
as a separate raw part, so nothing is copied twice into RAM and the transport
can report progress while a multi-gigabyte model is uploading.

Tensors are hooked through ``persistent_id``.  The pickler consults it before
the memo table, so object identity is decided here and here only: the same
tensor met twice encodes to the same part index and decodes back to the same
object.  That is what keeps ``optimizer.param_groups`` pointing at the very
tensors held by ``models`` -- if it ever broke, training would silently
optimize a detached copy.
"""

from __future__ import annotations

import ctypes
import io
import pickle
from dataclasses import dataclass, field

import cloudpickle
import torch

try:
    import zstandard
except ImportError:  # optional
    zstandard = None


TENSOR_TYPES = (torch.Tensor, torch.nn.Parameter)


@dataclass
class Encoded:
    """An object split into a pickled structure plus raw tensor parts."""

    header: bytes = b""
    parts: list = field(default_factory=list)
    descriptors: list = field(default_factory=list, repr=False)
    # Contiguous CPU tensors backing the zero-copy memoryviews in ``parts``.
    # They must outlive the send, hence the reference.
    _keepalive: list = field(default_factory=list, repr=False)

    @property
    def nbytes(self) -> int:
        return len(self.header) + sum(len(p) for p in self.parts)


def _raw_view(t: torch.Tensor) -> tuple[memoryview, torch.Tensor]:
    """Zero-copy bytes of a tensor, dtype-agnostic (bfloat16 included)."""
    tc = t.detach().to("cpu", copy=False).contiguous()
    n = tc.numel() * tc.element_size()
    if n == 0:
        return memoryview(b""), tc
    buf = (ctypes.c_char * n).from_address(tc.data_ptr())
    return memoryview(buf), tc


def _from_raw(data, dtype: torch.dtype, shape) -> torch.Tensor:
    t = torch.empty(tuple(shape), dtype=dtype)
    n = t.numel() * t.element_size()
    if n:
        src = (ctypes.c_char * n).from_buffer_copy(data)
        ctypes.memmove(t.data_ptr(), ctypes.addressof(src), n)
    return t


class _Pickler(cloudpickle.Pickler):
    def __init__(self, file, encoded: Encoded, compress):
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self._enc = encoded
        self._compress = compress
        self._seen: dict[int, int] = {}

    def persistent_id(self, obj):
        if type(obj) not in TENSOR_TYPES:
            return None

        key = id(obj)
        idx = self._seen.get(key)
        if idx is None:
            view, keep = _raw_view(obj)
            idx = len(self._enc.parts)
            self._seen[key] = idx

            raw_len = len(view)
            comp = None
            if self._compress and raw_len >= _COMPRESS_MIN and zstandard is not None:
                packed = zstandard.ZstdCompressor(level=1).compress(view)
                if len(packed) < raw_len * 0.9:
                    view, comp = memoryview(packed), "zstd"
                    keep = None

            self._enc.parts.append(view)
            self._enc._keepalive.append(keep)
            self._enc.descriptors.append(
                {
                    "param": type(obj) is torch.nn.Parameter,
                    "dtype": str(obj.dtype).removeprefix("torch."),
                    "shape": tuple(obj.shape),
                    "requires_grad": bool(obj.requires_grad),
                    "comp": comp,
                    "raw_len": raw_len,
                }
            )
        return ("t", idx)


class _Unpickler(pickle.Unpickler):
    def __init__(self, file, descriptors, parts):
        super().__init__(file)
        self._descriptors = descriptors
        self._parts = parts
        self._cache: dict[int, torch.Tensor] = {}

    def persistent_load(self, pid):
        tag, idx = pid
        if tag != "t":
            raise pickle.UnpicklingError(f"unknown persistent id {pid!r}")

        obj = self._cache.get(idx)
        if obj is None:
            d = self._descriptors[idx]
            data = self._parts[idx]
            if d["comp"] == "zstd":
                if zstandard is None:
                    raise RuntimeError(
                        "payload is zstd-compressed but zstandard is not installed"
                    )
                data = zstandard.ZstdDecompressor().decompress(
                    bytes(data), max_output_size=d["raw_len"]
                )
            obj = _from_raw(data, getattr(torch, d["dtype"]), d["shape"])
            if d["param"]:
                obj = torch.nn.Parameter(obj, requires_grad=d["requires_grad"])
            else:
                obj.requires_grad_(d["requires_grad"])
            self._cache[idx] = obj
        return obj


_COMPRESS_MIN = 1 << 16  # below this, framing overhead dominates


def encode(obj, compress: bool = False) -> Encoded:
    """Split ``obj`` into a pickled structure and a list of raw tensor parts."""
    enc = Encoded()
    buf = io.BytesIO()
    _Pickler(buf, enc, compress).dump(obj)
    enc.header = pickle.dumps(
        {"descriptors": enc.descriptors, "structure": buf.getvalue()},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    return enc


def decode(header: bytes, parts) -> object:
    head = pickle.loads(header)
    descriptors = head["descriptors"]
    if len(parts) != len(descriptors):
        raise ValueError(
            f"expected {len(descriptors)} tensor parts, got {len(parts)}"
        )
    return _Unpickler(
        io.BytesIO(head["structure"]), descriptors, parts
    ).load()


def has_zstd() -> bool:
    return zstandard is not None
