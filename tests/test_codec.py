import pickle

import pytest
import torch

from easy_nn import codec


def roundtrip(obj, compress=False):
    enc = codec.encode(obj, compress=compress)
    # Simulate the wire: parts arrive as immutable bytes, in order.
    return codec.decode(enc.header, [bytes(p) for p in enc.parts]), enc


def test_mixed_dtypes_and_shapes():
    obj = {
        "f32": torch.randn(4, 5),
        "bf16": torch.randn(3, dtype=torch.bfloat16),
        "f16": torch.randn(2, 2, dtype=torch.float16),
        "i64": torch.arange(6),
        "bool": torch.tensor([True, False, True]),
        "scalar": torch.tensor(3.5),
        "empty": torch.zeros(0, 7),
        "nested": [torch.ones(2), {"deep": torch.eye(3)}],
        "plain": ["caption one", 42, None],
    }
    out, _ = roundtrip(obj)
    for key in ("f32", "bf16", "f16", "i64", "bool", "scalar", "empty"):
        assert out[key].dtype == obj[key].dtype, key
        assert out[key].shape == obj[key].shape, key
        assert torch.equal(out[key], obj[key]), key
    assert torch.equal(out["nested"][1]["deep"], torch.eye(3))
    assert out["plain"] == ["caption one", 42, None]


def test_non_contiguous_and_views():
    src = torch.randn(6, 8)
    obj = {"t": src.t(), "slice": src[:, ::2]}
    out, _ = roundtrip(obj)
    assert torch.equal(out["t"], src.t())
    assert torch.equal(out["slice"], src[:, ::2])


def test_shared_tensor_sent_once_and_stays_shared():
    shared = torch.randn(10)
    obj = {"a": shared, "b": shared, "c": [shared]}
    out, enc = roundtrip(obj)
    assert len(enc.parts) == 1, "the same tensor must not be sent twice"
    assert out["a"] is out["b"] is out["c"][0]


def test_optimizer_keeps_pointing_at_model_parameters():
    """The failure this guards against is silent: training would run against a
    detached copy and the model would never move."""
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Give the optimizer some state, so exp_avg buffers travel too.
    model(torch.randn(3, 4)).sum().backward()
    opt.step()

    out, _ = roundtrip({"model": model, "opt": opt})
    m, o = out["model"], out["opt"]

    model_params = list(m.parameters())
    opt_params = [p for g in o.param_groups for p in g["params"]]
    assert len(opt_params) == len(model_params)
    for mp, op in zip(model_params, opt_params):
        assert mp is op, "optimizer must own the same objects as the model"
        assert isinstance(op, torch.nn.Parameter)
        assert op.requires_grad

    assert o.state_dict()["param_groups"][0]["lr"] == 1e-3
    assert any(torch.is_tensor(v.get("exp_avg")) for v in o.state.values())


def test_module_survives_and_computes():
    model = torch.nn.Linear(4, 3)
    x = torch.randn(2, 4)
    expected = model(x)
    out, _ = roundtrip(model)
    assert torch.allclose(out(x), expected)


def test_locally_defined_class_travels_by_value():
    def make():
        class Recipe:
            def __init__(self):
                self.weight = torch.ones(3)

            def apply(self, x):
                return x * self.weight * 2

        return Recipe()

    out, _ = roundtrip(make())
    assert torch.equal(out.apply(torch.ones(3)), torch.full((3,), 2.0))


@pytest.mark.skipif(not codec.has_zstd(), reason="zstandard not installed")
def test_compression_roundtrip_and_shrinks_compressible_data():
    obj = {"zeros": torch.zeros(200_000), "noise": torch.randn(1000)}
    plain = codec.encode(obj, compress=False)
    packed = codec.encode(obj, compress=True)
    assert packed.nbytes < plain.nbytes / 2
    out = codec.decode(packed.header, [bytes(p) for p in packed.parts])
    assert torch.equal(out["zeros"], obj["zeros"])
    assert torch.equal(out["noise"], obj["noise"])


def test_part_count_mismatch_is_rejected():
    enc = codec.encode({"a": torch.ones(3), "b": torch.ones(3)})
    with pytest.raises(ValueError, match="tensor parts"):
        codec.decode(enc.header, [bytes(enc.parts[0])])
