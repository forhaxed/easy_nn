"""
How work reaches the executor, end to end.

The case that shaped the design: a dataset so small that one epoch is a single
sample. The stream has to stay continuous across epoch boundaries, or the queue
dribbles out one unit per epoch and the executor spends its life waiting.
"""

import glob
import os

from easy_nn import Local
from tests import toy


def scalars(output_dir, tag):
    from tensorboard.backend.event_processing import event_accumulator

    log_dir = glob.glob(os.path.join(output_dir, "logs", "*"))[0]
    accumulator = event_accumulator.EventAccumulator(log_dir)
    accumulator.Reload()
    return [event.value for event in accumulator.Scalars(tag)]


def test_a_one_batch_epoch_still_fills_the_queue(tmp_path):
    """20 epochs of a single batch must ship as a few full groups, not 20 ones."""
    data = toy.RecordingData(n=8, batch_size=8)
    trainer = toy.build(tmp_path, epochs=20, blob_size=16, blob_size_prepare=8)
    trainer.data = data
    trainer.batch_size = 8

    result = trainer.train(on=Local())

    assert result["global_step"] == 20, "one step per epoch"
    # 20 units at 8 per refill: 8, 8, 4 -- never one unit at a time.
    assert data.group_sizes == [8, 8, 4], data.group_sizes


def test_refill_respects_the_queue_depth(tmp_path):
    """With a shallow queue the client must not run ahead of the executor."""
    data = toy.RecordingData(n=8, batch_size=8)
    trainer = toy.build(tmp_path, epochs=12, blob_size=2, blob_size_prepare=2)
    trainer.data = data
    trainer.batch_size = 8

    result = trainer.train(on=Local())

    assert result["global_step"] == 12
    assert data.group_sizes == [2] * 6, data.group_sizes


def test_telemetry_reaches_tensorboard(tmp_path):
    trainer = toy.build(tmp_path)
    trainer.train(on=Local())

    output = str(tmp_path / "output")
    for tag in ("time/step_s", "time/data_wait_s", "time/unpack_s", "queue/depth"):
        assert scalars(output, tag), f"{tag} missing"

    # Client-side counters land in the same log.
    for tag in ("net/up_MBps", "net/up_total_GB", "queue/outstanding"):
        assert scalars(output, tag), f"{tag} missing"

    assert sum(scalars(output, "net/up_total_GB")) > 0


def test_artifact_round_trip_finishes_locally(tmp_path):
    trainer = toy.build(tmp_path, trainer_class=toy.ArtifactTrainer)
    trainer.train(on=Local())

    received = getattr(trainer, "received", [])
    assert len(received) == 1, "the executor sent exactly one probe"
    name, step, note, shape = received[0]
    assert name == "probe"
    assert step == 3
    assert note == "computed on the executor"
    assert shape == (8, 1)

    output = str(tmp_path / "output")
    assert scalars(output, "artifact/rows") == [8.0]

    # The image was written locally, from data the executor never rendered.
    from tensorboard.backend.event_processing import event_accumulator

    log_dir = glob.glob(os.path.join(output, "logs", "*"))[0]
    accumulator = event_accumulator.EventAccumulator(
        log_dir, size_guidance={"images": 10}
    )
    accumulator.Reload()
    assert accumulator.Images("artifact/grid")
