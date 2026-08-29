"""End-to-end tests against the Local executor -- a real subprocess."""

import glob
import json
import os
import threading
import time

import pytest

from easy_nn import Local
from easy_nn.client.session import RemoteError
from tests import toy


def snapshot(directory):
    return {
        os.path.relpath(os.path.join(root, name), directory)
        for root, _, files in os.walk(directory)
        for name in files
    }


def scalars(output_dir, tag):
    from tensorboard.backend.event_processing import event_accumulator

    log_dir = glob.glob(os.path.join(output_dir, "logs", "*"))[0]
    accumulator = event_accumulator.EventAccumulator(log_dir)
    accumulator.Reload()
    return [event.value for event in accumulator.Scalars(tag)]


def test_runs_end_to_end_and_writes_only_on_this_machine(tmp_path):
    workdir = tmp_path / "executor"
    workdir.mkdir()
    before = snapshot(workdir)

    trainer = toy.build(tmp_path, save_checkpoint_every_steps=16)
    result = trainer.train(on=Local(workdir=str(workdir)))

    # The executor's working directory is untouched: no logs, no checkpoints.
    assert snapshot(workdir) == before

    output = tmp_path / "output"
    assert glob.glob(str(output / "logs" / "*" / "events.out.tfevents.*"))
    assert (output / "checkpoints" / "step_16").is_dir()
    assert result["global_step"] == 32

    metadata = json.loads(
        (output / "checkpoints" / "step_16" / "trainer_metadata.json").read_text()
    )
    assert metadata["global_step"] == 16


def test_model_actually_learns(tmp_path):
    trainer = toy.build(tmp_path, epochs=3)
    trainer.train(on=Local())

    losses = scalars(str(tmp_path / "output"), "train/loss")
    assert len(losses) > 20
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 10


def test_custom_training_loop_travels_and_runs(tmp_path):
    trainer = toy.build(tmp_path, trainer_class=toy.CustomLoopTrainer)
    result = trainer.train(on=Local())
    assert result["global_step"] == 32
    assert len(scalars(str(tmp_path / "output"), "custom/loss")) == 32


def test_remote_failure_arrives_with_its_traceback(tmp_path):
    trainer = toy.build(tmp_path, trainer_class=toy.ExplodingTrainer)
    with pytest.raises(RemoteError) as excinfo:
        trainer.train(on=Local())
    text = str(excinfo.value)
    assert "deliberate failure inside train_step" in text
    assert "train_step" in text


def test_runs_with_the_shallowest_possible_queue(tmp_path):
    trainer = toy.build(tmp_path, blob_size=1, blob_size_prepare=1)
    result = trainer.train(on=Local())
    assert result["global_step"] == 32


def test_evaluation_streams_from_the_local_side(tmp_path):
    trainer = toy.build(tmp_path, eval_every_steps=16)
    trainer.eval_data = toy.ToyData(n=64, batch_size=8, seed=7)
    trainer.train(on=Local())

    eval_losses = scalars(str(tmp_path / "output"), "eval/loss")
    assert len(eval_losses) == 2


def test_resume_continues_from_the_saved_step(tmp_path):
    first = toy.build(tmp_path, save_checkpoint_every_steps=16)
    first.train(on=Local())
    checkpoint = tmp_path / "output" / "checkpoints" / "step_16"
    assert checkpoint.is_dir()

    second = toy.build(tmp_path, epochs=2, resume_from=str(checkpoint))
    result = second.train(on=Local())

    # Resumed at 16, the first epoch had 16 steps left, then a full second one.
    assert result["global_step"] == 64


def start_run(trainer):
    """Run a job on a background thread and hand back a joiner."""
    outcome = {}

    def run():
        try:
            outcome["result"] = trainer.train(on=Local())
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    return thread, outcome


def wait_until_training(tmp_path, timeout=90):
    """Block until the loop has logged a step.

    Dropping a lock file before the executor has finished importing torch and
    initializing CUDA is useless: pause and resume would both be drained on the
    first pump and cancel out.
    """
    deadline = time.time() + timeout
    pattern = str(tmp_path / "output" / "logs" / "*" / "events.out.tfevents.*")
    while time.time() < deadline:
        if glob.glob(pattern):
            return
        time.sleep(0.05)
    raise AssertionError("training never started")


def test_save_lock_forces_a_checkpoint(tmp_path):
    trainer = toy.build(tmp_path, n=512, unpack_delay=0.5)
    trainer.save_checkpoint_every_steps = 0  # only the lock can save early
    trainer.data = toy.ToyData(n=512, batch_size=8)

    thread, outcome = start_run(trainer)
    wait_until_training(tmp_path)

    lock = tmp_path / "output" / "save_checkpoint.lock"
    lock.write_text("")

    thread.join(timeout=120)
    assert not thread.is_alive(), "training did not finish"
    assert "error" not in outcome, outcome.get("error")

    saved = {p.name for p in (tmp_path / "output" / "checkpoints").iterdir()}
    assert not lock.exists(), "the lock file should be consumed"
    assert any(not name.endswith("_final") for name in saved), (
        f"the lock did not trigger an extra checkpoint; got {saved}"
    )


def test_pause_lock_halts_and_releases(tmp_path, capfd):
    trainer = toy.build(tmp_path, unpack_delay=0.5)
    trainer.data = toy.ToyData(n=512, batch_size=8)

    thread, outcome = start_run(trainer)
    wait_until_training(tmp_path)

    lock = tmp_path / "output" / "pause.lock"
    lock.write_text("")
    time.sleep(2.5)
    lock.unlink()

    thread.join(timeout=120)
    assert not thread.is_alive(), "training did not resume after the lock went away"
    assert "error" not in outcome, outcome.get("error")

    captured = capfd.readouterr()
    text = captured.out + captured.err
    assert "Pausing" in text
    assert "Resuming" in text
