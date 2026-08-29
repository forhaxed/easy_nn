"""
The training loop.

A ``Trainer`` subclass travels to the executor whole -- cloudpickle sends the
class by value, so the pod runs code it has never seen and needs no copy of
your project.  Everything you assign to it travels too.

Two halves, one object.  Methods under "Local side" run on your machine on the
original instance; everything else runs on the executor, reading units of work
out of a queue instead of a dataloader and reporting back over the link.  It
never writes logs or checkpoints to the executor's disk.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
import time
from datetime import datetime

import torch
from colorama import Fore, Style

CTRL_PAUSE = "pause"
CTRL_RESUME = "resume"
CTRL_SAVE = "save"
CTRL_EVAL = "eval"
CTRL_STOP = "stop"


class _NullLink:
    """Stand-in used before a job is dispatched, so attribute access is safe."""

    def log(self, values, step):
        pass

    def print(self, text):
        print(text, end="")

    def progress(self, **kw):
        pass

    def checkpoint(self, name, payload):
        raise RuntimeError("no executor attached")

    def artifact(self, name, payload, step, compress=False):
        raise RuntimeError("no executor attached")

    def take_control(self):
        return []


class Trainer:
    """Subclass this and override ``train_step``.

    The knobs match ``AnyTrainer``.  What changed: the dataloader lives in a
    ``DataSource`` on your machine, ``precache_dataset`` is split into
    ``DataSource.pack`` (local) and ``Trainer.unpack`` (executor), and saving
    goes through ``save_checkpoint`` instead of writing files.
    """

    #: Attributes that stay behind when the trainer is shipped to an executor.
    LOCAL_ONLY = frozenset(
        {
            "data",
            "eval_data",
            "accelerator",
            "_link",
            "_feed",
            "_session",
            "_executor",
        }
    )

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        self.batch_size = None
        self.epochs = None
        self.gradient_accumulation_steps = 1
        self.optimizer = None
        self.scheduler = None
        self.models = []
        self.non_trainable_models = []

        # Local halves -- never shipped.
        self.data = None
        self.eval_data = None

        self.repeats = 1

        #: How many units of work should be sitting on the executor. A unit is
        #: one sample: the queue depth, not the size of a message.
        self.blob_size = 16
        #: How many units the local side prepares and ships in one go, once the
        #: executor's queue has dropped below ``blob_size``.
        self.blob_size_prepare = 8

        #: The executor holds the whole eval set in memory; this caps it.
        self.eval_memory_limit = 2 << 30
        self.compression = "auto"  # "auto" | "zstd" | "off"
        self.requirements = []

        self.global_step = 0
        self.epochs_trained = 0
        self.steps_in_epoch = 0
        self.save_checkpoint_every_steps = 0
        self.eval_every_steps = 0
        self.max_grad_norm = None
        self.mixed_precision = "no"
        self.seed = None
        self.resume_from = None
        self.allow_skip_batches_on_resume = True
        #: Upload even when the executor's Python or torch series differs.
        #: Both normally fail only after the whole model has been sent, and a
        #: Python mismatch misexecutes bytecode rather than failing cleanly.
        self.allow_version_mismatch = False
        #: Modules the executor must import before the job is sent. None means
        #: derive them from ``requirements``; set a list to override.
        self.verify_imports = None
        #: Extra directories whose modules should travel by value.
        self.ship_modules = []

        #: Filled in locally from ``len(data)`` before the job is sent.
        self.dataset_size = 0

        self.accelerator = None
        self.weight_dtype = torch.float32

        self._link = _NullLink()
        self._feed = None
        self._paused = False
        self._save_requested = False
        self._eval_requested = False
        self._stop_requested = False
        self._stream_done = False
        self._unpack_seconds = 0.0

    # ================================================================
    #  Local side -- runs on your machine
    # ================================================================
    def __getstate__(self):
        """Drop the local-only halves so the trainer can cross the wire."""
        return {k: v for k, v in self.__dict__.items() if k not in self.LOCAL_ONLY}

    def __setstate__(self, state):
        self.__dict__.update(state)
        for name in self.LOCAL_ONLY:
            self.__dict__.setdefault(name, None)
        self._link = _NullLink()

    def train(self, on=None):
        """Run this job on ``on`` -- an executor such as ``Local()``.

        This is the local entry point.  It packs the trainer up, ships it, and
        then feeds work and collects logs and checkpoints until the run ends.
        To customize the loop itself, override ``training_loop``.
        """
        if on is None:
            raise TypeError(
                "train() needs an executor, e.g. train(on=Local()). "
                "To customize the loop that runs on the executor, override "
                "training_loop() instead."
            )
        from easy_nn.client.session import run_job

        return run_job(self, on)

    def on_artifact(self, name, payload, step, reporter):
        """Handle something the executor sent home. **Runs locally.**

        This is the counterpart of ``send_artifact``: work the executor should
        not finish itself lands here, on your machine, with your models and
        your data at hand.  Validation latents, for instance, get decoded here
        by the local VAE and written to TensorBoard through ``reporter``.
        """

    # ================================================================
    #  Executor side -- services available to your code
    # ================================================================
    @property
    def device(self):
        return self.accelerator.device

    @property
    def steps_per_epoch(self) -> int:
        effective = self.batch_size * self.gradient_accumulation_steps
        return max(1, (self.dataset_size // effective) * self.repeats)

    def log(self, values: dict, step: int | None = None):
        """Send scalars to the local TensorBoard writer."""
        self._link.log(values, self.global_step if step is None else step)

    def print(self, *args, **kwargs):
        """Print on the local terminal."""
        end = kwargs.pop("end", "\n")
        sep = kwargs.pop("sep", " ")
        self._link.print(sep.join(str(a) for a in args) + end)

    def send_artifact(self, name: str, payload, step: int | None = None):
        """Ship a tensor-bearing payload home for ``on_artifact`` to handle."""
        self._link.artifact(
            name, payload, self.global_step if step is None else step
        )

    # ================================================================
    #  Executor side -- hooks to override
    # ================================================================
    def unpack(self, blob, device, weight_dtype):
        """Turn one prepared batch back into a list of training batches.

        Runs on the executor with the GPU available, so this is the place to
        do anything expensive that you would rather not send over the wire.
        """
        return blob

    def train_step(self, step, batch, device, weight_dtype):
        raise NotImplementedError(f"{type(self).__name__} must define train_step()")

    def eval_begin(self, step):
        pass

    def eval_step(self, step, batch, device, weight_dtype):
        return self.train_step(step, batch, device, weight_dtype)

    def eval_end(self, step):
        pass

    def gradient_sync(self, step):
        pass

    def save_checkpoint(self, step) -> dict:
        """Return ``{filename: bytes | tensor-bearing object}`` to save locally.

        The default hands back a full ``accelerator`` state, which is what you
        need to resume exactly.  Override to send less -- adapter weights only,
        say -- and then nothing at all touches the executor's disk.
        """
        return self._full_state_payload()

    def load_checkpoint(self, payload: dict) -> None:
        self._load_full_state(payload)

    # ================================================================
    #  Executor side -- setup
    # ================================================================
    def init(self):
        from accelerate import Accelerator

        if self.seed is not None:
            from accelerate.utils import set_seed

            set_seed(self.seed)

        # No project_dir and no trackers: the executor writes nothing to disk.
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
        )

        self.weight_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }.get(self.accelerator.mixed_precision, torch.float32)

        self.print(f"{Fore.GREEN}easy_nn training started...{Style.RESET_ALL}\n")
        self.print(f"{Fore.BLUE}Models Summary:{Style.RESET_ALL}")
        for model in self.models:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print(
                f" {model.__class__.__name__}:\n"
                f"  Total Params: {total}\n"
                f"  Trainable Params: {trainable}"
            )

        for i, model in enumerate(self.models):
            model.to(self.accelerator.device)
            for param in model.parameters():
                param.data = param.to(
                    dtype=torch.float32 if param.requires_grad else self.weight_dtype
                )
            model.train()
            self.models[i] = self.accelerator.prepare(model)

        if self.scheduler is None and self.optimizer is not None:
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer=self.optimizer, factor=1.0
            )

        self.optimizer = self.accelerator.prepare(self.optimizer)
        self.scheduler = self.accelerator.prepare(self.scheduler)

    # ================================================================
    #  Executor side -- work
    # ================================================================
    def batches(self):
        """Yield training batches from the queue, continuously.

        Epoch boundaries do not appear here: the client sends one unbroken
        stream and the executor derives epoch bookkeeping from the step count.
        Credit for a unit goes back only once its batches have been consumed,
        so the client's picture of the queue reflects real progress.
        """
        for payload, units in self._feed.stream():
            started = time.perf_counter()
            with torch.no_grad():
                unpacked = self.unpack(
                    payload, self.accelerator.device, self.weight_dtype
                )
            self._unpack_seconds += time.perf_counter() - started

            credited = 0
            for batch in unpacked:
                if isinstance(batch, dict):
                    batch.setdefault("batch_hash", random.randint(0, 2**63))
                for _ in range(self.repeats):
                    yield batch
                if credited < units:
                    self._feed.consumed(1)
                    credited += 1

            # Keep credit exact even if unpack changed the count.
            if credited < units:
                self._feed.consumed(units - credited)
        self._stream_done = True

    # ================================================================
    #  Executor side -- the loop
    # ================================================================
    def training_loop(self):
        """The default loop.  Override this to write your own."""
        if self.batch_size is None or self.epochs is None or self.optimizer is None:
            raise ValueError("trainer is missing batch_size, epochs or optimizer")

        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        total_steps = self.steps_per_epoch * self.epochs

        self.print(f"\n{Fore.BLUE}Training Configuration:{Style.RESET_ALL}")
        self.print(f" Device: {self.accelerator.device}")
        self.print(f" Mixed Precision: {self.accelerator.mixed_precision}")
        self.print(f" Dataset Size: {self.dataset_size}")
        self.print(f" Total Epochs: {self.epochs}")
        self.print(f" Batch Size: {self.batch_size}")
        self.print(f" Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        self.print(f" Effective Batch Size: {effective_batch_size}")
        self.print(f" Repeats: {self.repeats}")
        self.print(f" Queue depth on executor: {self.blob_size}")
        self.print(f" Prepared per refill: {self.blob_size_prepare}")
        self.print(f" Total Training Steps: {total_steps}\n")

        for model in self.non_trainable_models:
            model.to(self.accelerator.device, dtype=self.weight_dtype)
            model.eval()

        self._link.progress(total=total_steps, step=self.global_step, start=True)

        train_loss_accs = {}
        previous_wait = self._feed.wait_seconds if self._feed else 0.0
        previous_unpack = self._unpack_seconds
        step_started = time.perf_counter()

        for batch in self.batches():
            with self.accelerator.accumulate(self.models):
                loss, loss_dict = self.train_step(
                    self.global_step,
                    batch,
                    device=self.accelerator.device,
                    weight_dtype=self.weight_dtype,
                )
                merged = dict(loss_dict or {})
                merged["loss"] = loss

                for key, value in merged.items():
                    if torch.is_tensor(value):
                        value = self.accelerator.gather(
                            value.detach().repeat(self.batch_size)
                        ).mean().item()
                    train_loss_accs[key] = (
                        train_loss_accs.get(key, 0.0)
                        + float(value) / self.gradient_accumulation_steps
                    )

                self.accelerator.backward(loss)

                grad_norm = None
                if self.accelerator.sync_gradients:
                    params = [
                        p
                        for group in self.optimizer.param_groups
                        for p in group["params"]
                        if p.grad is not None
                    ]
                    if params:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            params,
                            float("inf")
                            if self.max_grad_norm is None
                            else self.max_grad_norm,
                        )

                self.optimizer.step()
                if self.accelerator.sync_gradients and self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()

            if not self.accelerator.sync_gradients:
                continue

            self.gradient_sync(self.global_step)

            now = time.perf_counter()
            wait_now = self._feed.wait_seconds if self._feed else 0.0
            log_dict = {
                "train/lr": self.scheduler.get_last_lr()[0],
                "time/step_s": now - step_started,
                "time/data_wait_s": wait_now - previous_wait,
                "time/unpack_s": self._unpack_seconds - previous_unpack,
                "queue/depth": float(self._feed.depth) if self._feed else 0.0,
            }
            step_started, previous_wait = now, wait_now
            previous_unpack = self._unpack_seconds

            from easy_nn.server.link import gpu_stats

            log_dict.update(gpu_stats())

            if grad_norm is not None:
                log_dict["train/grad_norm"] = float(grad_norm)

            for model in self.models:
                unwrapped = self.accelerator.unwrap_model(model)
                magnitude, count = 0.0, 0
                for param in unwrapped.parameters():
                    if param.requires_grad:
                        magnitude += param.data.abs().sum().item()
                        count += param.numel()
                if count:
                    name = unwrapped.__class__.__name__
                    log_dict[f"train/avg_magnitude/{name}"] = magnitude / count

            for key, value in train_loss_accs.items():
                log_dict[f"train/{key}"] = value

            self.log(log_dict, self.global_step)
            self._link.progress(
                # global_step is still the index of the step just finished.
                step=self.global_step + 1,
                total=total_steps,
                postfix={
                    "loss": round(float(loss.detach()), 4),
                    "gpu": f"{log_dict.get('gpu/allocated_GB', 0.0):.1f}G",
                    "wait": f"{log_dict['time/data_wait_s']:.2f}s",
                },
            )
            train_loss_accs = {}

            self.global_step += 1
            self.epochs_trained = self.global_step // self.steps_per_epoch
            self.steps_in_epoch = self.global_step % self.steps_per_epoch

            self._pump_controls()

            if self._save_requested or (
                self.save_checkpoint_every_steps > 0
                and self.global_step % self.save_checkpoint_every_steps == 0
            ):
                self._save_requested = False
                self._emit_checkpoint(f"step_{self.global_step}")

            if self._eval_requested or (
                self.eval_every_steps > 0
                and self.global_step % self.eval_every_steps == 0
            ):
                self._eval_requested = False
                self._run_eval()

            if self._paused:
                self._wait_while_paused()

            if self._stop_requested:
                self.print(
                    f"{Fore.YELLOW}Stopping at step {self.global_step} "
                    f"on request.{Style.RESET_ALL}"
                )
                break

            if self.global_step >= total_steps:
                break

            step_started = time.perf_counter()

        self._emit_checkpoint(f"step_{self.global_step}_final")
        self.print(f"{Fore.GREEN}Training complete!{Style.RESET_ALL}\n")

    # ================================================================
    #  Executor side -- control, eval, checkpoints
    # ================================================================
    def _pump_controls(self):
        for command in self._link.take_control():
            if command == CTRL_PAUSE:
                self._paused = True
            elif command == CTRL_RESUME:
                self._paused = False
            elif command == CTRL_SAVE:
                self._save_requested = True
            elif command == CTRL_EVAL:
                self._eval_requested = True
            elif command == CTRL_STOP:
                self._stop_requested = True

    def _wait_while_paused(self):
        self.print(f"{Fore.YELLOW}Pausing at step {self.global_step}.{Style.RESET_ALL}")
        for model in list(self.models) + list(self.non_trainable_models):
            model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        while self._paused and not self._stop_requested:
            time.sleep(0.2)
            self._pump_controls()

        for model in list(self.models) + list(self.non_trainable_models):
            model.to(self.accelerator.device)
        self.print(f"{Fore.GREEN}Resuming at step {self.global_step}.{Style.RESET_ALL}")

    def _run_eval(self):
        self.eval_begin(self.global_step)
        blobs = self._feed.eval_blobs() if self._feed is not None else []
        if not blobs:
            self.eval_end(self.global_step)
            return

        self.print(f"\nStarting evaluation at step {self.global_step}...")
        accs, count = {}, 0
        wrapped = list(self.models)
        for i, model in enumerate(wrapped):
            self.models[i] = self.accelerator.unwrap_model(model)
            self.models[i].eval()

        try:
            with torch.no_grad():
                for blob in blobs:
                    for batch in self.unpack(
                        blob, self.accelerator.device, self.weight_dtype
                    ):
                        loss, loss_dict = self.eval_step(
                            self.global_step,
                            batch,
                            device=self.accelerator.device,
                            weight_dtype=self.weight_dtype,
                        )
                        merged = dict(loss_dict or {})
                        if loss is not None:
                            merged["loss"] = loss
                        for key, value in merged.items():
                            accs[key] = accs.get(key, 0.0) + (
                                value.item() if torch.is_tensor(value) else float(value)
                            )
                        count += 1
        finally:
            self.models = wrapped
            for model in self.models:
                model.train()

        if count and accs:
            log_dict = {f"eval/{k}": v / count for k, v in accs.items()}
            self.log(log_dict, self.global_step)
            self.print(f"Evaluation at step {self.global_step}: {log_dict}\n")

        self.eval_end(self.global_step)

    def _emit_checkpoint(self, name):
        payload = dict(self.save_checkpoint(self.global_step) or {})
        payload["trainer_metadata.json"] = json.dumps(
            {
                "global_step": self.global_step,
                "epochs_trained": self.epochs_trained,
                "steps_in_epoch": self.steps_in_epoch,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=4,
        ).encode()
        self._link.checkpoint(name, payload)

    # -- default full-state implementation --------------------------------
    @staticmethod
    def _scratch_dir():
        """Somewhere to let accelerate write.  RAM-backed when we can."""
        if os.path.isdir("/dev/shm"):
            try:
                return tempfile.mkdtemp(prefix="easy_nn_ckpt_", dir="/dev/shm")
            except OSError:
                pass
        return tempfile.mkdtemp(prefix="easy_nn_ckpt_")

    def _full_state_payload(self) -> dict:
        directory = self._scratch_dir()
        try:
            self.accelerator.save_state(directory)
            payload = {}
            for root, _, files in os.walk(directory):
                for filename in files:
                    path = os.path.join(root, filename)
                    rel = os.path.relpath(path, directory).replace(os.sep, "/")
                    with open(path, "rb") as handle:
                        payload[rel] = handle.read()
            return payload
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def _load_full_state(self, payload: dict) -> None:
        directory = self._scratch_dir()
        try:
            for rel, data in payload.items():
                if rel == "trainer_metadata.json":
                    meta = json.loads(bytes(data))
                    self.global_step = meta.get("global_step", 0)
                    self.epochs_trained = meta.get("epochs_trained", 0)
                    self.steps_in_epoch = meta.get("steps_in_epoch", 0)
                    continue
                path = os.path.join(directory, *rel.split("/"))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as handle:
                    handle.write(bytes(data))
            self.accelerator.load_state(directory)
        finally:
            shutil.rmtree(directory, ignore_errors=True)
