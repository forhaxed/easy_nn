"""
A toy job used by the integration tests.

It deliberately lives in a module rather than in ``__main__``: that is the case
cloudpickle gets wrong by default, so if ``job.register_local_modules`` ever
stops working, every test in this suite fails with ModuleNotFoundError from the
executor.
"""

import time

import torch

from easy_nn import DataSource, Trainer


class ToyData(DataSource):
    def setup(self):
        generator = torch.Generator().manual_seed(self.options.get("seed", 0))
        n = self.options.get("n", 256)
        self.x = torch.randn(n, 4, generator=generator)
        self.y = self.x.sum(dim=1)
        self.batch_size = self.options.get("batch_size", 8)

    def __len__(self):
        return len(self.x)

    def stream(self):
        for start in range(0, len(self.x), self.batch_size):
            stop = start + self.batch_size
            yield {"x": self.x[start:stop], "y": self.y[start:stop]}


class ToyTrainer(Trainer):
    #: Seconds to stall in unpack, to make backpressure and locks observable.
    unpack_delay = 0.0

    def unpack(self, blob, device, weight_dtype):
        if self.unpack_delay:
            time.sleep(self.unpack_delay)
        return [
            {"x": b["x"].to(device), "y": b["y"].to(device)} for b in blob
        ]

    def train_step(self, step, batch, device, weight_dtype):
        prediction = self.models[0](batch["x"]).squeeze(-1)
        loss = torch.nn.functional.mse_loss(prediction, batch["y"])
        return loss, {}


class ExplodingTrainer(ToyTrainer):
    def train_step(self, step, batch, device, weight_dtype):
        raise ValueError("deliberate failure inside train_step")


class CustomLoopTrainer(ToyTrainer):
    """Proves a hand-written loop travels and runs on the executor."""

    def training_loop(self):
        seen = 0
        for batch in self.batches():
            with self.accelerator.accumulate(self.models):
                loss, _ = self.train_step(
                    seen, batch, self.device, self.weight_dtype
                )
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.log({"custom/loss": loss.detach()}, seen)
            seen += 1
        self.global_step = seen
        self.print(f"custom loop ran {seen} batches")


def build(tmp_path, trainer_class=ToyTrainer, **overrides):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)
    )

    trainer = trainer_class(output_dir=str(tmp_path / "output"))
    trainer.models = [model]
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    trainer.data = ToyData(n=256, batch_size=8)
    trainer.batch_size = 8
    trainer.epochs = 1
    trainer.gradient_accumulation_steps = 1
    trainer.precache_size = 4
    trainer.blob_buffer = 2
    trainer.mixed_precision = "no"
    for key, value in overrides.items():
        setattr(trainer, key, value)
    return trainer
