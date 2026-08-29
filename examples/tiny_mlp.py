"""
The smallest complete easy_nn job: fit a small MLP to a synthetic function.

Run it with::

    python examples/tiny_mlp.py

Nothing here is aware of where the training happens.  Swap ``Local()`` for a
``RunPod(...)`` and the same file trains on a rented GPU.
"""

import argparse

import torch

from easy_nn import DataSource, Local, Trainer

TRUE_W = torch.tensor([2.0, -3.0, 0.5, 1.5])


# ══════════════════════ LOCAL ══════════════════════
# Owns the data.  This class never leaves your machine.
class SyntheticData(DataSource):
    def setup(self):
        generator = torch.Generator().manual_seed(self.options.get("seed", 0))
        n = self.options.get("n", 2048)
        self.x = torch.randn(n, 4, generator=generator)
        self.y = self.x @ TRUE_W + 0.1 * torch.randn(n, generator=generator)
        self.batch_size = self.options.get("batch_size", 32)

    def __len__(self):
        return len(self.x)

    def stream(self):
        for start in range(0, len(self.x), self.batch_size):
            stop = start + self.batch_size
            yield {"x": self.x[start:stop], "y": self.y[start:stop]}

    def pack(self, batches):
        # Send fp16 on the wire; the executor casts back.  Halves the traffic.
        return [
            {"x": b["x"].to(torch.float16), "y": b["y"].to(torch.float16)}
            for b in batches
        ]


# ══════════════════ ON THE EXECUTOR ══════════════════
# The whole class travels; the executor has never seen this code.
class MLPTrainer(Trainer):
    def unpack(self, blob, device, weight_dtype):
        return [
            {"x": b["x"].to(device, torch.float32), "y": b["y"].to(device, torch.float32)}
            for b in blob
        ]

    def train_step(self, step, batch, device, weight_dtype):
        prediction = self.models[0](batch["x"]).squeeze(-1)
        loss = torch.nn.functional.mse_loss(prediction, batch["y"])
        return loss, {"mse": loss.detach()}


def build_trainer(output_dir="./output"):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)
    )

    trainer = MLPTrainer(output_dir=output_dir)
    trainer.models = [model]
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    trainer.data = SyntheticData(n=2048, batch_size=32, seed=0)
    trainer.eval_data = SyntheticData(n=256, batch_size=32, seed=99)

    trainer.batch_size = 32
    trainer.epochs = 3
    trainer.gradient_accumulation_steps = 2
    trainer.precache_size = 4
    trainer.blob_buffer = 2
    trainer.seed = 0
    trainer.save_checkpoint_every_steps = 50
    trainer.eval_every_steps = 50
    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./output")
    args = parser.parse_args()

    trainer = build_trainer(args.output)
    result = trainer.train(on=Local())

    print(f"\nfinished at step {result['global_step']}")
    print(f"tensorboard --logdir {result['log_dir']}")
    for path in result["checkpoints"]:
        print(f"  checkpoint: {path}")


if __name__ == "__main__":
    main()
