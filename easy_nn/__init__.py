"""
easy_nn -- train anything, anywhere.

Your script is the brain: it owns the dataset, the logs and the checkpoints.
The executor is muscle: it gets the code, the weights and a stream of data, and
gives back losses and checkpoints without writing a thing to its own disk.

    from easy_nn import Trainer, DataSource, Local

    class MyData(DataSource): ...
    class MyTrainer(Trainer): ...

    t = MyTrainer(output_dir="./output")
    t.data = MyData()
    t.train(on=Local())
"""

from easy_nn.datasource import DataSource, IterableSource
from easy_nn.executors.local import Local
from easy_nn.executors.runpod import RunPod
from easy_nn.trainer import Trainer

__all__ = ["Trainer", "DataSource", "IterableSource", "Local", "RunPod"]
__version__ = "0.1.0"
