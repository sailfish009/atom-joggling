import argparse

import numpy as np
import torch

# ensure reproducible results
np.random.seed(0)
torch.manual_seed(0)


def mean(lst: list) -> float:
    return sum(lst) / len(lst)


def save_checkpoint(state: dict, is_best: bool, path: str) -> None:
    torch.save(state, path + "/checkpoint")
    if is_best:
        torch.save(state, path + "/best_model")


parser = argparse.ArgumentParser(description="PyTorch MixMatch Training")
# Optimization options
parser.add_argument(
    "--epochs",
    default=10,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--batch-size", default=32, type=int, metavar="N", help="train batchsize"
)
parser.add_argument(
    "--bootstrap-idx",
    default=0,
    type=int,
    metavar="I",
    help="index of sample to leave out of training set",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=1e-3,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
# Checkpoints
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--robust", action="store_true", help="Whether to use a heteroscedastic loss"
)

parser.add_argument("--out-dir", default="", help="Directory to output the result")
