import argparse

import numpy as np
import torch

# ensure reproducible results
np.random.seed(0)
torch.manual_seed(0)


def mean(lst: list) -> float:
    return sum(lst) / (len(lst) or 1)


def save_checkpoint(state: dict, is_best: bool, path: str) -> None:
    torch.save(state, path + "/checkpoint")
    if is_best:
        torch.save(state, path + "/best_model")


parser = argparse.ArgumentParser()

parser.add_argument(
    "--task",
    default="classification",
    type=str,
    help="'regression' or 'classification'",
)
parser.add_argument(
    "--csv-path",
    default="data/supercon/labeled.csv",
    type=str,
    help="Path relative to project root to the CSV file holding the training data",
)
parser.add_argument(
    "--epochs", default=10, type=int, help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    help="manual epoch number (useful on restarts)",
)
parser.add_argument("--batch-size", default=128, type=int, help="train batchsize")
parser.add_argument(
    "--bootstrap-idx",
    default=0,
    type=int,
    help="index of sample to leave out of training set",
)
parser.add_argument(
    "--lr", "--learning-rate", default=1e-3, type=float, help="initial learning rate"
)
# Checkpoints
parser.add_argument(
    "--resume", default="", type=str, help="path to latest checkpoint (default: none)"
)
parser.add_argument(
    "--robust", action="store_true", help="Whether to use a heteroscedastic loss"
)
parser.add_argument(
    "--verbose", action="store_true", help="Whether to print metrics during training"
)

parser.add_argument("--out-dir", default="", help="Directory to output the result")

# joggle was found to be optimal at 0.03 Angstrom (https://arxiv.org/abs/2012.02920)
# anything larger than 0.1 probably doesn't make sense given typical bond
# lengths of 1 - 2 Angstroms in typical crystals
parser.add_argument(
    "--joggle",
    default=0,
    type=float,
    help="how many Angstroms to randomly perturb atom positions in a crystal",
)
