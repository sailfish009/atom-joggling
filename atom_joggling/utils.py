import argparse
from os.path import abspath, dirname

import numpy as np
import torch


# absolute path to the project's root directory
ROOT = dirname(dirname(abspath(__file__)))


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
    "-t", "--task", choices=("regression", "classification"), default="classification"
)
parser.add_argument(
    "--data-path",
    default="data/atom_joggling/labeled.csv",
    help="Relative path to CSV file with training data",
)
parser.add_argument("--target", help="Name(s) of the dataframe target column(s)")
parser.add_argument(
    "-e", "--epochs", type=int, default=10, help="number of epochs to run"
)
parser.add_argument(
    "--start-epoch",
    type=int,
    default=0,
    help="manual starting epoch (useful on restarts)",
)
parser.add_argument("--batch-size", type=int, default=128, help="training batch size")
parser.add_argument(
    "--bootstrap-idx",
    type=int,
    default=0,
    help="index of sample to leave out of training set",
)
parser.add_argument(
    "--learning-rate", type=float, default=1e-3, help="initial learning rate"
)
# Checkpoints
parser.add_argument(
    "--resume",
    default="",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--robust", action="store_true", help="Whether to use a heteroscedastic loss"
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Whether to print metrics while training",
)

parser.add_argument("--out-dir", default="", help="Directory for saving results")

# Joggles the position of atoms in a crystal lattice.
# Was found to be optimal at 0.03 Angstrom (https://arxiv.org/abs/2012.02920)
# Anything larger than 0.1 probably doesn't make sense given typical bond
# lengths of 1 - 2 Angstroms in typical crystals.
parser.add_argument(
    "--joggle",
    type=float,
    default=0,
    help="how many Angstroms to randomly perturb atom positions in a crystal",
)
