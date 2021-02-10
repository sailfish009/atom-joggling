import gzip
import json
import pickle
from functools import lru_cache
from os.path import abspath, dirname, exists
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# absolute path to the project's root directory
ROOT = dirname(dirname(abspath(__file__)))


class Normalizer:
    """Normalize a tensor and restore it later."""

    def __init__(self) -> None:
        self.mean = None
        self.std = None

    def fit(self, tensor: torch.Tensor, dim: int = 0) -> None:
        self.mean = tensor.mean(dim)
        self.std = tensor.std(dim)
        assert (self.std != 0).all(), "self.std has 0 entries, cannot divide by 0"

    def norm(self, tensor: torch.Tensor) -> torch.Tensor:
        assert self.is_fit, "Normalizer must be fit first"
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor: torch.Tensor) -> torch.Tensor:
        assert self.is_fit, "Normalizer must be fit first"
        return normed_tensor * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict: dict) -> None:
        self.mean = state_dict["mean"].cpu()
        self.std = state_dict["std"].cpu()

    @property
    def is_fit(self) -> bool:
        return [self.mean, self.std] != [None, None]


class CrystalGraphData(Dataset):
    """Dataset wrapper for crystal structure data in CIF format."""

    def __init__(
        self,
        df: pd.DataFrame,
        task: str,
        fea_path: str = f"{ROOT}/data/cgcnn-embedding.json",
        struct_path: str = f"{ROOT}/data/structures",
        max_num_nbr: int = 12,
        radius: int = 5,
        dmin: int = 0,
        step: float = 0.2,
        normalizer=Normalizer(),
    ) -> None:
        """
        Args:
            df (pd.DataFrame): Dataframe expected to have the following
                columns: [material_id, composition, targets]
            fea_path (str): The path to the element embedding
            task (str): "regression" or "classification"
            max_num_nbr (int, optional): The maximum number of neighbors while
                constructing the crystal graph. Defaults to 12.
            radius (int, optional): The cutoff radius for searching neighbors.
                Defaults to 8.
            dmin (int, optional): The minimum distance for constructing
                GaussianDistance. Defaults to 0.
            step (float, optional): The step size for constructing GaussianDistance.
                Defaults to 0.2.
            normalizer (Normalizer): For z-scoring target data (zero mean, unit std).
                Defaults to None.
        """

        assert exists(fea_path), f"fea_path='{fea_path}' does not exist!"
        assert exists(struct_path), f"struct_path='{struct_path}' does not exist!"

        self.df = df
        self.struct_path = struct_path
        self.ari = GraphFeaturizer.from_json(fea_path)
        self.elem_emb_len = self.ari.embedding_size

        self.max_num_nbr = max_num_nbr
        self.radius = radius

        self.gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)
        self.nbr_fea_len = self.gdf.embedding_size

        self.task = task
        targets = df.iloc[:, 2]
        self.n_targets = targets.max() + 1 if task == "classification" else 1

        self.normalizer = None if task == "classification" else normalizer
        if normalizer is not None and not normalizer.is_fit:
            normalizer.fit(targets)

    def __len__(self) -> int:
        return len(self.df)

    @lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx: int) -> Tuple[tuple, torch.Tensor, str, str]:
        """
        Returns:
            features (tuple):
            - atom_fea: torch.Tensor(n_i, atom_fea_len)
            - nbr_fea: torch.Tensor(n_i, M, nbr_fea_len)
            - self_fea_idx: torch.LongTensor(n_i, M)
            - neighbor_fea_idx: torch.LongTensor(n_i, M)
            target: torch.Tensor(1)
            comp: str
            material_id: str
        """
        material_id, formula, target = self.df.iloc[idx][:3]

        # NOTE getting primitive structure before constructing graph
        # significantly harms the performance of this model.
        with gzip.open(f"{self.struct_path}/{material_id}.zip", "rb") as file:
            crystal = pickle.loads(file.read())

        # atom features
        atom_fea = [atom.specie.symbol for atom in crystal]

        # neighbors
        self_fea_idx, neighbor_fea_idx, _, nbr_fea = crystal.get_neighbor_list(
            self.radius,
            numerical_tol=1e-8,
        )

        nbr_fea = np.array(nbr_fea)

        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = np.vstack([self.ari.get_fea(atom) for atom in atom_fea])

        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        self_fea_idx = torch.LongTensor(self_fea_idx)
        neighbor_fea_idx = torch.LongTensor(neighbor_fea_idx)

        if self.task == "regression":
            target = self.normalizer.norm(target)
            target = torch.Tensor([float(target)])
        elif self.task == "classification":
            target = torch.LongTensor([target])

        features = (atom_fea, nbr_fea, self_fea_idx, neighbor_fea_idx)

        return features, target, formula, material_id


class GaussianDistance:
    """ Expands distances by a Gaussian basis. (unit: angstrom) """

    def __init__(
        self, dmin: float, dmax: float, step: float, var: float = None
    ) -> None:
        """
        Args:
            dmin (float): Minimum interatomic distance
            dmax (float): Maximum interatomic distance
            step (float): Step size for the Gaussian filter
            var (float, optional): variance of the Gaussian filter. Defaults to step.
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        self.embedding_size = len(self.filter)
        self.var = var or step

    def expand(self, distances: np.ndarray) -> np.ndarray:
        """Apply Gaussian distance filter to a numpy distance array.

        Args:
            distances (np.array): an n-dim. distance matrix of any shape

        Returns:
            np.array: an (n+1)-dim. expanded distance matrix with the last
            dimension of length len(self.filter)
        """
        return np.exp(-((distances[..., None] - self.filter) ** 2) / self.var ** 2)


def collate_batch(batch: tuple, use_cuda: bool = False) -> tuple:
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
        (atom_fea, nbr_fea, neighbor_fea_idx, target)

        atom_fea: torch.Tensor shape (n_i, atom_fea_len)
        nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
        neighbor_fea_idx: torch.LongTensor shape (n_i, M)
        target: torch.Tensor shape (1, )
        material_id: str

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    atom_feas: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
    nbr_feas: torch.Tensor shape (N, M, nbr_fea_len)
        Bond features of each atom's M neighbors
    nbr_fea_idxs: torch.LongTensor shape (N, M)
        Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
        Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
        Target value for prediction
    material_ids: list
    """
    batch_self_fea_idx, batch_nbr_fea_idxs, crystal_atom_idx = [], [], []
    base_idx = 0
    features, targets, compositions, material_ids = zip(*batch)

    atom_feas, nbr_feas, self_fea_idxs, neighbor_fea_idxs = zip(*features)
    for (idx, atom_fea) in enumerate(atom_feas):

        batch_self_fea_idx.append(self_fea_idxs[idx] + base_idx)
        batch_nbr_fea_idxs.append(neighbor_fea_idxs[idx] + base_idx)

        n_atoms = atom_fea.shape[0]  # number of atoms for this crystal
        crystal_atom_idx.extend([idx] * n_atoms)
        base_idx += n_atoms

    out_features = (
        torch.cat(atom_feas, dim=0),
        torch.cat(nbr_feas, dim=0),
        torch.cat(batch_self_fea_idx, dim=0),
        torch.cat(batch_nbr_fea_idxs, dim=0),
        torch.LongTensor(crystal_atom_idx),
    )
    targets = torch.cat(targets, dim=0)
    if use_cuda:
        out_features = [t.cuda() for t in out_features]
        targets = targets.cuda()
    return out_features, targets, compositions, material_ids


class GraphFeaturizer:
    """Base class for featurizing nodes and edges in a crystal graph."""

    def __init__(self, allowed_types: Iterable[str]) -> None:
        """
        Args:
            allowed_types (Iterable[str]): element names for which embeddings
                are available
        """
        self.allowed_types = set(allowed_types)
        self._embedding = {}

    def get_fea(self, key: str) -> np.ndarray:
        assert key in self.allowed_types, f"{key} is not an allowed atom type"
        return self._embedding[key]

    def load_state_dict(self, state_dict: dict) -> None:
        self._embedding = state_dict
        self.allowed_types = set(self._embedding.keys())

    def get_state_dict(self) -> dict:
        return self._embedding

    @property
    def embedding_size(self) -> int:
        return len(self._embedding[list(self._embedding.keys())[0]])

    @classmethod
    def from_json(cls, embedding_file: str) -> "GraphFeaturizer":
        with open(embedding_file) as file:
            embedding = json.load(file)
        allowed_types = set(embedding.keys())
        instance = cls(allowed_types)
        for key, value in embedding.items():
            instance._embedding[key] = np.array(value, dtype=float)
        return instance
