import json
from functools import lru_cache
from os.path import exists
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from torch import LongTensor, Tensor
from torch.utils.data import Dataset

from atom_joggling.utils import ROOT


class OldCrystalGraphData(Dataset):
    """Dataset wrapper for crystal structure data in CIF format."""

    def __init__(
        self,
        df: pd.DataFrame,
        task: str,
        fea_path: str = f"{ROOT}/data/cgcnn-embedding.json",
        struct_path: str = f"{ROOT}/data/structures",
        max_num_nbr: int = 5,
        radius: int = 5,
        dmin: int = 0,
        step: float = 0.2,
        joggle: float = 0,
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
            joggle (int, optional): By how many Angstroms to randomly perturb the atom
                positions in a crystal. May improve model robustness. Defaults to 0.
        """

        assert exists(fea_path), f"{fea_path=} does not exist!"
        assert exists(struct_path), f"{struct_path=} does not exist!"

        self.df = df
        self.struct_path = struct_path
        self.ari = GraphFeaturizer.from_json(fea_path)
        self.elem_emb_len = self.ari.embedding_size

        self.max_num_nbr = max_num_nbr
        self.radius = radius

        self.gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)
        self.nbr_fea_len = self.gdf.embedding_size

        self.task = task
        self.joggle = joggle

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[tuple, Tensor, str, str]:
        """
        Returns:
            features (tuple):
            - atom_fea: Tensor(n_i, atom_fea_len)
            - nbr_fea: Tensor(n_i, M, nbr_fea_len)
            - atom_indices: LongTensor(n_i, M)
            - neighbor_fea_idx: LongTensor(n_i, M)
            target: Tensor(1)
            composition: str
            material_id: str
        """
        material_id, composition, target = self.df.iloc[idx][:3]

        # NOTE getting primitive structure before constructing graph
        # significantly harms the performance of this model.
        # in principle, we should convert to the primitive unit cell before
        # constructing the structure graph as the graph of the primitive cell
        # encodes the system without loss, but in practice this seems to cause
        # weird bugs in Pymatgen
        crystal = load_struct(f"{self.struct_path}/{material_id}.json")

        # https://pymatgen.org/pymatgen.core.structure.html#pymatgen.core.structure.IStructure.get_neighbor_list
        # get_neighbor_list returns: center_indices, points_indices, offset_vectors, distances
        atom_indices, neighbor_indices, _, nbr_distances = crystal.get_neighbor_list(
            self.radius
        )

        if self.joggle:
            crystal.perturb(self.joggle)

        nbr_distances = self.gdf.expand(nbr_distances)

        # atom features
        atom_fea = [atom.specie.symbol for atom in crystal]
        atom_fea = np.vstack([self.ari.get_fea(atom) for atom in atom_fea])

        atom_fea = Tensor(atom_fea)
        nbr_distances = Tensor(nbr_distances)
        atom_indices = LongTensor(atom_indices)
        neighbor_indices = LongTensor(neighbor_indices)

        if self.task == "regression":
            target = Tensor([float(target)])
        elif self.task == "classification":
            target = LongTensor([target])

        features = (atom_fea, nbr_distances, atom_indices, neighbor_indices)

        return features, target, composition, material_id


class CrystalGraphData(Dataset):
    """Dataset wrapper for crystal structure data in CIF format."""

    def __init__(
        self,
        task: str,
        df: pd.DataFrame,
        targets: List[str],
        struct_col: str = "structure",
        identifiers=["material_id", "composition"],
        fea_path: str = f"{ROOT}/data/cgcnn-embedding.json",
        max_num_nbr: int = 12,
        radius: int = 5,
        dmin: int = 0,
        step: float = 0.2,
        joggle: float = 0,
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
            joggle (int, optional): By how many Angstroms to randomly perturb the atom
                positions in a crystal. May improve model robustness. Defaults to 0.
        """

        assert exists(fea_path), f"{fea_path=} does not exist!"

        self.df = df
        self.ari = GraphFeaturizer.from_json(fea_path)
        self.elem_emb_len = self.ari.embedding_size

        self.max_num_nbr = max_num_nbr
        self.radius = radius

        self.gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)
        self.nbr_fea_len = self.gdf.embedding_size

        self.task = task
        self.struct_col = struct_col
        self.targets = targets
        self.identifiers = identifiers

        self.joggle = joggle

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[tuple, Tensor, str, str]:
        """
        Returns:
            features (tuple):
            - atom_fea: Tensor(n_i, atom_fea_len)
            - nbr_fea: Tensor(n_i, M, nbr_fea_len)
            - atom_indices: LongTensor(n_i, M)
            - neighbor_fea_idx: LongTensor(n_i, M)
            target: Tensor(1)
            composition: str
            material_id: str
        """
        series = self.df.iloc[idx]
        crystal = series[self.struct_col]
        targets = series[self.targets]

        # https://pymatgen.org/pymatgen.core.structure.html#pymatgen.core.structure.IStructure.get_neighbor_list
        # get_neighbor_list returns: center_indices, points_indices, offset_vectors, distances
        atom_idxs, neighbor_idxs, _, nbr_distances = crystal.get_neighbor_list(
            self.radius
        )

        if self.joggle:
            crystal.perturb(self.joggle)

        nbr_distances = self.gdf.expand(nbr_distances)

        # atom features
        atom_fea = [atom.specie.symbol for atom in crystal]
        atom_fea = np.vstack([self.ari.get_fea(atom) for atom in atom_fea])

        atom_fea = Tensor(atom_fea)
        nbr_distances = Tensor(nbr_distances)
        atom_idxs = LongTensor(atom_idxs)
        neighbor_idxs = LongTensor(neighbor_idxs)

        if self.task == "regression":
            target = Tensor([float(targets)])
        elif self.task == "classification":
            target = LongTensor([targets])

        features = (atom_fea, nbr_distances, atom_idxs, neighbor_idxs)

        return features, target, *series[self.identifiers]


@lru_cache(maxsize=None)  # Cache loaded structures
def load_struct(filepath: str) -> Structure:
    """Load a Pymatgen structure (in CIF format) from disk."""
    return Structure.from_file(filepath)


class GaussianDistance:
    """Expands distances by a Gaussian basis. (unit: angstrom)"""

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


def collate_batch(batch: list, use_cuda: bool = False) -> tuple:
    """Collate a batch of samples into tensors for predicting crystal properties.

    Args:
        batch (list): list of tuples for each sample containing
            - atom_fea: Tensor shape [n_i, atom_fea_len]
            - nbr_fea: Tensor shape [n_i, M, nbr_fea_len]
            - neighbor_fea_idx: LongTensor shape [n_i, M]
            - target: Tensor shape [1]
            - material_id: str
        use_cuda (bool, optional): [description]. Defaults to False.

    Returns:
        tuple: contains
            atom_feas: Tensor shape [N, orig_atom_fea_len]
                Atom features from atom type
            nbr_feas: Tensor shape [N, M, nbr_fea_len]
                Bond features of each atom's M neighbors
            neighbor_indices: LongTensor shape [N, M]
                Indices of M neighbors of each atom
            crystal_atom_idx: list of LongTensors of length N0
                Mapping from the crystal idx to atom idx
            target: Tensor shape [N, 1]
                Target value for prediction
            material_ids: list[str]
        where N = sum(n_i); N0 = sum(i)
    """
    batch_atom_indices, batch_neighbor_indices, crystal_atom_idx = [], [], []
    base_idx = 0
    features, targets, *rest = zip(*batch)

    atom_feas, nbr_feas, atom_indices, neighbor_indices = zip(*features)
    for idx, atom_fea in enumerate(atom_feas):

        batch_atom_indices.append(atom_indices[idx] + base_idx)
        batch_neighbor_indices.append(neighbor_indices[idx] + base_idx)

        n_atoms = atom_fea.shape[0]  # number of atoms for this crystal
        crystal_atom_idx.extend([idx] * n_atoms)
        base_idx += n_atoms

    out_features = (
        torch.cat(atom_feas, dim=0),
        torch.cat(nbr_feas, dim=0),
        torch.cat(batch_atom_indices, dim=0),
        torch.cat(batch_neighbor_indices, dim=0),
        torch.LongTensor(crystal_atom_idx),
    )

    targets = torch.cat(targets, dim=0)

    if use_cuda:
        out_features = [tensor.cuda() for tensor in out_features]
        targets = targets.cuda()

    return out_features, targets, *rest


class GraphFeaturizer:
    """Base class for featurizing nodes and edges in a crystal graph."""

    def __init__(self, allowed_types: Iterable[str]) -> None:
        """
        Args:
            allowed_types (Iterable[str]): names of element names for which
            to store embeddings
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
        return len(list(self._embedding.values())[0])

    @classmethod
    def from_json(cls, embedding_file: str) -> "GraphFeaturizer":
        with open(embedding_file) as file:
            embedding = json.load(file)
        allowed_types = set(embedding.keys())
        instance = cls(allowed_types)
        for key, value in embedding.items():
            instance._embedding[key] = np.array(value, dtype=float)
        return instance
