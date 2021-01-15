import gzip
import os
import pickle
from functools import lru_cache
from os.path import abspath, dirname

import numpy as np
import torch
from torch.utils.data import Dataset

from supercon.core import Featurizer

# absolute path to the project's root directory
ROOT = dirname(dirname(abspath(__file__)))


class CrystalGraphData(Dataset):
    """Dataset wrapper for crystal structure data in CIF format."""

    def __init__(
        self,
        df,
        task,
        fea_path=f"{ROOT}/data/cgcnn-embedding.json",
        struct_path=f"{ROOT}/data/structures",
        max_num_nbr=12,
        radius=8,
        dmin=0,
        step=0.2,
    ):
        """
        Args:
            df (str): dataframe with materials to train
                expected cols: [material_id, formula/composition, target(s)]
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
        """

        assert os.path.exists(fea_path), f"fea_path={fea_path} does not exist!"
        assert os.path.exists(struct_path), f"struct_path={struct_path} does not exist!"

        self.df = df
        self.struct_path = struct_path
        self.ari = Featurizer.from_json(fea_path)
        self.elem_emb_len = self.ari.embedding_size

        self.max_num_nbr = max_num_nbr
        self.radius = radius

        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.nbr_fea_len = self.gdf.embedding_size

        self.task = task
        self.n_targets = self.df.label.max() + 1 if task == "classification" else 1

    def __len__(self):
        return len(self.df)

    @lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        """
        Returns:
            atom_fea: torch.Tensor(n_i, atom_fea_len)
            nbr_fea: torch.Tensor(n_i, M, nbr_fea_len)
            self_fea_idx: torch.LongTensor(n_i, M)
            neighbor_fea_idx: torch.LongTensor(n_i, M)
            target: torch.Tensor(1)
            comp: str
            cif_id: str or int
        """
        material_id, formula, target = self.df.iloc[idx][:3]

        # NOTE getting primitive structure before constructing graph
        # significantly harms the performance of this model.
        with gzip.open(f"{self.struct_path}/{material_id}.zip", "rb") as file:
            crystal = pickle.loads(file.read())

        # atom features
        atom_fea = [atom.specie.symbol for atom in crystal]

        # neighbors
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        self_fea_idx, neighbor_fea_idx, nbr_fea = [], [], []

        for i, nbr in enumerate(all_nbrs):
            # NOTE due to using a geometric learning library we do not
            # need to set a maximum number of neighbors but do so in
            # order to replicate the original code.
            if len(nbr) < self.max_num_nbr:
                neighbor_fea_idx.extend([x[2] for x in nbr])
                nbr_fea.extend([x[1] for x in nbr])
            else:
                neighbor_fea_idx.extend([x[2] for x in nbr[: self.max_num_nbr]])
                nbr_fea.extend([x[1] for x in nbr[: self.max_num_nbr]])

            if len(nbr) == 0:
                raise ValueError(
                    f"Isolated atom found in {material_id} ({formula}) - "
                    "increase maximum radius or remove structure"
                )
            self_fea_idx.extend([i] * min(len(nbr), self.max_num_nbr))

        nbr_fea = np.array(nbr_fea)

        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = np.vstack([self.ari.get_fea(atom) for atom in atom_fea])

        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        self_fea_idx = torch.LongTensor(self_fea_idx)
        neighbor_fea_idx = torch.LongTensor(neighbor_fea_idx)

        if self.task == "regression":
            target = torch.Tensor([float(target)])
        elif self.task == "classification":
            target = torch.LongTensor([target])

        return (
            (atom_fea, nbr_fea, self_fea_idx, neighbor_fea_idx),
            target,
            formula,
            material_id,
        )


class GaussianDistance:
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
            Minimum interatomic distance
        dmax: float
            Maximum interatomic distance
        step: float
            Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        self.embedding_size = len(self.filter)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian distance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
            A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
            Expanded distance matrix with the last dimension of length
            len(self.filter)
        """
        return np.exp(-((distances[..., None] - self.filter) ** 2) / self.var ** 2)


def collate_batch(dataset_list):
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
        cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
        Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
        Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
        Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea = [], []
    batch_self_fea_idx, batch_nbr_fea_idx = [], []
    crystal_atom_idx, batch_target = [], []
    batch_comps = []
    batch_cif_ids = []
    base_idx = 0
    for (i, data) in enumerate(dataset_list):
        (atom_fea, nbr_fea, self_fea_idx, neighbor_fea_idx), target, comp, cif_id = data

        n_i = atom_fea.shape[0]  # number of atoms for this crystal

        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_self_fea_idx.append(self_fea_idx + base_idx)
        batch_nbr_fea_idx.append(neighbor_fea_idx + base_idx)

        crystal_atom_idx.extend([i] * n_i)
        batch_target.append(target)
        batch_comps.append(comp)
        batch_cif_ids.append(cif_id)
        base_idx += n_i

    return (
        (
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_self_fea_idx, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.LongTensor(crystal_atom_idx),
        ),
        torch.stack(batch_target, dim=0).squeeze(1),
        batch_comps,
        batch_cif_ids,
    )
