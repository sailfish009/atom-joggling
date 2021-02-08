import gzip
import json
import pickle
from functools import lru_cache
from os.path import abspath, dirname, exists

import numpy as np
import torch
from torch.utils.data import Dataset

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

        assert exists(fea_path), f"fea_path='{fea_path}' does not exist!"
        assert exists(struct_path), f"struct_path='{struct_path}' does not exist!"

        self.df = df
        self.struct_path = struct_path
        self.ari = Featurizer.from_json(fea_path)
        self.elem_emb_len = self.ari.embedding_size

        self.max_num_nbr = max_num_nbr
        self.radius = radius

        self.gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)
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

        features = (atom_fea, nbr_fea, self_fea_idx, neighbor_fea_idx)

        return features, target, formula, material_id


class GaussianDistance:
    """ Expands distances by a Gaussian basis. (unit: angstrom) """

    def __init__(self, dmin, dmax, step, var=None):
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

    def expand(self, distances):
        """Apply Gaussian distance filter to a numpy distance array.

        Args:
            distances (np.array): an n-dim. distance matrix of any shape

        Returns:
            np.array: an (n+1)-dim. expanded distance matrix with the last
            dimension of length len(self.filter)
        """
        return np.exp(-((distances[..., None] - self.filter) ** 2) / self.var ** 2)


def collate_batch(batch, use_cuda=False):
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
    cif_ids: list
    """
    batch_self_fea_idx, batch_nbr_fea_idxs, crystal_atom_idx = [], [], []
    base_idx = 0
    features, targets, compositions, cif_ids = zip(*batch)

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
    return out_features, targets, compositions, cif_ids


class Featurizer:
    """Base class for featurizing nodes and edges."""

    def __init__(self, allowed_types):
        self.allowed_types = set(allowed_types)
        self._embedding = {}

    def get_fea(self, key):
        assert key in self.allowed_types, f"{key} is not an allowed atom type"
        return self._embedding[key]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.allowed_types = set(self._embedding.keys())

    def get_state_dict(self):
        return self._embedding

    @property
    def embedding_size(self):
        return len(self._embedding[list(self._embedding.keys())[0]])

    @classmethod
    def from_json(cls, embedding_file):
        with open(embedding_file) as file:
            embedding = json.load(file)
        allowed_types = set(embedding.keys())
        instance = cls(allowed_types)
        for key, value in embedding.items():
            instance._embedding[key] = np.array(value, dtype=float)
        return instance
