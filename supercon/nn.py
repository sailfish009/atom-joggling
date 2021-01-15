from typing import List

import torch
from torch import nn
from torch.nn.functional import softplus
from torch_scatter import scatter_add
from torch_scatter.scatter import scatter_mean


class ConvLayer(nn.Module):
    """ Convolutional operation on graphs """

    def __init__(self, elem_fea_len, nbr_fea_len):
        """
        Args:
            elem_fea_len (int): Number of atom hidden features
            nbr_fea_len (int): Number of bond features
        """
        super().__init__()
        self.elem_fea_len = elem_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(
            2 * self.elem_fea_len + self.nbr_fea_len, 2 * self.elem_fea_len
        )
        self.bn1 = nn.BatchNorm1d(2 * self.elem_fea_len)
        self.bn2 = nn.BatchNorm1d(self.elem_fea_len)

    def forward(self, atom_in_fea, nbr_fea, self_fea_idx, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, elem_fea_len)
            Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
            Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
            Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, elem_fea_len)
            Atom hidden features after convolution

        """
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        atom_self_fea = atom_in_fea[self_fea_idx, :]

        total_fea = torch.cat([atom_self_fea, atom_nbr_fea, nbr_fea], dim=1)

        total_fea = self.fc_full(total_fea)
        total_fea = self.bn1(total_fea)

        filter_fea, core_fea = total_fea.chunk(2, dim=1)
        filter_fea = torch.sigmoid(filter_fea)
        core_fea = softplus(core_fea)

        # take the elementwise product of the filter and core
        nbr_msg = filter_fea * core_fea
        nbr_sumed = scatter_add(nbr_msg, self_fea_idx, dim=0)  # sum pooling

        nbr_sumed = self.bn2(nbr_sumed)
        out = softplus(atom_in_fea + nbr_sumed)

        return out


class DescriptorNet(nn.Module):
    """
    The Descriptor Network is the message passing section of the
    CrystalGraphConvNet Model.
    """

    def __init__(self, elem_emb_len, nbr_fea_len, elem_fea_len=64, n_graph=4):
        super().__init__()

        self.embedding = nn.Linear(elem_emb_len, elem_fea_len)

        self.convs = nn.ModuleList(
            [ConvLayer(elem_fea_len, nbr_fea_len) for _ in range(n_graph)]
        )

    def forward(self, atom_fea, nbr_fea, self_fea_idx, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, elem_fea_len)
            Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
            Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
            Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
            Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
            Atom hidden features after convolution

        """
        atom_fea = self.embedding(atom_fea)

        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, self_fea_idx, nbr_fea_idx)

        crys_fea = scatter_mean(atom_fea, crystal_atom_idx, dim=0)  # mean pooling

        # NOTE required to match the reference implementation
        crys_fea = nn.functional.softplus(crys_fea)

        return crys_fea


class SimpleNet(nn.Module):
    """ Simple Feed Forward Neural Network """

    def __init__(
        self, dims: List[int], activation=nn.LeakyReLU, batchnorm: bool = False
    ):
        super().__init__()
        output_dim = dims.pop()

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        if batchnorm:
            self.bns = nn.ModuleList(
                [nn.BatchNorm1d(dims[i + 1]) for i in range(len(dims) - 1)]
            )
        else:
            self.bns = nn.ModuleList([nn.Identity() for i in range(len(dims) - 1)])

        self.acts = nn.ModuleList([activation() for _ in range(len(dims) - 1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, bn, act in zip(self.fcs, self.bns, self.acts):
            x = act(bn(fc(x)))

        return self.fc_out(x)
