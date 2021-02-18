from typing import List

import torch
from torch import Tensor, nn
from torch.nn.functional import softplus
from torch_scatter import scatter_add
from torch_scatter.scatter import scatter_mean

from supercon.base import BaseModel


class CGCNN(BaseModel):
    """
    A crystal graph convolutional NN for predicting material properties.

    This model is based on: https://github.com/txie-93/cgcnn (MIT License).
    Changes to the code were made to allow for the removal of zero-padding
    and to inherit from the BaseModel class. The architectural
    choices of the model remain largely unchanged.
    """

    def __init__(
        self,
        task: str,
        robust: bool,
        elem_emb_len: int,
        nbr_fea_len: int,
        n_targets: int,
        elem_fea_len: int = 64,
        n_graph: int = 4,
        h_fea_len: int = 128,
        n_hidden: int = 1,
        **kwargs,
    ) -> None:
        """
        Args:
            task (str): regression or classification
            robust (bool): whether to use a robust loss function with self-attenuation
            elem_emb_len (int): Length of embedding vectors used to describe elements.
                Projected onto elem_fea_len dimensions by a linear layer of the model.
            nbr_fea_len (int): length of the feature vector used to describe vertices
            n_targets (int): number of regression or classification targets
            elem_fea_len (int, optional): Number of hidden atom features in the
                convolutional layers. Defaults to 64.
            n_graph (int, optional): Number of convolutional layers. Defaults to 4.
            h_fea_len (int, optional): Number of hidden features after pooling.
                Defaults to 128.
            n_hidden (int, optional): Number of hidden layers after pooling.
                Defaults to 1.
        """
        super().__init__(task=task, robust=robust, **kwargs)

        desc_dict = {
            "elem_emb_len": elem_emb_len,
            "nbr_fea_len": nbr_fea_len,
            "elem_fea_len": elem_fea_len,
            "n_graph": n_graph,
        }

        self.material_nn = DescriptorNet(**desc_dict)

        model_params = {
            "task": task,
            "robust": robust,
            "n_targets": n_targets,
            "h_fea_len": h_fea_len,
            "n_hidden": n_hidden,
        }
        self.model_params.update({**model_params, **desc_dict})

        out_hidden = [h_fea_len] * n_hidden

        output_dim = 2 * n_targets if robust else n_targets

        # NOTE the original model used softpluses as activation functions
        self.output_nn = SimpleNet([elem_fea_len, *out_hidden, output_dim], nn.Softplus)

    def forward(
        self, atom_fea, nbr_fea, atom_indices, neighbor_indices, crystal_atom_idx
    ) -> Tensor:
        """Forward pass through CGCNN.

        Args (see collate_batch):
            atom_fea (Tensor of shape [N, elem_fea_len]):
                Atom features from element type.
            nbr_fea (Tensor of shape [N, M, nbr_fea_len]):
                Bond features of each atom's M neighbors.
            atom_indices (LongTensor of shape [N, M]):
                Indices enumerating each atom in the batch.
            neighbor_indices (Tensor of shape [N, M]):
                Indices of M neighbors of each atom.
            crystal_atom_idx (Tensor): list of LongTensors of length N_0
                mapping from crystal index to atom index.

        Returns:
            Tensor of shape [N]: Atomic Atom hidden features after convolution
        """
        crys_fea = self.material_nn(
            atom_fea, nbr_fea, atom_indices, neighbor_indices, crystal_atom_idx
        )

        # apply neural network to map from learned features to target
        return self.output_nn(crys_fea)


class ConvLayer(nn.Module):
    """ Performs a 'convolution' on crystal graphs """

    def __init__(self, elem_fea_len: int, nbr_fea_len: int) -> None:
        """
        Args:
            elem_fea_len (int): Number of atom hidden features
            nbr_fea_len (int): Number of bond features
        """
        super().__init__()
        self.elem_fea_len = elem_fea_len
        self.nbr_fea_len = nbr_fea_len

        self.fc_full = nn.Linear(2 * elem_fea_len + nbr_fea_len, 2 * elem_fea_len)
        self.bn1 = nn.BatchNorm1d(2 * elem_fea_len)
        self.bn2 = nn.BatchNorm1d(elem_fea_len)

    def forward(self, atom_in_fea, nbr_fea, atom_indices, neighbor_indices) -> Tensor:
        """Forward pass through the ConvLayer.

        Args (see collate_batch):
            atom_in_fea (Tensor of shape [N, elem_fea_len]):
                Atomic features before convolution.
            nbr_fea (Tensor of shape [N, M, nbr_fea_len]):
                Bond features of each atom's M neighbors.
            atom_indices (LongTensor of shape [N, M]):
                Indices enumerating each atom in the batch.
            neighbor_indices (LongTensor of shape [N, M]):
                Indices of the M neighbors of each atom.

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Returns:
            Tensor of shape [N, elem_fea_len]: Atomic features after convolution.
        """
        # convolution
        atom_nbr_fea = atom_in_fea[neighbor_indices, :]
        atom_self_fea = atom_in_fea[atom_indices, :]

        total_fea = torch.cat([atom_self_fea, atom_nbr_fea, nbr_fea], dim=1)

        total_fea = self.fc_full(total_fea)
        total_fea = self.bn1(total_fea)

        filter_fea, core_fea = total_fea.chunk(2, dim=1)
        filter_fea = torch.sigmoid(filter_fea)
        core_fea = softplus(core_fea)

        # take the element-wise product of the filter and core
        nbr_msg = filter_fea * core_fea
        nbr_summed = scatter_add(nbr_msg, atom_indices, dim=0)  # sum pooling

        nbr_summed = self.bn2(nbr_summed)
        out = softplus(atom_in_fea + nbr_summed)

        return out


class DescriptorNet(nn.Module):
    """ DescriptorNet is the message passing section of CGCNN. """

    def __init__(
        self,
        elem_emb_len: int,
        nbr_fea_len: int,
        elem_fea_len: int = 64,
        n_graph: int = 4,
    ) -> None:
        super().__init__()

        self.embedding = nn.Linear(elem_emb_len, elem_fea_len)

        self.convs = nn.ModuleList(
            ConvLayer(elem_fea_len, nbr_fea_len) for _ in range(n_graph)
        )

    def forward(
        self, atom_fea, nbr_fea, atom_indices, neighbor_indices, crystal_atom_idx
    ) -> Tensor:
        """Forward pass through the DescriptorNet.

        Args (see collate_batch):
            atom_fea (Tensor of shape [N, elem_fea_len]):
                Atom features from element type.
            nbr_fea (Tensor of shape [N, M, nbr_fea_len]):
                Bond features of each atom's M neighbors.
            atom_indices (LongTensor of shape [N, M]):
                Indices enumerating each atom in the batch.
            neighbor_indices (LongTensor of shape [N, M]):
                Indices of the M neighbors of each atom.
            crystal_atom_idx (LongTensor of shape [N_0]):
                Mapping from crystal indices to atom indices.

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N_0: Total number of crystals in the batch

        Returns:
            Tensor of shape [N]: crystal features predicted after n_graph conv operations
        """
        atom_fea = self.embedding(atom_fea)

        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, atom_indices, neighbor_indices)

        crys_fea = scatter_mean(atom_fea, crystal_atom_idx, dim=0)  # mean pooling

        # NOTE required to match the reference implementation
        # SoftPlus is a smooth ReLU approximation used to ensure positive output.
        crys_fea = softplus(crys_fea)

        return crys_fea


class SimpleNet(nn.Module):
    """ Simple fully-connected neural net """

    def __init__(self, dims: List[int], activation=nn.LeakyReLU) -> None:
        super().__init__()
        out_dim = dims.pop()

        self.layers = nn.ModuleList(nn.Linear(d1, d2) for d1, d2 in zip(dims, dims[1:]))

        self.acts = nn.ModuleList(activation() for _ in dims[1:])

        self.linear_out = nn.Linear(dims[-1], out_dim)

    def forward(self, x: Tensor) -> Tensor:
        for linear, act in zip(self.layers, self.acts):
            x = act(linear(x))

        return self.linear_out(x).squeeze()
