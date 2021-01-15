from torch import nn

from supercon.base import BaseModel
from supercon.nn import DescriptorNet, SimpleNet


class CGCNN(BaseModel):
    """
    A crystal graph convolutional NN for predicting material properties.

    This model is based on: https://github.com/txie-93/cgcnn (MIT License).
    Changes to the code were made to allow for the removal of zero-padding
    and to benefit from the BaseModel functionality. The architectural
    choices of the model remain unchanged.
    """

    def __init__(
        self,
        task,
        robust,
        elem_emb_len,
        nbr_fea_len,
        n_targets,
        elem_fea_len=64,
        n_graph=4,
        h_fea_len=128,
        n_hidden=1,
        **kwargs,
    ):
        """
        Parameters
        ----------

        elem_fea_len: int
            Number of atom features in the input.
        nbr_fea_len: int
            Number of bond features.
        elem_fea_len: int
            Number of hidden atom features in the convolutional layers
        n_graph: int
            Number of convolutional layers
        h_fea_len: int
            Number of hidden features after pooling
        n_hidden: int
            Number of hidden layers after pooling
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
        crys_fea = self.material_nn(
            atom_fea, nbr_fea, self_fea_idx, nbr_fea_idx, crystal_atom_idx
        )

        # apply neural network to map from learned features to target
        return self.output_nn(crys_fea)
