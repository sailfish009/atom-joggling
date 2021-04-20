# %%
import gzip
import pickle

import nglview as nv
import numpy as np
import pandas as pd
from pymatgen.core import Structure

from atom_joggling.utils import ROOT


# %%
df = pd.read_csv(f"{ROOT}/data/atom_joggling/combined.csv").drop(columns=["class"])
struct_path = f"{ROOT}/data/structures"


# %%
material_id, formula = df.iloc[0][:2]

# NOTE getting primitive structure before constructing graph
# significantly harms the performance of this model.
with gzip.open(f"{struct_path}/{material_id}.zip", "rb") as file:
    crystal = pickle.loads(file.read())


# %%
# for site in range(crystal.num_sites):
#     perturbation = np.random.normal(0, 0.01, [3])
#     crystal.translate_sites(site, perturbation)

crystal.perturb(0.03)

print(f"crystal: {material_id} ({formula})")
view = nv.show_pymatgen(crystal)
view.add_unitcell()
view


# %%
def perturb_rel(
    structure: Structure, std: float = 0.01, frac_coords: bool = True
) -> Structure:
    """Performs in-place perturbation to all atom position in a Pymatgen crystal structure
    where displacements are sampled from a Gaussian with std measured in percent of
    lattice vectors.

    Args:
        structure (Structure): Pymatgen structure
        std (float, optional): Gaussian standard deviation. Defaults to 0.01.
        frac_coords (bool, optional):  Whether the vector corresponds to fractional or
            cartesian coordinates. Defaults to True.

    Returns:
        Structure: Joggled Pymatgen structure
    """
    for site in range(crystal.num_sites):
        perturbation = np.random.normal(0, std, [3])
        crystal.translate_sites(site, perturbation, frac_coords=frac_coords)

    return structure
