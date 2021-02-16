# %%
import os

import nglview as nv
import pandas as pd
from pymatgen import MPRester, Structure
from tqdm import tqdm

from supercon.data import ROOT

# %%
# Yunwei's hand-crafted superconductivity dataset
df = pd.read_csv(f"{ROOT}/data/supercon/labeled.csv")
# Rhys' larger polymorph formation energy dataset
# (originally compiled to compare Wren with CGCNN)
# df = pd.read_csv(f"{ROOT}/data/e_formation/mp-polymorphs-spglib.csv")
# df = pd.read_csv(f"{ROOT}/data/e_formation/cgcnn-regr.csv")
df = df.set_index("material_id")
mp_ids = {*df.index}


# %%
# Materials Project API keys available at https://materialsproject.org/dashboard.
API_KEY = "X2UaF2zkPMcFhpnMN"

with MPRester(API_KEY) as mp:
    # mp.query performs the actual API call
    structures = mp.query(
        {"material_id": {"$in": list(mp_ids)}}, ["material_id", "structure"]
    )

# use set difference operation
ids_missing_structs = mp_ids - {x["material_id"] for x in structures}

print(f"got structures for {len(structures)} out of {len(mp_ids)} MP IDs")
if ids_missing_structs:
    print(f"material IDs with missing structures: {ids_missing_structs}")


# %%
os.makedirs(f"{ROOT}/data/structures", exist_ok=True)

for dic in tqdm(structures, desc="Saving structures to disk"):
    mp_id = dic["material_id"]
    struct_path = f"{ROOT}/data/structures/{mp_id}.cif"
    if os.path.exists(struct_path):
        continue

    crystal = dic["structure"]
    crystal.to(filename=struct_path)


# %% load a structure from disk
if hasattr(__builtins__, "__IPYTHON__"):
    crystal = Structure.from_file(f"{ROOT}/data/structures/mp-1000.cif")

    self_fea_idx, nbr_fea_idx, _, nbr_fea = crystal.get_neighbor_list(r=5)

    crystal.make_supercell([4, 4, 4])
    view = nv.show_pymatgen(crystal)
    view.add_unitcell()
    view
