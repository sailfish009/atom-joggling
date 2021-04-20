# %%
import builtins
import os
import time

import nglview as nv
import pandas as pd
from pymatgen.ext.matproj import MPRester, Structure
from tqdm import tqdm

from atom_joggling.utils import ROOT


# %%
# Yunwei's hand-crafted superconductivity dataset
df_labeled = pd.read_csv(f"{ROOT}/data/atom_joggling/labeled.csv")
df_unlabeled = pd.read_csv(f"{ROOT}/data/atom_joggling/unlabeled.csv")
# Rhys' larger polymorph formation energy dataset
# (originally compiled to compare Wren with CGCNN)
df_mp_p = pd.read_csv(f"{ROOT}/data/e_formation/mp-polymorphs-spglib.csv")
df_regr = pd.read_csv(f"{ROOT}/data/e_formation/cgcnn-regr.csv")
df = pd.concat([df_labeled, df_unlabeled, df_mp_p, df_regr])
df = df.set_index("material_id")
mp_ids = {*df.index}


# %%
# Materials Project API keys available at https://materialsproject.org/dashboard.
with MPRester(api_key="X2UaF2zkPMcFhpnMN") as mp:
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
    struct_path = f"{ROOT}/data/structures/{mp_id}.json"
    if os.path.exists(struct_path):
        continue

    crystal = dic["structure"]
    crystal.to(filename=struct_path)


# %% Check if running python interactively. If so, load a structure from disk and plot it.
if hasattr(builtins, "__IPYTHON__"):
    crystal = Structure.from_file(f"{ROOT}/data/structures/mp-1000.json")

    self_fea_idx, nbr_fea_idx, _, nbr_fea = crystal.get_neighbor_list(r=5)

    crystal.make_supercell([4, 4, 4])
    view = nv.show_pymatgen(crystal)
    view.add_unitcell()
    crystal.get_primitive_structure()
    view


# %%
if hasattr(builtins, "__IPYTHON__"):
    # use JSON rather than CIF as it parses back to Pymatgen structures 4x faster
    start = time.perf_counter()
    for mp_id in ["mp-1008", "mp-7544", "mp-983459"]:
        fsize = os.path.getsize(f"{ROOT}/data/structures/{mp_id}.json")
        print(f"{fsize=}")
        for i in range(1000):
            Structure.from_file(f"{ROOT}/data/structures/{mp_id}.json")

    print(f"took {1000 * (time.perf_counter() - start):.3f} ms")
