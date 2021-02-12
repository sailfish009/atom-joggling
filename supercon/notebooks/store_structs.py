# %%
import gzip
import os
import pickle

import pandas as pd
from pymatgen import MPRester
from tqdm import tqdm

from supercon.data import ROOT

# %%
# Yunwei's hand-crafted superconductivity dataset
df = pd.read_csv(f"{ROOT}/data/supercon/labeled.csv")
# Rhys' larger polymorph formation energy dataset
# (originally compiled to compare Wren with CGCNN)
# df = pd.read_csv(f"{ROOT}/data/e_formation/mp-polymorphs-spglib.csv")
df = df.set_index("material_id")
mp_ids = df.index.tolist()


# %%
# Materials Project API keys available at https://materialsproject.org/dashboard.
API_KEY = "X2UaF2zkPMcFhpnMN"

with MPRester(API_KEY) as mp:
    # mp.query performs the actual API call
    structures = mp.query(
        {"material_id": {"$in": mp_ids}}, ["material_id", "structure"]
    )

# use set difference operation
ids_missing_structs = {*mp_ids} - {x["material_id"] for x in structures}

print(f"found {len(structures)}/{len(mp_ids)} structures")
print(f"material IDs with missing structures: {ids_missing_structs}")


# %%
os.makedirs(f"{ROOT}/data/structures", exist_ok=True)

for dic in tqdm(structures, desc="Saving structures to disk"):
    mp_id = dic["material_id"]
    struct_path = f"{ROOT}/data/structures/{mp_id}.zip"
    if os.path.exists(struct_path):
        continue

    with gzip.open(struct_path, "wb") as file:
        pickle.dump(dic["structure"], file, protocol=-1)
