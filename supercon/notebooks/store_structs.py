# %%
import gzip
import os
import pickle

import pandas as pd
from pymatgen import MPRester
from tqdm import tqdm

from supercon.data import ROOT

# %% Yunwei's hand-crafted superconductivity dataset
df = pd.read_csv(f"{ROOT}/data/supercon/combined.csv", index_col="material_id")


# %%
mp_ids = df.index.tolist()

# Materials Project API keys available at https://materialsproject.org/dashboard.
API_KEY = "X2UaF2zkPMcFhpnMN"

with MPRester(API_KEY) as mp:
    # mp.query performs the actual API call
    structures = mp.query(
        {"material_id": {"$in": mp_ids}}, ["material_id", "structure"]
    )


# %%
for dic in tqdm(structures, desc="Saving structures to disk"):
    mp_id = dic["material_id"]
    struct_path = f"{ROOT}/data/structures/{mp_id}.zip"
    if os.path.exists(struct_path):
        continue

    with gzip.open(struct_path, "wb") as file:
        pickle.dump(dic["structure"], file, protocol=-1)
