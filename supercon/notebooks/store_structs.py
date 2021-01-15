# %%
import gzip
import os
import pickle

import pandas as pd
from pymatgen import MPRester

from supercon.data import ROOT

# %%
df = pd.read_csv(f"{ROOT}/data/supercon/combined.csv", index_col="material_id")
mp_ids = df.index.tolist()


# %%
# Materials Project API keys available at https://materialsproject.org/dashboard.
API_KEY = "X2UaF2zkPMcFhpnMN"

with MPRester(API_KEY) as mp:
    # mp.query performs the actual API call
    structures = mp.query({"material_id": {"$in": mp_ids}}, ["structure"])


# %%
for id, struct in zip(mp_ids, structures):
    struct_path = f"{ROOT}/data/structures/{id}.zip"
    if os.path.exists(struct_path):
        continue

    with gzip.open(struct_path, "wb") as file:
        pickle.dump(struct, file, protocol=-1)
