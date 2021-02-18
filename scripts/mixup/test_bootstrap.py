# %%
import os

import pandas as pd
import torch

from supercon.cgcnn import CGCNN
from supercon.data import ROOT, CrystalGraphData, collate_batch
from supercon.mixup import args

# %%
task = "classification"

df = pd.read_csv(f"{ROOT}/data/supercon/combined.csv").drop(columns=["class"])
labeled_df = df[df.label >= 0].reset_index(drop=True)

labeled_set = CrystalGraphData(labeled_df, task)

# Model
elem_emb_len = labeled_set.elem_emb_len
nbr_fea_len = labeled_set.nbr_fea_len
model = CGCNN(task, args.robust, elem_emb_len, nbr_fea_len, n_targets=2)

id_to_idx_map = {val: key for key, val in labeled_df.material_id.to_dict().items()}

# %%
results = []

for file in os.listdir("runs/mixup/bootstrap"):
    mp_id = file.replace("_left_out", "")
    file = f"runs/mixup/bootstrap/{file}/best_model"

    checkpoint = torch.load(file, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()  # disables batch norm

    sample_idx = id_to_idx_map[mp_id]
    inputs, [target], *_ = collate_batch([labeled_set[sample_idx]])

    with torch.no_grad():
        [pred] = model(*inputs)

    pred = pred.softmax(0).numpy()
    results.append([mp_id, target.item(), pred, pred.argmax()])


# %%
df = pd.DataFrame(results, columns=["material_id", "target", "softmax", "pred"])
print(df)


print(f"acc: {(df.target == df.pred).mean():.3f}")
