# %%
from datetime import datetime

import pandas as pd
import torch
from sklearn.model_selection import train_test_split as split
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from supercon.bench import benchmark
from supercon.cgcnn import CGCNN
from supercon.data import ROOT, CrystalGraphData, collate_batch

# %%
model_dir = f"{ROOT}/runs/mixmatch"
task = "classification"

df = pd.read_csv(f"{ROOT}/data/supercon/combined.csv").drop(columns=["class"])
labeled_df = df[df.label >= 0].reset_index(drop=True)

print(f"dummy accuracy: {labeled_df.label.mean():.3f}")
train_df, test_df = split(labeled_df, test_size=0.2, random_state=0)

train_set = CrystalGraphData(train_df, task)
test_set = CrystalGraphData(test_df, task)


# %%
batch_size = 128
epochs = 20
verbose = True

robust = False
criterion = NLLLoss() if robust else CrossEntropyLoss()

elem_emb_len = train_set.elem_emb_len
nbr_fea_len = train_set.nbr_fea_len

model = CGCNN(task, robust, elem_emb_len, nbr_fea_len, n_targets=2)

optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[])


# %%
train_loader = DataLoader(train_set, collate_fn=collate_batch, batch_size=batch_size)
test_loader = DataLoader(test_set, collate_fn=collate_batch, batch_size=batch_size)


now = f"{datetime.now():%d-%m-%Y_%H-%M-%S}"
writer = SummaryWriter(f"{model_dir}/runs/{now}")


# %%
model.fit(
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    epochs,
    criterion,
    None,  # normalizer (not needed for classification)
    model_dir,
    verbose,
    writer=writer,
)


# %%
benchmark(model, test_loader, model_dir)
