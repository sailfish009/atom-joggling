# %%
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
from sklearn.model_selection import train_test_split as split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from supercon.bench import benchmark
from supercon.cgcnn import CGCNN
from supercon.data import ROOT, CrystalGraphData, collate_batch
from supercon.mixup import (
    SemiLoss,
    args,
    mean,
    save_checkpoint,
    train,
    validate,
)

# %%
os.makedirs(args.out_dir, exist_ok=True)

use_cuda = torch.cuda.is_available()
print("use_cuda:", use_cuda)
best_acc = 0  # best test accuracy
torch.manual_seed(0)


# Data
robust = False
task = "classification"

df = pd.read_csv(f"{ROOT}/data/supercon/combined.csv").drop(columns=["class"])
labeled_df = df[df.label >= 0].reset_index(drop=True)
unlabeled_df = df[df.label == -1].reset_index(drop=True)
train_df, test_df = split(labeled_df, test_size=0.2, random_state=0)

labeled_set = CrystalGraphData(train_df, task)
unlabeled_set = CrystalGraphData(unlabeled_df, task)
test_set = CrystalGraphData(test_df, task)

loader_args = {
    "batch_size": args.batch_size,
    "collate_fn": lambda batch: collate_batch(batch, use_cuda),
}

labeled_trainloader = DataLoader(
    labeled_set, shuffle=True, **loader_args, drop_last=True
)
unlabeled_trainloader = DataLoader(
    unlabeled_set, shuffle=True, **loader_args, drop_last=True
)
test_loader = DataLoader(test_set, shuffle=False, **loader_args)

# Model
elem_emb_len = labeled_set.elem_emb_len
nbr_fea_len = labeled_set.nbr_fea_len
model = CGCNN(task, robust, elem_emb_len, nbr_fea_len, n_targets=2)
if use_cuda:
    model.cuda()

print(f"Total params: {model.n_params:,d}")

# Train/test losses and optimizer
train_criterion = SemiLoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
start_epoch = 0

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isfile(args.resume), "Error: checkpoint file not found!"
    args.out_dir = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    best_acc = checkpoint["best_acc"]
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

writer = SummaryWriter(args.out_dir)
test_accs = []


# %% Train and validate
for epoch in range(start_epoch, args.epochs):

    print(f"\nEpoch: {epoch + 1}/{args.epochs}")

    train_loss, train_loss_x, train_loss_u = train(
        labeled_trainloader,
        unlabeled_trainloader,
        model,
        optimizer,
        train_criterion,
        epoch,
    )
    _, train_acc = validate(labeled_trainloader, model, criterion)
    test_loss, test_acc = validate(test_loader, model, criterion)

    writer.add_scalar("losses/train_loss", train_loss, epoch)
    writer.add_scalar("losses/test_loss", test_loss, epoch)

    writer.add_scalar("accuracy/train_acc", train_acc, epoch)
    writer.add_scalar("accuracy/test_acc", test_acc, epoch)

    # save model
    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    save_checkpoint(
        {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "acc": test_acc,
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
        },
        is_best,
    )
    print(
        f"test_acc: {test_acc:<7.3g} train_acc: {train_acc:<7.3g} "
        f"test_loss: {test_loss:<7.3g} train_loss: {train_loss:<7.3g}"
    )
    test_accs.append(test_acc)
writer.close()

print(f"\nBest acc: {best_acc:.3g}")
print(f"Mean acc: {mean(test_accs[-20:]):.3g}")

# %%
benchmark(model, test_loader, args.out_dir)

# %%
