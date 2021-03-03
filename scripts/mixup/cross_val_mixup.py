# %%
import os

import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from supercon.cgcnn import CGCNN, CrystalGraphData, collate_batch
from supercon.mixup import SemiLoss, args, train_with_mixup, validate_mixup
from supercon.utils import ROOT, mean, save_checkpoint

# %%
best_accs, mean_accs = [], []

use_cuda = torch.cuda.is_available()

# Data
task = "classification"

df = pd.read_csv(f"{ROOT}/data/supercon/combined.csv").drop(columns=["class"])
labeled_df = df[df.label >= 0].reset_index(drop=True)
unlabeled_df = df[df.label == -1].reset_index(drop=True)
n_splits = 5
kfold = KFold(n_splits, random_state=0, shuffle=True)

unlabeled_set = CrystalGraphData(unlabeled_df, task)

elem_emb_len = unlabeled_set.elem_emb_len
nbr_fea_len = unlabeled_set.nbr_fea_len
model = CGCNN(task, args.robust, elem_emb_len, nbr_fea_len, n_targets=2)

print(f"- Task: {task}")
print(f"- Using CUDA: {use_cuda}")
print(f"- Batch size: {args.batch_size:,d}")
print(f"- Train iterations per epoch: {args.train_iterations:,d}")
print(f"- Output directory: {(out_dir := args.out_dir)}")
print(f"- Unlabeled samples: {len(unlabeled_set):,d}")
print(f"- Model params: {model.n_params:,d}")

os.makedirs(out_dir, exist_ok=True)

for fold, (train_idx, test_idx) in enumerate(kfold.split(labeled_df), 1):
    print(f"\nFold {fold}/{n_splits}")

    train_df, test_df = labeled_df.iloc[train_idx], labeled_df.iloc[test_idx]

    labeled_set = CrystalGraphData(train_df, task)
    test_set = CrystalGraphData(test_df, task)

    loader_args = {
        "batch_size": args.batch_size,
        "collate_fn": lambda batch: collate_batch(batch, use_cuda),
    }

    labeled_loader = DataLoader(
        labeled_set, shuffle=True, **loader_args, drop_last=True
    )
    unlabeled_loader = DataLoader(
        unlabeled_set, shuffle=True, **loader_args, drop_last=True
    )
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    # Model
    elem_emb_len = labeled_set.elem_emb_len
    nbr_fea_len = labeled_set.nbr_fea_len
    model = CGCNN(task, args.robust, elem_emb_len, nbr_fea_len, n_targets=2)

    if use_cuda:
        model.cuda()

    print(f"- Labeled samples: {len(labeled_set):,d}")
    print(f"- Test samples: {len(test_set):,d}")

    # Train/test losses and optimizer
    train_criterion = SemiLoss(u_ramp_length=3 * args.train_iterations)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    best_acc = 0  # best test accuracy

    if args.resume:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        assert os.path.isfile(args.resume), "Error: checkpoint file not found!"
        out_dir = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume, map_location="cuda" if use_cuda else "cpu")
        best_acc = checkpoint["best_acc"]
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    writer = SummaryWriter(out_dir)
    test_accs = []

    # %% Train and validate
    for epoch in range(start_epoch, args.epochs):

        print(f"\nEpoch: {epoch + 1}/{args.epochs}")

        train_loss, train_loss_x, train_loss_u, u_ramp = train_with_mixup(
            labeled_loader, unlabeled_loader, model, optimizer, train_criterion
        )
        _, train_acc = validate_mixup(labeled_loader, model, criterion)
        test_loss, test_acc = validate_mixup(test_loader, model, criterion)

        writer.add_scalar("losses/train_loss", train_loss, epoch)
        writer.add_scalar("losses/train_loss_x", train_loss_x, epoch)
        writer.add_scalar("losses/train_loss_u", train_loss_u, epoch)
        writer.add_scalar("losses/test_loss", test_loss, epoch)
        writer.add_scalar("losses/u_ramp", u_ramp, epoch)

        writer.add_scalar("accuracy/train_acc", train_acc, epoch)
        writer.add_scalar("accuracy/test_acc", test_acc, epoch)

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        state = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "acc": test_acc,
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(state, is_best, out_dir)
        print(
            f"test_acc: {test_acc:<7.3g} train_acc: {train_acc:<7.3g} "
            f"test_loss: {test_loss:<7.3g} train_loss: {train_loss:<7.3g}"
        )
        test_accs.append(test_acc)
    writer.close()

    print(f"\nBest acc: {best_acc:.3g}")
    print(f"Mean acc: {mean(test_accs[-20:]):.3g}")
    mean_accs.append(mean(test_accs[-20:]))
    best_accs.append(best_acc)

# %%
print(f"{n_splits}-fold split results")
print(f"mean accuracy: {mean(mean_accs):.3g}")
print(f"mean best accuracies: {mean(best_accs):.3g}")
