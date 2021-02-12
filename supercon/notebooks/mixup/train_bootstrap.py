# submitted as CSD3 array job with:
# sbatch -J supercon_bootstrap -t 0:10:0 --array 0-85
# --export CMD="python supercon/notebooks/train_bootstrap.py
#   --epoch 10 --out-dir runs/mixup/bootstrap
#   --bootstrap-idx \$SLURM_ARRAY_TASK_ID"
# hpc/gpu_submit

# %%
import os

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from supercon.cgcnn import CGCNN
from supercon.data import ROOT, CrystalGraphData, collate_batch
from supercon.mixup import SemiLoss, args, train, validate
from supercon.utils import save_checkpoint

# %%
df = pd.read_csv(f"{ROOT}/data/supercon/combined.csv").drop(columns=["class"])
labeled_df = df[df.label >= 0].reset_index(drop=True)
unlabeled_df = df[df.label == -1].reset_index(drop=True)

# generate leave-one-out/bootstrap training set
test_sample = labeled_df.T.pop(args.bootstrap_idx)
train_df = labeled_df.drop(args.bootstrap_idx)

out_dir = f"{args.out_dir}/{test_sample.material_id}_left_out"
os.makedirs(out_dir, exist_ok=True)

print(f"- Task: {(task := args.task)}")
print(f"- Using CUDA: {(use_cuda := torch.cuda.is_available())}")
print(f"- Batch size: {args.batch_size:,d}")
print(f"- Train iterations per epoch: {args.train_iterations:,d}")
print(f"- Output directory: {out_dir}")

labeled_set = CrystalGraphData(train_df, task)
unlabeled_set = CrystalGraphData(unlabeled_df, task)

loader_args = {
    "batch_size": args.batch_size,
    "collate_fn": lambda batch: collate_batch(batch, use_cuda),
}

labeled_loader = DataLoader(labeled_set, shuffle=True, **loader_args, drop_last=True)
unlabeled_loader = DataLoader(
    unlabeled_set, shuffle=True, **loader_args, drop_last=True
)


# Model
elem_emb_len = labeled_set.elem_emb_len
nbr_fea_len = labeled_set.nbr_fea_len
model = CGCNN(task, args.robust, elem_emb_len, nbr_fea_len, n_targets=2)
if use_cuda:
    model.cuda()

print(f"- Total params: {model.n_params:,d}")
print(f"- Labeled samples: {len(labeled_set):,d}")
print(f"- Unlabeled samples: {len(unlabeled_set):,d}")

# Train/test losses and optimizer
criterion = SemiLoss(u_ramp_length=3 * args.train_iterations)
val_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
start_epoch = 0

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isfile(args.resume), "Error: checkpoint file not found!"
    out_dir = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    best_acc = checkpoint["best_acc"]
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

writer = SummaryWriter(out_dir)


# %% Train and validate
for epoch in range(start_epoch, args.epochs):

    print(f"\nEpoch: {epoch + 1}/{args.epochs}")

    train_loss, train_loss_x, train_loss_u, u_ramp = train(
        labeled_loader, unlabeled_loader, model, optimizer, criterion
    )
    _, train_acc = validate(labeled_loader, model, val_criterion)

    writer.add_scalar("losses/train_loss", train_loss, epoch)
    writer.add_scalar("losses/train_loss_x", train_loss_x, epoch)
    writer.add_scalar("losses/train_loss_u", train_loss_u, epoch)
    writer.add_scalar("losses/u_ramp", u_ramp, epoch)

    writer.add_scalar("accuracy/train_acc", train_acc, epoch)

    # save model
    is_best = train_acc > best_acc
    best_acc = max(train_acc, best_acc)
    state = {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "acc": train_acc,
        "best_acc": best_acc,
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(state, is_best, out_dir)
writer.close()

print(f"\nBest acc: {best_acc:.3g}")
