# %%
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.model_selection import train_test_split as split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from supercon.bench import plot_prec, plot_roc
from supercon.cgcnn import CGCNN
from supercon.data import ROOT, CrystalGraphData, collate_batch
from supercon.utils import parser

# %%
args, _ = parser.parse_known_args()

data_df = pd.read_csv(f"{ROOT}/{args.csv_path}").iloc[:, 0:3]

print(f"\n- Task: {(task := args.task)}")
print(f"- Using CUDA: {(use_cuda := torch.cuda.is_available())}")
print(f"- Batch size: {(batch_size := args.batch_size):,d}")
print(f"- Epochs: {(epochs := args.epochs):,d}")
print(f"- Number of samples: {len(data_df):,d}")
print(f"- Verbose: {(verbose := args.verbose)}")
print(f"- Joggle: {(joggle := args.joggle)}")

csv_name = os.path.splitext(args.csv_path)[0].split(os.sep)[-1]
default_dir = f"{ROOT}/runs/cgcnn/{task}/{csv_name}-{epochs=}-{batch_size=}-{joggle=}"
print(f"- Output directory: {(out_dir := args.out_dir or default_dir)}\n")

loader_args = {
    "batch_size": batch_size,
    "collate_fn": lambda batch: collate_batch(batch, use_cuda),
}

targets = data_df.iloc[:, 2]
if task == "classification":
    print(f"dummy accuracy: {targets.mean():.3f}")
else:  # regression
    print(f"dummy MAE: {(targets - targets.mean()).abs().mean():.3f}")

# %%
train_df, test_df = split(data_df, test_size=0.2, random_state=0)

train_set = CrystalGraphData(train_df, task, joggle=joggle)
test_set = CrystalGraphData(test_df, task)

model = CGCNN(
    task,
    args.robust,
    train_set.elem_emb_len,
    train_set.nbr_fea_len,
    n_targets=train_set.n_targets,
    checkpoint_dir=out_dir,
)

if use_cuda:
    model.cuda()

optimizer = torch.optim.AdamW(model.parameters())

train_loader = DataLoader(train_set, **loader_args)
test_loader = DataLoader(test_set, **loader_args)

writer = SummaryWriter(out_dir)

# sanity check: test untrained model performance
_, _, targets, outputs = model.predict(test_loader, verbose=verbose)

if task == "classification":
    untrained_acc = (targets == outputs.argmax(dim=1)).float().mean()
    print(f"untrained accuracy: {untrained_acc:.3f}")
else:  # regression
    untrained_mae = (targets - outputs).abs().mean()
    print(f"untrained MAE: {untrained_mae:.3f}")

model.fit(train_loader, test_loader, optimizer, epochs, writer, verbose=verbose)

mp_ids, formulas, targets, preds = model.predict(test_loader, verbose=verbose)

out_df = pd.DataFrame(  # collect model output into dataframe
    zip(mp_ids, formulas, targets.numpy()), columns=["material_id", "formula", "target"]
)

if task == "classification":
    out_df["pred"] = preds.argmax(dim=1)
    for idx, arr in enumerate(preds.softmax(dim=1).numpy().T):
        out_df[f"softmax_{idx}"] = arr

    test_acc = (targets == preds.argmax(dim=1)).float().mean()
    print(f"test accuracy: {test_acc:.3f}")

else:  # regression
    if model.robust:
        preds, log_std = preds.chunk(2, dim=1)
        out_df["std"] = log_std.exp()
    out_df["pred"] = preds

    test_mae = (targets - preds).abs().mean()
    print(f"test MAE: {test_mae:.3f}")


# %%
out_df.to_csv(f"{out_dir}/output.csv", index=False)


if task == "classification":

    if len(out_df) < 100:
        out_df.plot.bar(x="formula", y=["softmax_1", "target"], figsize=[18, 8])
        plt.savefig(out_dir + "/cgcnn_val_preds.png", dpi=200, bbox_inches="tight")
        plt.close()

    roc_auc = plot_roc(out_df.target, out_df.softmax_1)
    plt.savefig(out_dir + "/roc_auc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()
    prec = plot_prec(out_df.target, out_df.softmax_1)
    plt.savefig(out_dir + "/precision_recall_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"mean ROC AUC: {roc_auc:.3g}")
    print(f"mean precisions: {prec:.3g}")

else:  # regression
    pass
