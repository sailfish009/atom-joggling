# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mlmatrics import density_scatter, precision_recall_curve, roc_curve
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from supercon.cgcnn import CGCNN, CrystalGraphData, collate_batch
from supercon.utils import ROOT, mean, parser

# %%
args, _ = parser.parse_known_args()

data_df = pd.read_csv(f"{ROOT}/{args.csv_path}").iloc[:, 0:3]

n_splits = 5
kfold = KFold(n_splits, random_state=0, shuffle=True)

untrained_perfs, results = [], []

print(f"\n- Task: {(task := args.task)}")
print(f"- Using CUDA: {(use_cuda := torch.cuda.is_available())}")
print(f"- Batch size: {(batch_size := args.batch_size):,d}")
print(f"- Epochs: {(epochs := args.epochs):,d}")
print(f"- Number of samples: {len(data_df):,d}")
print(f"- Verbose: {(verbose := args.verbose)}")
print(f"- Joggle: {(joggle := args.joggle)}")

csv_name = os.path.splitext(args.csv_path)[0].split(os.sep)[-1]
default_dir = f"{ROOT}/runs/cgcnn/{task}"
default_dir += f"/{csv_name}-{n_splits}folds-{epochs}epochs-{batch_size}batch"
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
for fold, (train_idx, test_idx) in enumerate(kfold.split(data_df), 1):
    print(f"\nFold {fold}/{n_splits}")
    fold_dir = f"{out_dir}/fold_{fold}"
    train_df, test_df = data_df.iloc[train_idx], data_df.iloc[test_idx]

    train_set = CrystalGraphData(train_df, task, joggle=joggle)
    test_set = CrystalGraphData(test_df, task)

    model = CGCNN(
        task,
        args.robust,
        train_set.elem_emb_len,
        train_set.nbr_fea_len,
        n_targets=train_set.n_targets,
        checkpoint_dir=fold_dir,
    )
    if use_cuda:
        model.cuda()

    optimizer = torch.optim.AdamW(model.parameters())

    train_loader = DataLoader(train_set, **loader_args)
    test_loader = DataLoader(test_set, **loader_args)

    writer = SummaryWriter(fold_dir)

    # sanity check: test untrained model performance
    material_ids, formulas, targets, outputs = model.predict(test_loader)

    if task == "classification":
        untrained_acc = (targets == outputs.argmax(dim=1)).float().mean()
        untrained_perfs.append(untrained_acc)
        print(f"untrained accuracy: {untrained_acc:.3f}")
    else:  # regression
        untrained_mae = (targets - outputs).abs().mean()
        untrained_perfs.append(untrained_mae)
        print(f"untrained MAE: {untrained_mae:.3f}")

    model.fit(train_loader, test_loader, optimizer, epochs, writer, verbose=verbose)

    mp_ids, formulas, targets, preds = model.predict(test_loader)

    out_df = pd.DataFrame(  # collect model output into dataframe
        zip([fold] * len(mp_ids), mp_ids, formulas, targets.numpy()),
        columns=["fold", "material_id", "formula", "target"],
    )

    if task == "classification":
        out_df["pred"] = preds.argmax(dim=1)
        for idx, arr in enumerate(preds.softmax(dim=1).numpy().T):
            out_df[f"softmax_{idx}"] = arr

        test_acc = (targets == preds.argmax(dim=1)).float().mean()
        results.append([out_df, test_acc])
        print(f"test accuracy: {test_acc:.3f}")

    else:  # regression
        if model.robust:
            preds, log_std = preds.chunk(2, dim=1)
            out_df["std"] = log_std.exp()
        out_df["pred"] = preds

        test_mae = (targets - preds).abs().mean()
        results.append([out_df, test_mae])
        print(f"test MAE: {test_mae:.3f}")


# %%
untrained_perf_str = f"{mean(untrained_perfs):.3g} +/- {np.std(untrained_perfs):.3g}"
out_dfs, test_accs = zip(*results)

out_df = pd.concat(out_dfs)
out_df.to_csv(f"{out_dir}/output.csv", index=False)

test_perf_str = f"{mean(test_accs):.3g} +/- {np.std(test_accs):.3g}"

print(f"\n\n{n_splits}-fold split results")

if task == "classification":

    if len(out_df) < 100:
        out_df.plot.bar(x="formula", y=["softmax_1", "target"], figsize=[18, 8])
        plt.savefig(out_dir + "/cgcnn_val_preds.png", dpi=200, bbox_inches="tight")
        plt.close()

    roc_auc, _ = roc_curve(out_df.target, out_df.softmax_1)
    plt.savefig(out_dir + "/roc_auc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()
    prec, _ = precision_recall_curve(out_df.target, out_df.softmax_1)
    plt.savefig(out_dir + "/precision_recall_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"mean untrained accuracy: {untrained_perf_str}")
    print(f"mean test accuracy: {test_perf_str}")
    print(f"mean ROC AUC: {roc_auc:.3g}")
    print(f"mean precisions: {prec:.3g}")

else:  # regression
    print(f"mean untrained MAE: {untrained_perf_str}")
    print(f"mean test MAE: {test_perf_str}")

    density_scatter(out_df.target, out_df.pred)
