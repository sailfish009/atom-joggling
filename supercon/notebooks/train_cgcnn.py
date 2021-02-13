# %%
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from supercon.bench import plot_prec, plot_roc
from supercon.cgcnn import CGCNN
from supercon.data import ROOT, CrystalGraphData, collate_batch
from supercon.utils import mean, parser

# %%
args, _ = parser.parse_known_args()

data_df = pd.read_csv(f"{ROOT}/{args.csv_path}").iloc[:, 0:3]

n_splits = 5
kfold = KFold(n_splits, random_state=0, shuffle=True)

untrained_accs, results = [], []

verbose = args.verbose
print(f"\n- Task: {(task := args.task)}")
print(f"- Using CUDA: {(use_cuda := torch.cuda.is_available())}")
print(f"- Batch size: {(batch_size := args.batch_size):,d}")
print(f"- Epochs: {(epochs := 1):,d}")
print(f"- Number of samples: {len(data_df):,d}")

csv_name = "_".join(os.path.splitext(args.csv_path)[0].split(os.sep))
default_dir = (
    f"{ROOT}/runs/cgcnn/{csv_name}-{n_splits}folds-{epochs}epochs-{batch_size}batch"
)
print(f"- Output directory: {(out_dir := args.out_dir or default_dir)}")

loader_args = {
    "batch_size": batch_size,
    "collate_fn": lambda batch: collate_batch(batch, use_cuda),
}

targets = data_df.iloc[:, 2]
if task == "classification":
    print(f"\ndummy accuracy: {targets.mean():.3f}")
else:  # regression
    print(f"dummy MAE: {(targets - targets.mean()).abs().mean():.3f}")

# %%
for fold, (train_idx, test_idx) in enumerate(kfold.split(data_df), 1):
    print(f"\nFold {fold}/{n_splits}")
    fold_dir = f"{out_dir}/fold_{fold}"
    train_df, test_df = data_df.iloc[train_idx], data_df.iloc[test_idx]

    train_set = CrystalGraphData(train_df, task)
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
        untrained_acc = (targets == outputs.argmax(1)).float().mean()
        untrained_accs.append(untrained_acc)
        print(f"untrained accuracy: {untrained_acc:.3f}")
    else:  # regression
        print(f"untrained MAE: {(targets - outputs).abs().mean():.3f}")

    model.fit(train_loader, test_loader, optimizer, epochs, writer, verbose=verbose)

    mp_ids, formulas, targets, preds = model.predict(test_loader)

    if task == "classification":
        preds = preds.softmax(1).numpy()
        softmax_cols = [f"softmax_{idx}" for idx in range(preds.shape[1])]

        out_df = pd.DataFrame(
            [mp_ids, formulas, targets.numpy(), *zip(*preds), preds.argmax(-1)],
            index=["material_id", "formula", "target", *softmax_cols, "pred"],
        ).T
        # convert non-string cols from dtype object to int/float
        out_df[out_df.columns[2:]] = out_df[out_df.columns[2:]].apply(pd.to_numeric)

        test_acc = (out_df.target == out_df.pred).mean()
        results.append([out_df, test_acc])
        print(f"test accuracy: {test_acc:.3f}")


# %%
if task == "classification":
    out_dfs, test_accs = zip(*results)

    df = pd.concat(out_dfs)
    df.to_csv(f"{out_dir}/output.csv")
    if len(df) < 100:
        df.plot.bar(x="formula", y=["softmax_1", "target"], figsize=[18, 8])
        plt.savefig(out_dir + "/cgcnn_val_preds.png", dpi=200, bbox_inches="tight")

    roc_auc = plot_roc(df.target, df.softmax_1.astype(float))
    plt.savefig(out_dir + "/roc_auc_curve.png", dpi=200, bbox_inches="tight")
    prec = plot_prec(df.target, df.softmax_1.astype(float))
    plt.savefig(out_dir + "/pred_recall_curve.png", dpi=200, bbox_inches="tight")

    print(f"\n\n{n_splits}-fold split results")
    print(f"mean untrained accuracy: {mean(untrained_accs):.3g}")
    print(f"mean test accuracy: {mean(test_accs):.3g}")
    print(f"mean ROC AUC: {roc_auc:.3g}")
    print(f"mean precisions: {prec:.3g}")
