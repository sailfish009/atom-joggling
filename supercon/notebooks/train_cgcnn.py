# %%
import os

import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from supercon.bench import benchmark_classifier
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
print(f"- Epochs: {(epochs := args.epochs):,d}")
print(f"- Number of samples: {len(data_df):,d}")

csv_name = os.path.basename(args.csv_path)
default_dir = (
    f"{ROOT}/runs/cgcnn/{n_splits}folds-{epochs}epochs-{batch_size}batch-{csv_name}"
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

    if task == "classification":
        out_df, roc_auc, avg_prec = benchmark_classifier(model, test_loader, out_dir)

        test_acc = (out_df.target == out_df.pred).mean()
        results.append([out_df, test_acc, roc_auc, avg_prec])
        print(f"test accuracy: {test_acc:.3f}")


# %%
if task == "classification":
    out_dfs, test_accs, roc_aucs, avg_precs = zip(*results)

    print(f"\n\n{n_splits}-fold split results")
    print(f"mean untrained accuracy: {mean(untrained_accs):.3g}")
    print(f"mean test accuracy: {mean(test_accs):.3g}")
    print(f"mean ROC AUC: {mean(roc_aucs):.3g}")
    print(f"mean avg. precisions: {mean(avg_precs):.3g}")

    pd.concat(out_dfs).to_csv(f"{out_dir}/output.csv")
