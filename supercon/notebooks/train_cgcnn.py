# %%
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from supercon.bench import benchmark
from supercon.cgcnn import CGCNN
from supercon.data import ROOT, CrystalGraphData, collate_batch
from supercon.utils import mean, parser

# %%
df = pd.read_csv(f"{ROOT}/data/supercon/combined.csv").drop(columns=["class"])
df = df[df.label >= 0].reset_index(drop=True)

n_splits = 5
kfold = KFold(n_splits, random_state=0, shuffle=True)

print(f"dummy accuracy: {df.iloc[:, 2].mean():.3f}")

args, _ = parser.parse_known_args()

untrained_accs, test_accs = [], []
roc_aucs, avg_precs = [], []
batch_size = args.batch_size
epochs = args.epochs
task = args.task

out_dir = (
    args.out_dir
    or f"{ROOT}/runs/cgcnn/{n_splits}folds-{epochs}epochs-{batch_size}bsize"
)

# %%
for fold, (train_idx, test_idx) in enumerate(kfold.split(df), 1):
    print(f"\nFold {fold}/{n_splits}")
    fold_dir = f"{out_dir}/fold_{fold}"
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    train_set = CrystalGraphData(train_df, task)
    test_set = CrystalGraphData(test_df, task, normalizer=train_set.normalizer)

    elem_emb_len = train_set.elem_emb_len
    nbr_fea_len = train_set.nbr_fea_len

    model = CGCNN(
        task,
        args.robust,
        elem_emb_len,
        nbr_fea_len,
        n_targets=train_set.n_targets,
        checkpoint_dir=fold_dir,
    )

    optimizer = torch.optim.AdamW(model.parameters())

    train_loader = DataLoader(
        train_set, collate_fn=collate_batch, batch_size=batch_size
    )
    test_loader = DataLoader(test_set, collate_fn=collate_batch, batch_size=batch_size)

    writer = SummaryWriter(fold_dir)

    # sanity check: test untrained model performance
    material_ids, formulas, targets, outputs = model.predict(test_loader)
    untrained_acc = (targets == outputs.argmax(1)).float().mean()
    untrained_accs.append(untrained_acc)
    print(f"untrained accuracy: {untrained_acc:.3f}")

    model.fit(train_loader, test_loader, optimizer, epochs, writer, verbose=True)

    df, roc_auc, avg_prec = benchmark(model, test_loader, out_dir)
    roc_aucs.append(roc_auc)
    avg_precs.append(avg_prec)
    test_accs.append((df.target == df.pred).mean())


# %%
print(f"\n\n{n_splits}-fold split results")
print(f"mean test accuracy: {mean(test_accs):.3g}")
print(f"mean ROC AUC: {mean(roc_aucs):.3g}")
print(f"mean avg. precisions: {mean(avg_precs):.3g}")
