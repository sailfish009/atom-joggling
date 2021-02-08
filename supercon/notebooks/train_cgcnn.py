# %%
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from supercon.bench import benchmark
from supercon.cgcnn import CGCNN
from supercon.data import ROOT, CrystalGraphData, collate_batch
from supercon.utils import mean, parser

# %%
args, _ = parser.parse_known_args()
task = "classification"

df = pd.read_csv(f"{ROOT}/data/supercon/combined.csv").drop(columns=["class"])
labeled_df = df[df.label >= 0].reset_index(drop=True)
n_splits = 5
kfold = KFold(n_splits, random_state=0, shuffle=True)

print(f"dummy accuracy: {labeled_df.label.mean():.3f}")


untrained_accs, test_accs = [], []
roc_aucs, avg_precs = [], []
batch_size = 32
epochs = 100

criterion = NLLLoss() if args.robust else CrossEntropyLoss()

out_dir = (
    args.out_dir
    or f"{ROOT}/runs/cgcnn/{n_splits}folds-{epochs}epochs-{batch_size}bsize"
)

for fold, (train_idx, test_idx) in enumerate(kfold.split(labeled_df), 1):
    print(f"\nFold {fold}/{n_splits}")

    train_df, test_df = labeled_df.iloc[train_idx], labeled_df.iloc[test_idx]

    train_set = CrystalGraphData(train_df, task)
    test_set = CrystalGraphData(test_df, task)

    # %%
    elem_emb_len = train_set.elem_emb_len
    nbr_fea_len = train_set.nbr_fea_len

    model = CGCNN(
        task,
        args.robust,
        elem_emb_len,
        nbr_fea_len,
        n_targets=2,
        checkpoint_dir=out_dir,
    )

    optimizer = torch.optim.AdamW(model.parameters())

    # %%
    train_loader = DataLoader(
        train_set, collate_fn=collate_batch, batch_size=batch_size
    )
    test_loader = DataLoader(test_set, collate_fn=collate_batch, batch_size=batch_size)

    writer = SummaryWriter(f"{out_dir}/fold{fold}")

    # %% sanity check: test untrained model performance
    material_ids, formulas, targets, outputs = model.predict(test_loader)
    untrained_acc = (targets == outputs.argmax(1)).float().mean()
    untrained_accs.append(untrained_acc)
    print(f"untrained accuracy: {untrained_acc:.3f}")

    # %%
    model.fit(
        train_loader, test_loader, optimizer, epochs, criterion, writer, verbose=False
    )

    # %%
    df, roc_auc, avg_prec = benchmark(model, test_loader, out_dir)
    roc_aucs.append(roc_auc)
    avg_precs.append(avg_prec)
    test_accs.append((df.target == df.pred).mean())


# %%
print(f"\n\n{n_splits}-fold split results")
print(f"mean test accuracy: {mean(test_accs):.3g}")
print(f"mean ROC AUC: {mean(roc_aucs):.3g}")
print(f"mean avg. precisions: {mean(avg_precs):.3g}")
