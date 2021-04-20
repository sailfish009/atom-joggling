# %%
import pandas as pd
from ml_matrics import precision_recall_curve, roc_curve

from atom_joggling.utils import ROOT


# %%
df = pd.read_csv(
    f"{ROOT}/runs/cgcnn/data_supercon_labeled-5folds-50epochs-128batch/output.csv"
)

df.plot.bar(x="formula", y=["softmax_1", "target"], figsize=[18, 8])


# %%
roc_auc, _ = roc_curve(df.target, df.softmax_1.astype(float))
# plt.savefig(out_dir + "/roc_auc_curve.png", dpi=200, bbox_inches="tight")


# %%
prec, _ = precision_recall_curve(df.target, df.softmax_1.astype(float))
# plt.savefig(out_dir + "/pred_recall_curve.png", dpi=200, bbox_inches="tight")

print(f"mean ROC AUC: {roc_auc:.3g}")
print(f"mean precisions: {prec:.3g}")
