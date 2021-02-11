import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
)


def benchmark_classifier(model, test_loader, out_dir=None):
    mp_ids, formulas, targets, preds = model.predict(test_loader)
    preds = preds.softmax(1).numpy()

    df = pd.DataFrame(
        [mp_ids, formulas, targets.numpy(), *zip(*preds), preds.argmax(-1)],
        index=["material_id", "formula", "target", "softmax_0", "softmax_1", "pred"],
    ).T

    df.plot.bar(x="formula", y=["softmax_1", "target"])
    if out_dir:
        plt.savefig(out_dir + "/cgcnn_val_preds.png", dpi=200)
    plt.show()

    fpr, tpr, _ = roc_curve(targets, preds.argmax(1))
    roc_auc = roc_auc_score(targets, preds.argmax(1))

    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, "b", label=f"AUC = {roc_auc:.2f}")
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.show()

    precision, recall, _ = precision_recall_curve(targets, preds.argmax(1))
    avg_prec = precision_score(targets, preds.argmax(1))

    plt.title("Precision Recall curve for positive label (1: superconductor)")
    plt.plot(precision, recall, "b", label=f"average precision = {avg_prec:.2f}")
    plt.legend(loc="lower left")
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.show()

    return df, roc_auc, avg_prec
