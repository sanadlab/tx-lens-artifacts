#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
)


# -------------------- Config (fixed) --------------------

LABEL_FILE = "ground-truth-labels.csv"
RESULT_FILE = "sota_results.csv"
OUTPUT_FILE = "Table-3-State-of-the-Art-Predictions.csv"

PRED_COLUMNS = {
    "etherscan_label": "Etherscan",
    "blocksec_label": "BlockSec",
    "certik_label": "CertiK",
}


# -------------------- Helpers --------------------

def safe_div(n, d):
    return float(n) / float(d) if d != 0 else float("nan")


def to_binary_labels(arr):
    numeric = pd.to_numeric(pd.Series(arr), errors="coerce").values
    if np.isnan(numeric).sum() == 0:
        return numeric.astype(int)
    return np.array([1 if str(x).strip() == "1" else 0 for x in arr])


def compute_metrics(y_true, y_pred_raw):
    y_pred = pd.to_numeric(pd.Series(y_pred_raw), errors="coerce").fillna(0).astype(int).values
    y_score = y_pred.astype(float)  # hard predictions (same as original script)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = safe_div(tn, tn + fp)
    sensitivity = safe_div(tp, tp + fn)

    # Same definition as your previous scripts
    balanced_acc = (specificity + sensitivity) / 2

    pr_auc = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")

    return balanced_acc, specificity, sensitivity, pr_auc


# -------------------- Main --------------------

def main():
    print("Loading data...")

    df_labels = pd.read_csv(LABEL_FILE, dtype={"hash": str})
    df_results = pd.read_csv(RESULT_FILE, dtype={"hash": str})

    merged = pd.merge(df_labels, df_results, on="hash", how="inner")
    print(f"Merged {len(merged)} samples")

    y_true = to_binary_labels(merged["label"].values)

    rows = []

    for col, tool_name in PRED_COLUMNS.items():
        if col not in merged.columns:
            print(f"Warning: {col} not found, skipping")
            continue

        print(f"Processing {tool_name}...")

        bal_acc, spec, sens, pr_auc = compute_metrics(y_true, merged[col].values)

        print(
            f"  Bal-Acc={bal_acc*100:.1f}% | "
            f"Spec={spec*100:.1f}% | "
            f"Sens={sens*100:.1f}% | "
            f"PR-AUC={pr_auc*100:.1f}%"
        )

        rows.append({
            "Tool": tool_name,
            "Balanced-Accuracy": round(bal_acc * 100, 1),
            "Specificity": round(spec * 100, 1),
            "Sensitivity": round(sens * 100, 1),
            "PR-AUC": round(pr_auc * 100, 1),
        })

    df_out = pd.DataFrame(rows)

    # Enforce order
    order = ["Etherscan", "BlockSec", "CertiK"]
    df_out["Tool"] = pd.Categorical(df_out["Tool"], categories=order, ordered=True)
    df_out = df_out.sort_values("Tool")

    df_out.to_csv(OUTPUT_FILE, index=False)

    print("\n=== Final Table ===")
    print(df_out)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()