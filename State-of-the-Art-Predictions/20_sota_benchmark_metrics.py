#!/usr/bin/env python3
"""
20_sota_benchmark_metrics.py

Evaluate benchmark binary (hard 0/1) predictions from multiple SOTA columns
against manual labels.

Usage example:
    python3 20_sota_benchmark_metrics.py \
        --labels Manual_Label_Eth_Txs.csv \
        --results sota_labeled_output.csv \
        --outdir sota_benchmark_report \
        --pos-label 1

Features:
- Merges labels & results by `hash` (configurable with --hash-col).
- Auto-detects binary prediction columns (or accept --pred-cols).
- Computes per-column metrics:
    accuracy, precision, recall, sensitivity,
    specificity, balanced accuracy, PR-AUC, f1,
    confusion matrix (tn, fp, fn, tp),
    NPV, FPR, FNR, FDR, FOR,
    support_pos, support_neg
- Writes:
    - merged_labels_results.csv
    - per_prediction_column_metrics.csv
    - sota_benchmark_report.txt (human-readable)
    - diagnostics_predictions_subset.csv

Note:
- PR-AUC is best computed from probability or confidence scores.
- If the prediction column is hard 0/1, PR-AUC will be computed from those
  hard outputs, which is less informative than score-based PR-AUC.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    average_precision_score,
)

# -------------------- Helpers --------------------


def detect_prediction_columns(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """
    Detect columns that look like binary predictions (0/1).
    Criteria:
      - After dropping NA, unique values are subset of {0,1} (numeric or '0'/'1' strings), OR
      - Numeric columns where at least 98% of values are in [0,1] and unique count <= 4.
    exclude: list of column names to ignore (e.g., hash, label).
    """
    if exclude is None:
        exclude = []
    cand = []
    for c in df.columns:
        if c in exclude:
            continue
        ser = df[c].dropna()
        if ser.shape[0] == 0:
            continue

        # Try numeric conversion
        numeric = pd.to_numeric(ser, errors="coerce")
        num_non_na = numeric.dropna().values
        if num_non_na.size > 0:
            uniq = set(np.unique(num_non_na).tolist())

            # treat near-integer 0/1
            if uniq.issubset({0.0, 1.0}):
                cand.append(c)
                continue

            # if most values are in [0,1] and value variety small, accept (robust)
            within01_frac = float(((numeric >= -1e-9) & (numeric <= 1.0 + 1e-9)).sum()) / float(len(numeric))
            uniq_count = len(set([float(x) for x in uniq]))
            if within01_frac >= 0.98 and uniq_count <= 4:
                cand.append(c)
                continue

        # fallback to string '0'/'1' detection
        uniq_str = set([str(x).strip() for x in ser.unique()])
        if uniq_str.issubset({"0", "1"}):
            cand.append(c)
            continue

    return cand


def safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if (d != 0) else float("nan")


def _to_binary_labels(arr: np.ndarray, pos_label: str = "1") -> np.ndarray:
    """
    Convert an array of labels to binary 0/1 using pos_label.
    """
    try:
        numeric = pd.to_numeric(pd.Series(arr), errors="coerce").values
        if np.isnan(numeric).sum() == 0:
            return numeric.astype(int)
    except Exception:
        pass
    pos_str = str(pos_label).strip()
    return np.array([1 if str(x).strip() == pos_str else 0 for x in arr], dtype=int)


def _to_binary_or_score(y_pred_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert raw predictions to:
      - y_pred_binary: binary 0/1 labels
      - y_score: score array for PR-AUC (if raw values are non-binary numeric, use them;
                 otherwise fall back to binary labels)
    """
    try:
        y_pred_numeric = pd.to_numeric(pd.Series(y_pred_raw), errors="coerce").fillna(np.nan).values
    except Exception:
        y_pred_numeric = np.asarray(y_pred_raw)

    # If too many NaNs, fallback to string mapping.
    if np.isnan(y_pred_numeric).sum() > 0.5 * len(y_pred_numeric):
        y_pred_binary = np.array([1 if str(x).strip() == "1" else 0 for x in y_pred_raw], dtype=int)
        y_score = y_pred_binary.astype(float)
        return y_pred_binary, y_score

    y_pred_numeric = np.where(np.isnan(y_pred_numeric), 0.0, y_pred_numeric)
    uniq = set(np.unique(y_pred_numeric).tolist())

    # Binary case
    if uniq.issubset({0.0, 1.0}):
        y_pred_binary = y_pred_numeric.astype(int)
        y_score = y_pred_binary.astype(float)
        return y_pred_binary, y_score

    # Non-binary numeric case: threshold at 0.5 for class labels, use raw values for PR-AUC
    y_pred_binary = (y_pred_numeric >= 0.5).astype(int)
    y_score = y_pred_numeric.astype(float)
    return y_pred_binary, y_score


def compute_binary_metrics_from_hard_preds(y_true: np.ndarray, y_pred_raw: np.ndarray) -> Dict[str, Any]:
    """
    y_true: array-like of 0/1 ints
    y_pred_raw: array-like which may be numeric (0/1) or string '0'/'1' or other numeric values.
    Returns dictionary of metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred, y_score = _to_binary_or_score(y_pred_raw)

    # Basic metrics
    acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else float("nan")
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)  # sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix (tn, fp, fn, tp)
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        else:
            flat = cm.flatten().tolist()
            tn = int(flat[0]) if len(flat) > 0 else 0
            fp = int(flat[1]) if len(flat) > 1 else 0
            fn = int(flat[2]) if len(flat) > 2 else 0
            tp = int(flat[3]) if len(flat) > 3 else 0
    except Exception:
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())

    sensitivity = rec
    specificity = safe_div(tn, tn + fp)
    npv = safe_div(tn, tn + fn)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    fdr = safe_div(fp, fp + tp)
    forate = safe_div(fn, fn + tn)

    # Balanced Accuracy = (Sensitivity + Specificity) / 2
    try:
        if len(np.unique(y_true)) > 1:
            bal_acc = balanced_accuracy_score(y_true, y_pred)
        else:
            bal_acc = float("nan")
    except Exception:
        bal_acc = float("nan")

    # PR-AUC / Average Precision
    # For hard predictions, this is computed from 0/1 outputs.
    try:
        if len(np.unique(y_true)) > 1:
            pr_auc = average_precision_score(y_true, y_score)
        else:
            pr_auc = float("nan")
    except Exception:
        pr_auc = float("nan")

    metrics = dict(
        accuracy=float(acc),
        precision=float(prec),
        recall=float(rec),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        balanced_accuracy=float(bal_acc),
        pr_auc=float(pr_auc),
        f1=float(f1),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        npv=float(npv),
        fpr=float(fpr),
        fnr=float(fnr),
        fdr=float(fdr),
        for_rate=float(forate),
        support_pos=int((y_true == 1).sum()),
        support_neg=int((y_true == 0).sum()),
    )
    return metrics


# -------------------- CLI / Main --------------------


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SOTA binary predictions vs manual labels (hard 0/1).")
    p.add_argument("--labels", required=True, help="CSV with at least columns 'hash' and 'label'")
    p.add_argument("--results", required=True, help="CSV with 'hash' and prediction columns (0/1)")
    p.add_argument("--outdir", required=True, help="Output directory to save report files")
    p.add_argument("--pos-label", default="1", help="Value representing the positive label in labels file (default: 1)")
    p.add_argument("--pred-cols", default=None, help="Comma-separated list of prediction columns in results CSV")
    p.add_argument("--label-col", default="label", help="Column name for labels in labels CSV (default: 'label')")
    p.add_argument("--hash-col", default="hash", help="Column name for hash key (default: 'hash')")
    p.add_argument(
        "--merge-how",
        choices=["inner", "left", "right", "outer"],
        default="inner",
        help="How to merge labels and results on hash (default: inner)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load CSVs
    try:
        df_labels = pd.read_csv(args.labels, dtype={args.hash_col: str})
    except Exception as e:
        print(f"ERROR: cannot read labels CSV '{args.labels}': {e}", file=sys.stderr)
        sys.exit(2)

    try:
        df_results = pd.read_csv(args.results, dtype={args.hash_col: str})
    except Exception as e:
        print(f"ERROR: cannot read results CSV '{args.results}': {e}", file=sys.stderr)
        sys.exit(2)

    # Validate required columns
    if args.hash_col not in df_labels.columns:
        print(f"ERROR: labels file missing hash column '{args.hash_col}'", file=sys.stderr)
        print("Available columns:", df_labels.columns.tolist(), file=sys.stderr)
        sys.exit(2)

    if args.hash_col not in df_results.columns:
        print(f"ERROR: results file missing hash column '{args.hash_col}'", file=sys.stderr)
        print("Available columns:", df_results.columns.tolist(), file=sys.stderr)
        sys.exit(2)

    if args.label_col not in df_labels.columns:
        print(f"ERROR: labels file missing label column '{args.label_col}'", file=sys.stderr)
        print("Available columns:", df_labels.columns.tolist(), file=sys.stderr)
        sys.exit(2)

    # Merge
    merged = pd.merge(df_labels, df_results, on=args.hash_col, how=args.merge_how, suffixes=("_label", "_res"))
    merged_path = os.path.join(args.outdir, "merged_labels_results.csv")
    merged.to_csv(merged_path, index=False)
    print(f"Merged {len(merged)} rows -> {merged_path}")

    # Decide prediction columns
    if args.pred_cols and args.pred_cols.strip():
        pred_cols = [c.strip() for c in args.pred_cols.split(",") if c.strip()]
    else:
        # exclude hash and label columns when auto-detecting
        pred_cols = detect_prediction_columns(merged, exclude=[args.hash_col, args.label_col])

    # Keep only columns that actually exist in merged
    pred_cols = [c for c in pred_cols if c in merged.columns]
    if not pred_cols:
        print("No prediction columns detected. Provide --pred-cols explicitly or check your results CSV.", file=sys.stderr)
        sys.exit(1)

    print("Prediction columns to evaluate:", pred_cols)

    # Prepare true labels
    y_true = _to_binary_labels(merged[args.label_col].values, pos_label=args.pos_label)

    # Per-column metrics
    per_col_metrics = []
    for col in pred_cols:
        preds = merged[col].values
        metrics = compute_binary_metrics_from_hard_preds(y_true, preds)
        entry = {"pred_col": col, "n_examples": int(len(merged))}
        entry.update(metrics)
        per_col_metrics.append(entry)

    # Save metrics CSV
    df_metrics = pd.DataFrame(per_col_metrics)
    metrics_csv = os.path.join(args.outdir, "per_prediction_column_metrics.csv")
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"Per-column metrics written to: {metrics_csv}")

    # Optional simple ranking by F1
    try:
        ranked = df_metrics.sort_values(by=["f1", "balanced_accuracy", "accuracy"], ascending=False)
        ranking_txt = os.path.join(args.outdir, "ranking_by_f1.txt")
        with open(ranking_txt, "w") as rf:
            rf.write("Ranking of prediction columns by F1 (desc):\n")
            for _, row in ranked.iterrows():
                rf.write(
                    f"{row['pred_col']}: "
                    f"F1={row['f1']:.6f}, "
                    f"Acc={row['accuracy']:.6f}, "
                    f"Precision={row['precision']:.6f}, "
                    f"Recall={row['recall']:.6f}, "
                    f"BalAcc={row['balanced_accuracy']:.6f}, "
                    f"PR-AUC={row['pr_auc']:.6f}\n"
                )
        print(f"Ranking by F1 written to: {ranking_txt}")
    except Exception:
        pass

    # Human-readable text report
    report_txt = os.path.join(args.outdir, "sota_benchmark_report.txt")
    with open(report_txt, "w") as f:
        f.write("SOTA Binary Predictions Benchmark Report\n")
        f.write("========================================\n\n")
        f.write(f"Labels file: {args.labels}\n")
        f.write(f"Results file: {args.results}\n")
        f.write(f"Merged rows: {len(merged)}\n")
        f.write(f"Prediction columns evaluated: {pred_cols}\n\n")

        for r in per_col_metrics:
            f.write(f"--- Column: {r['pred_col']} ---\n")
            f.write(f"n_examples: {r['n_examples']}\n")
            f.write(f"Support pos: {r.get('support_pos', 'NA')}, Support neg: {r.get('support_neg', 'NA')}\n")
            f.write(f"Accuracy: {r.get('accuracy'):.6f}\n")
            f.write(f"Precision: {r.get('precision'):.6f}\n")
            f.write(f"Recall: {r.get('recall'):.6f}\n")
            f.write(f"Sensitivity: {r.get('sensitivity'):.6f}\n")
            f.write(f"Specificity (TNR): {r.get('specificity'):.6f}\n")
            f.write(f"Balanced Accuracy: {r.get('balanced_accuracy'):.6f}\n")
            f.write(f"PR-AUC: {r.get('pr_auc'):.6f}\n")
            f.write(f"F1: {r.get('f1'):.6f}\n")
            f.write(f"Confusion (tn, fp, fn, tp): {r.get('tn')}, {r.get('fp')}, {r.get('fn')}, {r.get('tp')}\n")
            f.write(f"NPV: {r.get('npv'):.6f}\n")
            f.write(f"FPR: {r.get('fpr'):.6f}\n")
            f.write(f"FNR: {r.get('fnr'):.6f}\n")
            f.write(f"FDR: {r.get('fdr'):.6f}\n")
            f.write(f"FOR: {r.get('for_rate'):.6f}\n")
            f.write("\n")

    print(f"Text report written to: {report_txt}")

    # Diagnostics subset
    diag_cols = [args.hash_col, args.label_col] + pred_cols
    diag_present = [c for c in diag_cols if c in merged.columns]
    diag_df = merged.loc[:, diag_present]
    diag_path = os.path.join(args.outdir, "diagnostics_predictions_subset.csv")
    diag_df.to_csv(diag_path, index=False)
    print(f"Diagnostics CSV written to: {diag_path}")

    print("Done.")


if __name__ == "__main__":
    main()
