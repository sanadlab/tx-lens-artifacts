import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    confusion_matrix,
)


def load_and_merge(prediction_file, label_file, experiment_name):
    preds = pd.read_csv(prediction_file)

    preds.rename(columns={
        'ensemble_score': 'ens',
        'aae': 'AAE',
        'rf': 'RF',
        'et': 'ET',
        'xgb': 'XGB',
        'pu': 'PU'
    }, inplace=True)

    labels = pd.read_csv(label_file)

    merged = pd.merge(labels, preds, on='hash', how='inner')
    merged['experiment'] = experiment_name
    return merged


def get_specificity_sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return specificity, sensitivity, tn, fp, fn, tp


def generate_metrics_csv():
    label_path = 'ground-truth-labels.csv'
    files = {
        'P1: frozen embeddings': 'P1_prediction_results.csv',
        'P2: fine-tuned embeddings': 'P2_prediction_results.csv',
        'P3: expert features': 'P3_prediction_results.csv'
    }

    fixed_thresholds = {
        'P1: frozen embeddings': {
            'AAE': 0.999999642372131,
            'RF': 0.0016576434940089,
            'ET': 0.0143357727971129,
            'XGB': 0.0001840714830905,
            'PU': 0.0009032187159518,
        },
        'P2: fine-tuned embeddings': {
            'AAE': 1.0,
            'RF': 0.0033320957005065,
            'ET': 0.00999800039992,
            'XGB': 8.11014069768135e-06,
            'PU': 0.833444966954723,
        },
        'P3: expert features': {
            'AAE': 1.0,
            'RF': 0.0966666666666667,
            'ET': 0.259976015988489,
            'XGB': 0.0085241897031664,
            'PU': 0.0118313921445498,
        }
    }

    experiment_order = list(files.keys())
    models = ['AAE', 'RF', 'ET', 'XGB', 'PU']

    metrics_summary = []

    print("Starting evaluation...")

    for exp_name, file_path in files.items():
        print(f"\n=== Processing {exp_name} ({file_path}) ===")

        df = load_and_merge(file_path, label_path, exp_name)
        print(f"Loaded {len(df)} samples")

        y_true = df['label'].astype(int)

        for model in models:
            print(f"  -> Model: {model}")

            y_score = df[model].astype(float)
            best_thr = fixed_thresholds[exp_name][model]
            preds_binary = (y_score >= best_thr).astype(int)

            acc = accuracy_score(y_true, preds_binary)
            specificity, sensitivity, tn, fp, fn, tp = get_specificity_sensitivity(y_true, preds_binary)
            balanced_acc = (specificity + sensitivity) / 2
            pr_auc = average_precision_score(y_true, y_score)

            print(
                f"     "
                f"Bal-Acc={balanced_acc * 100:.1f}% | "
                f"Spec={specificity * 100:.1f}% | "
                f"Sens={sensitivity * 100:.1f}% | "
                f"PR-AUC={pr_auc * 100:.1f}%"
            )

            metrics_summary.append({
                'Experiment': exp_name,
                'Model': model,
                'Accuracy': balanced_acc * 100,
                'Specificity': specificity * 100,
                'Sensitivity': sensitivity * 100,
                'PR-AUC': pr_auc * 100,
                
            })

    metrics_df = pd.DataFrame(metrics_summary)

    metrics_df['Experiment'] = pd.Categorical(
        metrics_df['Experiment'],
        categories=experiment_order,
        ordered=True
    )
    metrics_df['Model'] = pd.Categorical(
        metrics_df['Model'],
        categories=models,
        ordered=True
    )

    metrics_df = metrics_df.sort_values(['Experiment', 'Model']).reset_index(drop=True)

    print("\n=== Final Results ===")
    print(metrics_df)

    # Export only the requested columns, as percentages
    output_df = metrics_df[['Experiment', 'Model', 'Accuracy', 'Specificity', 'Sensitivity', 'PR-AUC']].copy()

    # Keep one decimal place in the CSV output
    output_file = 'Table-1-TxSentinel-Predictions.csv'
    output_df.to_csv(output_file, index=False, float_format='%.1f')

    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    generate_metrics_csv()