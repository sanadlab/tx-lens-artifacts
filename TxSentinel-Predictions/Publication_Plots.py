import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    confusion_matrix,
)

# --- Plot style ---
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'figure.dpi': 300,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})


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


def plot_comparative_figures():
    os.makedirs('figures', exist_ok=True)

    label_path = 'labeled.csv'
    files = {
        'P1: frozen embeddings': 'prediction_results_based_on_Transformer_Embeddings.csv',
        'P2: fine-tuned embeddings': 'prediction_results_based_on_Finetuned_Transformer.csv',
        'P3: expert features': 'prediction_results_based_on_Expert_Features.csv'
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
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    metrics_summary = []

    print("Processing Experiments...")

    for model in models:
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        fig_pr, ax_pr = plt.subplots(figsize=(8, 6))

        for i, (exp_name, file_path) in enumerate(files.items()):
            df = load_and_merge(file_path, label_path, exp_name)

            y_true = df['label'].astype(int)
            y_score = df[model].astype(float)

            # ROC
            try:
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
            except:
                fpr, tpr, roc_auc, roc_thresholds = np.array([0,1]), np.array([0,1]), float('nan'), np.array([])

            ax_roc.plot(fpr, tpr, color=colors[i], lw=2,
                        label=f'{exp_name} (AUC = {roc_auc:.3f})')

            # PR
            try:
                precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
                pr_auc = average_precision_score(y_true, y_score)
            except:
                precision, recall, pr_thresholds, pr_auc = np.array([1,0]), np.array([0,1]), np.array([]), float('nan')

            ax_pr.plot(recall, precision, color=colors[i], lw=2,
                       label=f'{exp_name} (AP = {pr_auc:.3f})')

            best_thr = fixed_thresholds[exp_name][model]
            preds_binary = (y_score >= best_thr).astype(int)

            acc = accuracy_score(y_true, preds_binary)
            specificity, sensitivity, tn, fp, fn, tp = get_specificity_sensitivity(y_true, preds_binary)

            # --- NEW: Balanced Accuracy ---
            balanced_acc = (specificity + sensitivity) / 2

            fpr_val = fp / (fp + tn)
            fnr_val = fn / (fn + tp)

            metrics_summary.append({
                'Experiment': exp_name,
                'Model': model,
                'Threshold': best_thr,
                'Balanced-Accuracy': balanced_acc,
                'Specificity': specificity,
                'Sensitivity': sensitivity,
                'FPR': fpr_val,
                'FNR': fnr_val,
                'TN': tn,
                'FP': fp,
                'FN': fn,
                'TP': tp,
                'ROC-AUC': roc_auc,
                'PR-AUC': pr_auc
            })

        # ROC plot finalize
        ax_roc.plot([0,1],[0,1],'--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'ROC Curve ({model})')
        ax_roc.legend()
        fig_roc.tight_layout()
        fig_roc.savefig(f'figures/{model}_ROC.png')
        plt.close(fig_roc)

        # PR plot finalize
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title(f'PR Curve ({model})')
        ax_pr.legend()
        fig_pr.tight_layout()
        fig_pr.savefig(f'figures/{model}_PR.png')
        plt.close(fig_pr)

    # Dataframe
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df['Experiment'] = pd.Categorical(metrics_df['Experiment'],
                                              categories=experiment_order,
                                              ordered=True)

    # --- UPDATED metrics list ---
    metrics = [
        'Balanced-Accuracy',
        'Specificity',
        'Sensitivity',
        'FPR',
        'FNR',
        'ROC-AUC',
        'PR-AUC'
    ]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10,6))

        pct_col = metric + '_pct'
        metrics_df[pct_col] = metrics_df[metric] * 100

        sns.barplot(
            data=metrics_df,
            x='Model',
            y=pct_col,
            hue='Experiment',
            ax=ax
        )

        ax.set_ylabel(f"{metric} (%)")

        fig.tight_layout()
        fig.savefig(f'figures/{metric}.png')
        plt.close(fig)

    print(metrics_df)
    metrics_df.to_csv('metrics_df.csv', index=False)


if __name__ == "__main__":
    plot_comparative_figures()