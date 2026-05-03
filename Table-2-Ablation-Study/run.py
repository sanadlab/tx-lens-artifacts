import pandas as pd
from sklearn.metrics import confusion_matrix


def load_and_merge(prediction_file, label_file):
    preds = pd.read_csv(prediction_file)

    # Normalize column names to match the model names used below.
    preds = preds.rename(columns={
        'aae': 'AAE',
        'rf': 'RF',
        'et': 'ET',
        'xgb': 'XGB',
        'pu': 'PU'
    })

    labels = pd.read_csv(label_file)
    merged = pd.merge(labels, preds, on='hash', how='inner')
    return merged


def balanced_accuracy(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return (specificity + sensitivity) / 2.0


def main():
    label_file = 'ground-truth-labels.csv'
    with_file = 'P1_With_TxSummarization.csv'
    without_file = 'P1_Without_TxSummarization.csv'

    # Fixed thresholds for P1 with transaction summarization
    thresholds_with = {
        'AAE': 0.999999642372131,
        'RF': 0.0016576434940089,
        'ET': 0.0143357727971129,
        'XGB': 0.0001840714830905,
        'PU': 0.0009032187159518,
    }

    # Table display names
    model_display = {
        'AAE': 'AAE',
        'RF': 'RF',
        'ET': 'ETs',
        'XGB': 'XGB',
        'PU': 'nnPU',
    }

    models = ['AAE', 'RF', 'ET', 'XGB', 'PU']

    df_with = load_and_merge(with_file, label_file)
    df_without = load_and_merge(without_file, label_file)

    rows = []

    for model in models:
        y_true_with = df_with['label'].astype(int)
        y_score_with = df_with[model].astype(float)
        y_pred_with = (y_score_with >= thresholds_with[model]).astype(int)
        ba_with = balanced_accuracy(y_true_with, y_pred_with) * 100.0

        y_true_without = df_without['label'].astype(int)
        y_score_without = df_without[model].astype(float)
        y_pred_without = (y_score_without >= 0.5).astype(int)
        ba_without = balanced_accuracy(y_true_without, y_pred_without) * 100.0

        degradation = round(ba_with, 1) - round(ba_without, 1)

        rows.append({
            'Model': model_display[model],
            'P1: With Tx Summarization': round(ba_with, 1),
            'P1: Without Tx Summarization': round(ba_without, 1),
            'Degradation': round(degradation, 1),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv('Table-2-Ablation-Study.csv', index=False, float_format='%.1f')
    print(out_df)


if __name__ == "__main__":
    main()