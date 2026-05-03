# TxSentinel-artifacts

Artifacts for our CCS paper on Ethereum transaction anomaly detection and benchmarking.

## Overview

This repository substantiates our claims regarding model efficiency, real-time latency, and baseline detection performance. To support reproducibility while minimizing potential ecosystem risks, we release a curated set of artifacts that enable verification of our results without exposing sensitive components of the system.

## Released Artifacts

### 1. Evaluation Dataset

We provide a curated benchmark dataset comprising **439 Ethereum transaction traces**, including both malicious and benign samples, collected between **2023 and 2025**.

* The dataset is derived entirely from **public on-chain data**.
* Researchers can reconstruct execution traces using any standard Ethereum archive node.
* The dataset is publicly available via Zenodo for long-term access and reproducibility.

---

### 2. Table-1: TxSentinel Predictions

This component reproduces the results of TxSentinel across three experimental settings using precomputed model predictions.

**Contents**

The evaluation uses the following fixed files:

* `ground-truth-labels.csv` — manually curated labels
* `P1-prediction_results.csv` — frozen embeddings
* `P2-prediction_results.csv` — fine-tuned embeddings
* `P3-prediction_results.csv` — expert features

Each file contains prediction scores for the following models:

* AAE, RF, ET, XGB, PU

**Execution**

To reproduce Table 1:

```bash
python run.py
```

**Output**

The script generates:

* `Table-1-TxSentinel-Predictions.csv`

**Reported Metrics**

For each `(Experiment, Model)` pair, the following metrics are computed:

* Accuracy
* Specificity
* Sensitivity
* PR-AUC

All metrics are reported as percentages with one decimal place.

**Reproducibility Notes**

* Fixed decision thresholds are used for each model and experiment.
* Metrics are computed deterministically from the provided predictions.
* No training or randomness is involved.
* The output CSV matches the results reported in the paper.

---

### 3. Table-3: State-of-the-Art Predictions

This component evaluates external state-of-the-art tools on the same benchmark dataset using their binary predictions.

**Contents**

The evaluation uses the following fixed files:

* `ground-truth-labels.csv` — manually curated labels
* `sota_labeled_output.csv` — tool predictions

The following prediction columns are evaluated:

* `etherscan_label` → Etherscan
* `blocksec_label` → BlockSec
* `certik_label` → CertiK

**Execution**

To reproduce Table 3:

```bash
python run.py
```

**Output**

The script generates:

* `Table-3-State-of-the-Art-Predictions.csv`

**Reported Metrics**

For each tool, the following metrics are computed:

* Balanced-Accuracy
* Specificity
* Sensitivity
* PR-AUC

All metrics are reported as percentages with one decimal place.

**Reproducibility Notes**

* Predictions are treated as hard binary labels (0/1).
* Balanced Accuracy is computed as the average of specificity and sensitivity.
* PR-AUC is computed directly from the binary outputs, consistent with the original script.
* No parameter tuning or thresholding is applied.
* Results are fully deterministic and reproducible with a single command.

---

### 4. Reproducibility Scripts

We provide minimal scripts required to:

* Reproduce the empirical results reported in the paper.
* Evaluate performance metrics using the released predictions.

### 5. Controlled Release Components

The full TxSentinel anomaly detection system—particularly its **live detection capabilities**—is **not fully open-sourced**.

* Access is governed under a **controlled release model** (see Section 7 of the paper).
* This approach is designed to **mitigate potential misuse** while still supporting scientific validation.

## Notes

* All artifacts are designed to ensure **reproducibility, transparency, and safety**.
* No private or sensitive data is included; all resources are based on publicly accessible blockchain data.

## Citation

If you use this repository, please cite our CCS paper (details to be added).

---
