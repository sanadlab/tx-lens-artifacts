# TxSentinel-Artifacts

Artifacts for our CCS 2026 paper: *Real-Time Anomaly Detection of Ethereum Transactions via Representation Learning*. 

## Overview

This repository substantiates our claims regarding model efficiency, real-time latency, and baseline detection performance for Ethereum transaction anomaly detection. To support reproducibility while minimizing potential ecosystem risks, we release a curated set of artifacts that enable verification of our results without exposing sensitive components of the live detection system.

## Released Artifacts

### 1. Evaluation Dataset

We provide a curated benchmark dataset comprising **439 Ethereum transaction traces**, including both malicious and benign samples, collected between **2023 and 2025**. See the **Evaluation-Dataset** folder.

*   The dataset is derived entirely from public on-chain data on the Ethereum mainnet.
*   Researchers can reconstruct execution traces using any standard Ethereum archive node.
*   The dataset is also publicly deposited via Zenodo for long-term access and reproducibility.

---

### 2. Table-1: TxSentinel Predictions

This component reproduces the results of TxSentinel across three representation paradigms using precomputed model predictions.

**Contents**

The evaluation uses the following fixed files:

*   `ground-truth-labels.csv` — manually curated labels for the 439 benchmark transactions.
*   `P1-prediction_results.csv` — frozen transformer embeddings.
*   `P2-prediction_results.csv` — contrastively fine-tuned embeddings.
*   `P3-prediction_results.csv` — expert-engineered features.

Each file contains prediction scores for the following evaluated models:

*   Adversarial Autoencoder (AAE).
*   Random Forest (RF).
*   Extremely Randomized Trees (ETs).
*   XGBoost (XGB).
*   Positive-Unlabeled Learner (nnPU).

**Execution**

To reproduce Table 1:

```bash
python run.py
```

**Output**

The script generates:

*   `Table-1-TxSentinel-Predictions.csv`

**Reported Metrics**

For each `(Experiment, Model)` pair, the following metrics are computed to handle extreme class imbalance:

*   Accuracy.
*   Specificity.
*   Sensitivity.
*   PR-AUC.

All metrics are reported as percentages with one decimal place. 

**Reproducibility Notes**

*   Metrics are computed deterministically from the provided predictions.
*   No training or randomness is involved in this reproduction script.
*   The output CSV exactly matches the results reported in Table 1 of the paper.

---

### 3. Table-2: Ablation Study on P1

This component reproduces the ablation study for the P1 paradigm by measuring the effect of trace summarization on detection performance.

**Contents**

The evaluation uses the following fixed files:

*   `ground-truth-labels.csv` — manually curated labels.
*   `P1_With_TxSummarization.csv` — predictions generated with transaction summarization.
*   `P1_Without_TxSummarization.csv` — predictions generated using raw, unprocessed JSON transaction traces.

The following models are evaluated: AAE, RF, ETs, XGB, and nnPU.

**Execution**

To reproduce Table 2:

```bash
python run.py
```

**Output**

The script generates:

*   `Table-2-Ablation-Study.csv`

**Reported Metrics**

For each model, the output CSV reports:

*   Balanced Accuracy with Tx Summarization (%).
*   Balanced Accuracy without Tx Summarization (%).
*   Degradation (%).

The degradation is computed as the absolute drop between the two settings in percentage points.

---

### 4. Table-3: State-of-the-Art Predictions

This component evaluates external state-of-the-art tools on the same benchmark dataset using their binary predictions.

**Contents**

The evaluation uses the following fixed files:

*   `ground-truth-labels.csv` — manually curated labels.
*   `sota_labeled_output.csv` — tool predictions derived from public web interfaces.

The following prediction columns are evaluated against our benchmark:

*   `etherscan_label` → Etherscan.
*   `blocksec_label` → BlockSec Explorer.
*   `certik_label` → CertiK Skylens.

**Execution**

To reproduce Table 3:

```bash
python run.py
```

**Output**

The script generates:

*   `Table-3-State-of-the-Art-Predictions.csv`

**Reported Metrics**

For each tool, the following metrics are computed:

*   Accuracy.
*   Sensitivity.
*   Specificity.
*   PR-AUC.

---

### 5. Reproducibility Scripts

We provide minimal Python scripts required to reproduce the empirical results reported in the paper and evaluate performance metrics using the released predictions.

### 6. Controlled Release Components

The full TxSentinel anomaly detection system—particularly its **live detection capabilities**—is **not fully open-sourced**. 

*   Access to the live inference system, pre-trained model weights, and source code is governed under a **controlled release model** via Hugging Face Spaces. 
*   This approach is designed to mitigate potential misuse (such as adversarial adaptation or evasion testing) while still supporting scientific validation.
*   Interested parties can request access by submitting a form and agreeing to non-commercial research use terms at https://huggingface.co/spaces/blockchainsecurity/EthTxShieldReleaseAgreement.

## Notes

*   All artifacts are designed to ensure reproducibility, transparency, and safety.
*   No private or sensitive data is included; all resources are based on publicly accessible blockchain data, and all address anonymization in our pipeline protects pseudonymous identities.

## Citation

If you use this repository or our datasets, please cite our CCS 2026 paper:

```bibtex
@inproceedings{txsentinel2026,
  author = {Anonymous Author(s)},
  title = {Real-Time Anomaly Detection of Ethereum Transactions via Representation Learning},
  booktitle = {Proceedings of The ACM Conference on Computer and Communications Security (CCS)},
  year = {2026},
  publisher = {ACM},
  address = {New York, NY, USA},
  pages = {14 pages}
}
```