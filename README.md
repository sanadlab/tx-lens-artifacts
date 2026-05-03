Here is a polished, professional `README.md` version of your content:

---

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


### 2. Table-1: TxSentinel Predictions

This repository provides **precomputed model predictions and evaluation metrics** for the benchmark dataset to support artifact evaluation and reproducibility.

#### Contents

* **Prediction Files**
  For each experimental setting, we provide model outputs in:

  * `P1-prediction_results.csv`
  * `P2-prediction_results.csv`
  * `P3-prediction_results.csv`

  Each file contains:

  * Transaction identifiers (`hash`)
  * Model scores for the following methods:

    * AAE, RF, ET, XGB, PU

* **Summary Metrics**
  The file:

  * `Table-1-TxSentinel-Predictions.csv`

  reports aggregated evaluation results for all `(Experiment, Model)` pairs.

#### Metrics

All metrics are reported as **percentages (0–100 scale)**:

* **Accuracy**
* **Specificity**
* **Sensitivity (Recall)**
* **PR-AUC (Average Precision)**

#### Experimental Settings

The results are organized across three experimental configurations:

* **P1: Frozen embeddings**
* **P2: Fine-tuned embeddings**
* **P3: Expert features**

#### Reproducibility

The summary file (`Table-1-TxSentinel-Predictions.csv`) can be regenerated using the provided script:

```bash
python run.py
```

This script:

1. Loads ground-truth labels from `ground-truth-labels.csv`
2. Merges them with prediction files
3. Applies fixed thresholds (as specified in the paper)
4. Computes all reported metrics
5. Outputs the final CSV file

No additional training or external dependencies are required beyond standard Python libraries.



### 3. State-of-the-Art Predictions

To enable fair comparison, we include benchmarking results from three prominent academic systems:

* *BlockGPT*
* *BlockLens*
* *BlockScan*

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
