# ThreatLens — Model Card: Static Malware Detection LightGBM Classifier

**Model Name:** ThreatLens-LightGBM-EMBER  
**Model Version:** 1.0.0  
**Model Type:** Gradient Boosted Decision Trees (LightGBM)  
**Date:** 2026-10-31  
**Status:** `SYNTHETIC — RESEARCH ONLY`  
**License:** MIT  

---

## 1. Model Details

* **Architecture**: LightGBM Binary Classification Booster (`gbdt` objective: `binary`).
* **Input Dimensionality**: 2,351 continuous and discrete numerical features adhering to the EMBER 2018 schema.
* **Output**: Calibrated probability $P(y = 1 \mid x) \in [0.0, 1.0]$ and discrete prediction ($\text{label} \in \{0: \text{Benign}, 1: \text{Malicious}\}$).
* **Calibration Layer**: Post-processing Platt scaling (parametric sigmoid) fitted exclusively on calibration split (`data/processed/X_cal.npy`).
* **Explainability Layer**: TreeSHAP ([`shap.TreeExplainer`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/explainability/explainer.py)) with mathematically verified local additivity ($|E[f(x)] + \sum \phi_i - f(x)| < 10^{-6}$).

---

## 2. Intended Use & Out-of-Scope Applications

### Primary Intended Use:
* **Academic & Architectural Research**: Benchmarking explainability mechanics (SHAP), post-hoc probability calibration, and operational threshold trade-offs on high-dimensional static PE representations.
* **SOC Analyst Demonstration**: Demonstrating how machine learning predictions can be paired with local feature explanations and interactive decision cutoffs.

### Out-of-Scope & Prohibited Applications:
* **Production Endpoint Protection**: ThreatLens must **NOT** be used as a standalone or primary antivirus, EDR, or perimeter gateway blocker.
* **Autonomous Quarantine**: Due to the synthetic training distribution, autonomous file deletion or process termination based on ThreatLens scores is strictly prohibited.
* **Raw PE Binary Evaluation on Model A**: The classifier cannot evaluate raw binary files directly; it operates only on validated pre-extracted 2,351-dimensional EMBER vectors.

---

## 3. Dataset Provenance & Known Limitations

> [!WARNING]
> ### Critical Dataset Disclosure: `synthetic-ember-2018`
> The active model artifacts (`models/lightgbm_ember.txt`) were trained on synthetic benchmark distributions generated to validate pipeline integrity while the multi-gigabyte genuine EMBER 2018 download was incomplete.
> 
> * **Test Accuracy**: $50.56\%$
> * **ROC-AUC**: $50.74\%$
> * **Implication**: Model predictions represent random chance (coin-flip). The feature attributions explain internal model decisions on synthetic patterns, not genuine malware signatures.

### Transition to Genuine EMBER:
To upgrade this model to genuine threat discrimination:
1. Complete verified download of `ember_dataset_2018_2.tar.bz2` (SHA-256: `b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812`) via:
   ```bash
   python -m threatlens.data.download_ember --download --confirm
   ```
2. Unpack and run `scripts/prepare_dataset.py`.
3. Retrain model via `scripts/train_model.py`.

---

## 4. Feature Extraction Methodology

ThreatLens maintains two strictly isolated feature pipelines:

1. **Pipeline A (EMBER 2018 Representation — 2,351 Features)**:
   - Byte Histogram (256): Normalized byte frequency across file.
   - Byte-Entropy Histogram (256): Joint 2D distribution of byte values vs sliding-window Shannon entropy.
   - String Statistics (256): Counts, printable character distribution, entropy, URL/registry markers.
   - General File Info (10): Raw size, virtual size, section count, symbols.
   - Header Characteristics (61): COFF characteristics, subsystem flags, timestamps.
   - Section Table (196): MurmurHash buckets, section entropies, and characteristics.
   - Imported Functions (1,248): MurmurHash representation of `dll:function` pairs.
   - Exported Functions (68): MurmurHash representation of exported symbols.

2. **Pipeline B (ThreatLens Custom PE Extractor — 85 Features)**:
   - Non-executing structural parsing via `pefile`.
   - Preserves semantic transparency (raw section entropies, permissions, exact DLL counts, suspicious API categories, Authenticode table).
   - Incompatible with Pipeline A; strictly isolated to prevent feature contamination.

---

## 5. Training Methodology & Leakage Prevention

* **Split Configuration**: Stratified 4-way split with fixed seed `42`:
  - **Train Split (60%, 1,440 samples)**: Fits imputer, standard scaler, and LightGBM trees.
  - **Validation Split (15%, 360 samples)**: Monitored for early stopping (50 rounds patience).
  - **Calibration Split (15%, 360 samples)**: Reserved exclusively for Platt scaling.
  - **Test Split (10%, 240 samples)**: Completely held out until final evaluation.
* **Leakage Verification**: Zero index overlap across splits ($A \cap B = \emptyset$). Scalers and calibrators never touch test samples.

---

## 6. Operational Cybersecurity Trade-offs

| Threshold ($t$) | Posture Role | Recall (TPR) | FNR (Missed Malware) | FPR (Alert Fatigue) | Strategic Context |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0.45** | **High Security** | 87.2% | **12.8%** | 90.0% | Deep quarantine inspection; missing threats is unacceptable. |
| **0.50** | **Balanced** | 55.0% | 45.0% | 51.7% | Baseline operating trade-off. |
| **0.55** | **Low Interruption** | 13.3% | 86.7% | **17.2%** | Automated endpoint blocking; false positives disrupt business. |

---

## 7. Ethical & Security Considerations

* **Dual-Use Concerns**: Model interpretability aids defenders in diagnosing malware features; however, adversarial attackers can exploit SHAP explanations to craft evasion attacks (e.g. padding bytes to mimic benign histogram distributions).
* **Defensive Engineering**: Explainability output is intended for vetted security analysts. Automated pipeline blocking should combine ML scores with YARA signatures, Authenticode verification, and behavioral sandboxing.
