# ThreatLens — Stage 7 Comprehensive Repository Audit Report

**Date:** 2026-10-31  
**Project:** ThreatLens — Explainable AI-Powered Static Malware Detection and Analysis System  
**Auditor:** Senior Explainable AI & Machine Learning Cybersecurity Systems Engineer  
**Status:** Audit Completed  

---

## 1. Executive Summary

A comprehensive architectural, code, and security audit of the complete ThreatLens codebase (Stages 1 through 6) was conducted. The system demonstrates a high level of modularity, defensive typing, robust schema validation, and separation of concerns. All 199 existing tests pass cleanly without regression.

However, the audit confirmed critical scientific and operational constraints that must govern Stage 7:
1. **Dataset Provenance Limitation**: The models and artifacts currently deployed were trained on synthetic data (`synthetic-ember-2018`) because the previously attempted download of the genuine 1.70 GB EMBER 2018 archive was truncated (`1,572,003,840` bytes vs `1,696,539,273` bytes expected).
2. **Two Incompatible Feature Pipelines**: The custom static PE extractor parses 85 structural features via `pefile`, whereas the LightGBM classifier requires exactly 2,351 EMBER 2018 features. These pipelines are strictly isolated.
3. **Model Provenance Tri-State**: Backend endpoints and dashboard indicators currently utilize static synthetic notices. They must be refactored into a dynamic tri-state model status:
   - `SYNTHETIC — RESEARCH ONLY`
   - `GENUINE DATASET — EVALUATION MODEL`
   - `MODEL UNAVAILABLE`
4. **Security & Input Sanitization**: Upload validation must enforce strict DOS header magic (`MZ`) checks, path traversal mitigation, and upload processing timeouts.

---

## 2. Stage-by-Stage Architectural Inspection

### 2.1 Stage 1: Dataset Infrastructure & Preprocessing
* **Implementation**: [`threatlens/dataset/ember_loader.py`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/dataset/ember_loader.py), [`threatlens/dataset/validator.py`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/dataset/validator.py), [`threatlens/dataset/split.py`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/dataset/split.py).
* **Audit Findings**:
  - `ember_loader.py` includes a `download_dataset()` method, but it lacks SHA-256 verification, resumable download chunking (`Range` HTTP header), and explicit user confirmation prompts for multi-gigabyte transfers.
  - The partial archive on disk (`data/raw/ember_dataset_2018_2.tar.bz2`) has size `1,572,003,840` bytes and SHA-256 `fb2213e9aeb8a18b8bc47d3f4b6d9dd67ed8b897548aa6c65e500f996857851e`, failing against the official checksum `b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812`.
  - Splitting logic (`DataSplitter`) correctly implements leak-free stratified splitting (60% Train, 15% Val, 15% Cal, 10% Test) with fixed seed `42`.

### 2.2 Stage 2: Custom Static PE Feature Extraction & Adapter
* **Implementation**: [`threatlens/core/extractor.py`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/core/extractor.py), [`threatlens/core/schema.py`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/core/schema.py), [`threatlens/core/adapter.py`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/core/adapter.py).
* **Audit Findings**:
  - Extracts 62 raw static features and transforms them into an 85-dimensional numerical vector via [`ThreatLensCoreAdapter`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/core/adapter.py).
  - Strictly non-executing; utilizes `pefile` without invoking binary execution.
  - Correctly guards against silent zero-padding into the EMBER 2,351-dimensional pipeline by raising `IncompatibleFeatureError`.

### 2.3 Stage 3: LightGBM Classification Engine
* **Implementation**: [`threatlens/model/trainer.py`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/model/trainer.py), [`threatlens/model/classifier.py`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/model/classifier.py), [`threatlens/model/evaluator.py`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/model/evaluator.py).
* **Audit Findings**:
  - Trains a GBDT binary classifier on 2,351 features with early stopping on validation loss.
  - Model serialization to text booster (`models/lightgbm_ember.txt`) and metadata (`models/model_metadata.json`) is deterministic and reproducible.
  - Evaluator correctly calculates FNR, FPR, ROC-AUC, and PR-AUC, explicitly emphasizing the cybersecurity severity of false negatives (missed malware).

### 2.4 Stage 4: SHAP Explainability Engine
* **Implementation**: [`threatlens/explainability/explainer.py`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/explainability/explainer.py), [`threatlens/explainability/global_explanations.py`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/explainability/global_explanations.py), [`threatlens/explainability/feature_interpreter.py`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/explainability/feature_interpreter.py).
* **Audit Findings**:
  - Utilizes `shap.TreeExplainer` on LightGBM tree structures.
  - Verifies exact additivity ($|E[f(x)] + \sum \phi_i - f(x)| < 10^{-6}$) on both single and batch predictions.
  - Distinguishes mathematical differences between SHAP feature attributions, LightGBM Gain, and LightGBM Split metrics.

### 2.5 Stage 5: Probability Calibration & FastAPI Backend
* **Implementation**: [`threatlens/model/calibration.py`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/model/calibration.py), [`threatlens/model/threshold_analysis.py`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/model/threshold_analysis.py), [`threatlens/api/main.py`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/api/main.py).
* **Audit Findings**:
  - Implements Platt scaling (sigmoid) and isotonic regression fitted exclusively on `X_cal.npy` / `y_cal.npy`, preventing test set leakage.
  - Evaluates ECE (Expected Calibration Error) and MCE across 10 reliability bins.
  - Threshold analysis accurately defines operational trade-offs (High Security vs Low Interruption).
  - FastAPI backend implements lifespan loading of model singletons and Pydantic v2 schemas.

### 2.6 Stage 6: Streamlit Security Analyst Dashboard
* **Implementation**: [`threatlens/dashboard/app.py`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/dashboard/app.py), [`threatlens/dashboard/api_client.py`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/dashboard/api_client.py), pages, and components.
* **Audit Findings**:
  - Comprehensive dark SOC UI with 7 pages.
  - Page 2 maintains strict separation between Mode A (Custom 85-dim extraction only) and Mode B (EMBER 2,351-dim inference demo).
  - Threshold slider on Page 3 re-evaluates labels locally without re-running inference.
  - Zero binary caching: uploaded samples are parsed in temporary storage and deleted in `try...finally`.

---

## 3. Vulnerability, Defect & Leakage Analysis

| Category | Finding | Severity | Proposed Remediation in Stage 7 |
| :--- | :--- | :---: | :--- |
| **Data Integrity** | Existing archive `ember_dataset_2018_2.tar.bz2` is truncated (1.46 GB) with checksum mismatch. | High | Implement standalone resumable download utility (`threatlens/data/download_ember.py`) with SHA-256 verification and explicit user confirmation. |
| **Model Provenance** | Model status is currently embedded via static disclaimers rather than dynamic metadata-driven state. | Medium | Formalize `ModelProvenance` enum (`SYNTHETIC`, `GENUINE_EMBER`, `UNAVAILABLE`) across model metadata, API schemas, and dashboard badges. |
| **Feature Pipeline** | Custom 85-dim PE extractor cannot produce EMBER 2,351 features. | Architectural | Document feature pipeline incompatibility in depth; provide architectural blueprint for native EMBER extraction or dedicated 85-dim classifier. |
| **Upload Security** | Uploaded PE binaries are checked by extension and size, but lack magic byte validation before temporary write. | Medium | Add pre-parse magic byte check (`b"MZ"` header verification) and filename sanitization against path traversal. |
| **Data Leakage** | All scalers, models, and calibrators are verified to respect split boundaries without test contamination. | Informational | Maintain strict split isolation; add automated test ensuring test split is never touched during fitting. |

---

## 4. Conclusion & Action Plan for Stage 7

The codebase is architecturally solid and well-tested (199 passing tests). Stage 7 will proceed with:
1. Creating the resumable, checksum-verified dataset acquisition utility (`threatlens/data/download_ember.py`).
2. Implementing the strict dataset report generator (`artifacts/dataset_report.json`).
3. Formalizing dynamic model status indicators (`SYNTHETIC — RESEARCH ONLY` vs `GENUINE DATASET — EVALUATION MODEL`).
4. Applying security hardening to upload handling, path sanitization, and API boundaries.
5. Providing comprehensive documentation (`docs/dataset_setup.md`, `docs/security_audit.md`, `docs/model_card.md`, `docs/architecture.md`).
