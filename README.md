# ThreatLens – Explainable AI-Powered Static Malware Detection and Analysis System

> **Stage 1: Project Foundation and Dataset Preparation** ✅ Complete  
> **Stage 2: Static PE Feature Extraction & Model Adapter Interface** ✅ Complete  
> **Stage 3: Machine Learning Classification Engine (LightGBM)** ✅ Complete  
> **Stage 4: Explainable AI Engine Using SHAP (Global & Local Explanations)** ✅ Complete  
> **Stage 5: Probability Calibration & FastAPI Backend Service** ✅ Complete  
> **Stage 6: Streamlit Security Analyst Dashboard / UI** ✅ Complete  
> **Stage 7: Final Integration, Verified Dataset Tooling, Testing & Production Hardening** ✅ Complete  

ThreatLens is a modular, production-ready static malware detection and analysis system for Windows PE executables. It combines static structural analysis using `pefile` with gradient-boosted decision trees (LightGBM), strict feature schema validation, SHAP-based explainability, rigorous probability calibration (Platt scaling & Isotonic regression), an asynchronous FastAPI REST microservice, an interactive Streamlit SOC Analyst Console, and cryptographically verified genuine dataset acquisition pipelines, while supporting the benchmark **EMBER 2018** dataset representation.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- 8 GB+ RAM (for full EMBER dataset)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd MalExplain

# Create virtual environment
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
MalExplain/
├── .gitignore                 # Excludes data/, models/, artifacts/, logs/, .venv/
├── requirements.txt           # Python dependencies (pinned versions)
├── pytest.ini                # Pytest configuration
├── README.md                 # This file
├── docs/
│   ├── pe-feature-extraction.md # Comprehensive PE extraction documentation
│   ├── explainability.md        # Comprehensive SHAP explainability engine documentation
│   ├── api.md                   # Comprehensive FastAPI endpoints & architecture guide
│   ├── dashboard.md             # Streamlit SOC Analyst Console guide & page reference
│   ├── stage7_audit.md          # Comprehensive Stage 7 repository & architecture audit
│   ├── dataset_setup.md         # Genuine EMBER 2018 dataset acquisition & verification guide
│   ├── security_audit.md        # Focused cybersecurity & production hardening audit
│   ├── model_card.md            # Comprehensive model card & limitations disclosure
│   └── architecture.md          # System architecture & component design
│
├── threatlens/               # Main package
│   ├── __init__.py
│   ├── config.py             # Centralized configuration (seeds, paths, model, calibration, API, dashboard)
│   │
│   ├── data/                 # Stage 7: Genuine Dataset Acquisition & Management
│   │   ├── __init__.py       # Exports EmberDatasetAcquisition
│   │   └── download_ember.py # Verified, resumable EMBER 2018 acquisition utility
│   │
│   ├── core/                 # Stage 2: Static PE Analysis & Adapters
│   │   ├── __init__.py
│   │   ├── extractor.py      # PEFeatureExtractor (pefile-based, non-executing)
│   │   ├── schema.py         # PE_FEATURE_SCHEMA (v1.0.0, strict validation)
│   │   └── adapter.py        # FeatureVectorAdapter (ThreatLens Core vs EMBER isolation)
│   │
│   ├── model/                # Stage 3 & 5: ML Classification, Calibration & Thresholds
│   │   ├── __init__.py
│   │   ├── trainer.py        # ModelTrainer (LightGBM on 2351 EMBER features, early stopping)
│   │   ├── evaluator.py      # ModelEvaluator (ROC, PR, Confusion Matrix, FNR/FPR, PNGs)
│   │   ├── classifier.py     # MalwareClassifier (Inference engine, calibration support, batch)
│   │   ├── calibration.py    # ProbabilityCalibrator (Platt scaling / Isotonic, ECE/MCE)
│   │   └── threshold_analysis.py # ThresholdEvaluator (trade-off curves & operational postures)
│   │
│   ├── explainability/       # Stage 4: Explainable AI Engine Using SHAP
│   │   ├── __init__.py
│   │   ├── explainer.py      # ShapExplainer (TreeExplainer, additivity, single/batch)
│   │   ├── feature_interpreter.py # FeatureInterpreter (8 EMBER groups, canonical descriptions)
│   │   └── global_explanations.py # GlobalExplainer (global metrics, beeswarm, bar, group plots)
│   │
│   ├── api/                  # Stage 5: Asynchronous FastAPI REST Microservice
│   │   ├── __init__.py       # Exports app & create_app
│   │   ├── main.py           # Application factory, lifespan, CORS, error handlers
│   │   ├── dependencies.py   # ModelContainer singleton & FastAPI dependencies
│   │   ├── schemas/          # Pydantic v2 validation schemas
│   │   │   ├── requests.py   # PredictRequest, BatchPredictRequest, ExplainRequest
│   │   │   └── responses.py  # HealthResponse, PredictResponse, ExplainResponse, etc.
│   │   └── routes/           # Modular route controllers
│   │       ├── health.py     # GET /health
│   │       ├── model_info.py # GET /model/info, GET /metrics/model
│   │       ├── prediction.py # POST /predict, POST /predict/batch
│   │       └── explanation.py# POST /explain
│   │
│   ├── dashboard/            # Stage 6: Streamlit Security Analyst Dashboard
│   │   ├── __init__.py       # Exports ThreatLensAPIClient and exceptions
│   │   ├── app.py            # Main Streamlit application entrypoint & routing
│   │   ├── api_client.py     # Resilient HTTP client for FastAPI endpoints
│   │   ├── state.py          # Session state lifecycle manager (no raw binary caching)
│   │   ├── components/       # Reusable dark SOC UI widgets and Plotly charts
│   │   │   ├── styles.py     # Dark SOC theme CSS & warning banners
│   │   │   ├── metrics.py    # Status cards, classification badges, metric rows
│   │   │   └── charts.py     # Dark Plotly chart builders (gauge, SHAP bar, trade-offs)
│   │   └── pages/            # 7 dedicated dashboard pages
│   │       ├── security_overview.py # Page 1: Health, status cards, test metrics
│   │       ├── analyze_sample.py    # Page 2: PE static extractor & EMBER vector demo
│   │       ├── prediction_results.py# Page 3: Calibrated risk gauge, threshold slider
│   │       ├── local_shap.py        # Page 4: TreeSHAP local attribution bars & additivity
│   │       ├── global_analysis.py   # Page 5: Macro feature importance, 8 groups
│   │       ├── threshold_analysis.py# Page 6: Simulated operating postures & curves
│   │       └── model_information.py # Page 7: Architecture specs & limitations
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logger.py         # Structured logging with rotation
│   │
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── ember_loader.py   # EMBER 2018 download, load, synthetic generation
│   │   ├── validator.py      # Comprehensive data validation suite
│   │   └── split.py          # Train/Val/Cal/Test splitting (no leakage)
│   │
│   └── preprocessing/
│       ├── __init__.py
│       ├── pipeline.py       # EmberPreprocessor (impute + scale, fitted on train only)
│       └── metadata.py       # Dataset metadata & config export
│
├── scripts/
│   ├── explain.py            # CLI: Run global & local SHAP explanations and export reports
│   ├── train_model.py        # CLI: Train LightGBM model & generate evaluation reports
│   ├── predict.py            # CLI: Run validated inference on feature vectors
│   ├── extract_features.py   # CLI: Extract static PE features to JSON / model vector
│   ├── prepare_dataset.py    # CLI: Full Stage 1 dataset preparation pipeline
│   └── run_validation_report.py # CLI: Generate Markdown validation report
│
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_dataset_validation.py
│   ├── test_ember_loader.py
│   ├── test_preprocessing.py
│   ├── test_pe_extractor.py  # Stage 2: PE extraction tests
│   ├── test_feature_schema.py# Stage 2: Schema validation tests
│   ├── test_model_adapter.py # Stage 2: Adapter & pipeline isolation tests
│   ├── test_model_training.py# Stage 3: Model training & serialization tests
│   ├── test_classifier.py    # Stage 3: Classifier inference & dimension tests
│   ├── test_evaluator.py     # Stage 3: Evaluation metrics & artifact generation tests
│   ├── test_explainability.py# Stage 4: SHAP explainer & global/local interpretation tests
│   ├── test_calibration.py   # Stage 5: Probability calibration & threshold analysis tests
│   ├── test_api.py           # Stage 5: FastAPI REST endpoint & schema validation tests
│   ├── test_dashboard.py     # Stage 6: Streamlit dashboard, API client, & visualization tests
│   └── test_stage7_integration.py # Stage 7: Acquisition, security, provenance & leakage tests
│
├── data/                     # Git-ignored: raw & processed datasets
│   ├── raw/                  # EMBER JSONL/CSV files
│   └── processed/            # Train/Val/Cal/Test .npy splits (X_train, y_train, etc.)
│
├── models/                   # Git-ignored: trained model artifacts
│   ├── lightgbm_ember.txt    # Serialized LightGBM Booster (text format)
│   ├── model_metadata.json   # Model hyperparams, iteration, provenance metadata
│   ├── calibrator.joblib     # Serialized fitted ProbabilityCalibrator (Platt scaling)
│   └── calibration_metadata.json # Calibration split Brier score, LogLoss, ECE, MCE
│
├── artifacts/                # Git-ignored: fitted preprocessor, metadata
│   ├── preprocessor.joblib   # Fitted EmberPreprocessor
│   └── dataset_metadata.json # Complete dataset provenance
│
├── logs/                     # Git-ignored: application logs
│
└── reports/                  # Generated reports and evaluation plots
    ├── classification_report.json
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── precision_recall_curve.png
    ├── feature_importance.json
    ├── shap_global_importance.json
    ├── shap_summary.png
    ├── shap_bar.png
    ├── shap_group_importance.png
    ├── threshold_analysis.json
    └── threshold_analysis.png
```

---

## 🤖 Stage 3: Machine Learning Classification Engine

ThreatLens trains a primary LightGBM gradient boosted binary classifier on the 2,351-dimensional EMBER 2018 representation:

- **Classes**: `0` (Benign) and `1` (Malicious).
- **Early Stopping**: Monitored on the validation split (`X_val`, `y_val`) to prevent overfitting.
- **Calibration Split**: Kept untouched for Stage 4 probability calibration.
- **Test Split**: Evaluated exclusively at the conclusion of training.
- **False Negative Focus**: Explicit calculation of False Negative Rate ($\text{FNR} = \frac{\text{FN}}{\text{TP} + \text{FN}}$) alongside False Positive Rate ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$).

### Train the Model via CLI:

```bash
# Train LightGBM model and generate evaluation plots
python scripts/train_model.py
```

### Run Model Inference via CLI:

```bash
# Predict on a 2,351-dimensional EMBER feature vector (.npy or .json)
python scripts/predict.py path/to/vector.npy

# Predict with custom decision threshold
python scripts/predict.py path/to/vector.npy --threshold 0.7 -o artifacts/prediction.json
```

### Python API Usage:

```python
from threatlens.model import MalwareClassifier

# Initialize inference classifier
classifier = MalwareClassifier()

# Single sample prediction (shape: 2351,)
result = classifier.predict_single(feature_vector, threshold=0.5)
print(f"Prediction: {result.label_name} (Probability: {result.probability:.4f})")

# Batch prediction (shape: N, 2351)
batch = classifier.predict_batch(feature_matrix)
print(f"Batch Malicious Count: {sum(batch.labels)}")
```

---

## 🔍 Stage 2: Static PE Feature Extractor & Adapters

ThreatLens extracts 62 canonical static features from PE files without ever executing them:

- **File Information**: Size, type (`PE32`/`PE32+`), machine architecture, timestamps, checksum.
- **PE Headers**: Full DOS header, COFF characteristics, Optional header fields, subsystem, and DLL flags.
- **Sections**: Detailed per-section metrics, Shannon entropy calculation ($0.0 \le H \le 8.0$), raw/virtual sizes, and permissions (readable, writable, executable, code/data).
- **Imports**: DLL count, function count, import names/ordinals, and anomalous indicators (empty names, ordinal-only imports, sensitive APIs).
- **Exports**: Export counts, function names, and RVAs.
- **Resources**: Resource count, types (`RT_ICON`, `RT_VERSION`, `RT_MANIFEST`), languages, and sizes.
- **Additional Data**: Appended overlay detection and entropy, digital certificate table, and debug directories.

### Extract Features via CLI:

```bash
# Extract features and display JSON to stdout
python scripts/extract_features.py path/to/binary.exe

# Save output to JSON with strict schema validation
python scripts/extract_features.py path/to/binary.exe --strict -o artifacts/features.json

# Extract features and generate 85-dimensional model vector
python scripts/extract_features.py path/to/binary.exe --vectorize -o artifacts/vectorized.json
```

---

## 🛡️ Strict Pipeline Isolation

ThreatLens strictly maintains two separate pipelines:

1. **EMBER 2018 Pipeline**:
   - 2,351 features per sample.
   - Built from raw byte histograms, 2D byte entropy histograms, string statistics, and hashed structural tables.
   - Used by `MalwareClassifier` and `ModelTrainer`.
2. **ThreatLens Custom PE Pipeline**:
   - 85 features per sample.
   - Built from static structural parsing using `pefile`.
   - Used by `ThreatLensCoreAdapter`.

> [!CAUTION]
> Custom PE features are **never silently mapped or zero-padded** into EMBER feature representations. `MalwareClassifier` and `EmberFeatureAdapter` will explicitly raise `IncompatibleFeatureError` if fed custom PE features, protecting models from invalid inference.

---

## 🔧 Stage 1: Dataset Preparation

### Run Full Dataset Pipeline:

```bash
# Synthetic test run (instant validation)
python scripts/prepare_dataset.py --max-samples 2000 --verbose

# Full run with EMBER 2018 download (~4 GB)
python scripts/prepare_dataset.py --verbose

# Generate human-readable Markdown validation report
python scripts/run_validation_report.py
```

---

## 🔍 Stage 4: SHAP Explainability Engine

ThreatLens provides a mathematically verified SHAP explainability layer leveraging `shap.TreeExplainer` on the 2,351 EMBER features.

### Run Global Explainability Analysis:

```bash
# Calculate global SHAP feature importance & export summary, bar, and group plots
python scripts/explain.py --mode global --data-path data/processed/X_test.npy --output-dir reports
```

Generated artifacts:
- `reports/shap_global_importance.json` – Full global rankings, group importance metrics, and LightGBM gain/split comparisons.
- `reports/shap_summary.png` – Publication-grade SHAP beeswarm summary plot.
- `reports/shap_bar.png` – Global feature importance bar chart (mean absolute SHAP).
- `reports/shap_group_importance.png` – Dual-panel EMBER schema group attribution chart.

### Run Local Prediction Explanation:

```bash
# Explain sample 0 from test split
python scripts/explain.py --mode local --sample-index 0

# Explain custom feature vector and export JSON explanation
python scripts/explain.py --mode local --input-path path/to/sample.npy --output reports/sample_explanation.json
```

---

## ⚖️ Stage 5: Probability Calibration & Threshold Analysis

Tree-based gradient boosted models often produce uncalibrated probabilities that cluster around margin decision boundaries. ThreatLens implements a mathematically sound post-processing probability calibration layer fitted exclusively on the isolated calibration split (`data/processed/X_cal.npy`, `y_cal.npy`) without data leakage.

### Calibration Features:
- **Platt Scaling (Sigmoid)**: Parametric logistic regression fitted over raw model probabilities $P(y=1|x) = \frac{1}{1 + \exp(A \cdot p + B)}$. Selected as the production default for small/moderate calibration sets ($N=360$) to avoid non-parametric overfitting.
- **Isotonic Regression**: Non-parametric piecewise constant isotonic fitting.
- **Metrics Computed**: Brier score ($0 \le \text{Brier} \le 1$), Logarithmic Loss, Expected Calibration Error (ECE), and Maximum Calibration Error (MCE) across 10 uniform reliability bins.
- **Operational Threshold Analysis**: Sweeps classification thresholds ($0.05 \le t \le 0.95$) to calculate TPR, FPR, FNR, Precision, F1, Specificity, and Balanced Accuracy. Evaluates security-critical postures:
  - **High Security (Low FNR)**: Prioritizes catching every malicious sample ($FNR \le 15\%$), trading off higher SOC alert triage.
  - **Low Interruption (Low FPR)**: Prioritizes business continuity ($FPR \le 20\%$) to minimize false positives in automated blocking.
  - **Balanced**: Maximizes F1 / Youden's $J$-statistic.

### Calibration & Threshold Artifacts:
- `models/calibrator.joblib` – Serialized fitted `ProbabilityCalibrator` instance.
- `models/calibration_metadata.json` – Fitted calibration metrics and provenance.
- `reports/threshold_analysis.json` – Comprehensive threshold evaluation matrix and posture recommendations.
- `reports/threshold_analysis.png` – Dual-panel trade-off visualization (ROC/PR curves and metric sweep vs. decision threshold).

---

## ⚡ Stage 5: FastAPI REST Backend

ThreatLens provides an asynchronous, production-hardened REST microservice powered by FastAPI and Pydantic v2:

- **Lifespan Initialization**: Models, preprocessors, calibrators, and SHAP explainers are loaded **once** at server startup and cached across requests.
- **Strict Input Validation**: Rejects inputs with dimension $\neq 2,351$, payloads containing `NaN` or `Inf`, and batches exceeding maximum size ($N > 100$).
- **API Documentation**: Interactive OpenAPI / Swagger UI automatically generated at `/docs` and ReDoc at `/redoc`.

### Endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check, model status, calibration status, dataset provenance disclaimer |
| `GET` | `/model/info` | Architecture, feature dimensionality, training metadata, limitations |
| `GET` | `/metrics/model` | Model evaluation metrics (ROC-AUC, PR-AUC, FNR, FPR, Brier score) |
| `POST` | `/predict` | Single sample inference (raw & calibrated probability, thresholding) |
| `POST` | `/predict/batch` | Batch inference with size enforcement ($N \le 100$) |
| `POST` | `/explain` | Local SHAP explanation with positive/negative attribution rankings & descriptions |

### Run the FastAPI Microservice:

```bash
# Launch server with Uvicorn (port 8000)
uvicorn threatlens.api.main:app --host 127.0.0.1 --port 8000 --reload

# Alternatively, run via Python module:
python -m threatlens.api.main
```

### Example API Request (Prediction):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.0, 0.0, ..., 0.0],
    "threshold": 0.5
  }'
```

See [docs/api.md](file:///C:/Users/ALPHA_JUNIOR/MalExplain/docs/api.md) for full API reference, request/response schemas, and deployment guidelines.

---

## 🖥️ Stage 6: Streamlit Security Analyst Dashboard

ThreatLens features a production-grade, dark-themed SOC analyst interface built with Streamlit and Plotly. It connects directly to the FastAPI microservice to deliver non-executing PE binary analysis, calibrated probability scoring, additive SHAP attributions, and simulated operational threshold trade-off analysis.

### Dashboard Capabilities & Architecture:
- **Page 1: Security Overview**: Real-time service health, model release specifications, confusion matrix heatmap, and untouched test split metrics loaded from `/metrics/model`.
- **Page 2: Analyze Sample**:
  - **Mode A (Custom PE Extraction)**: Accepts `.exe`, `.dll`, `.sys` binaries ($\le 50$ MB). Parses headers, entropy, sections, and imports without execution. Explicitly discloses that no classifier exists for these 85 features and computes no malware scores.
  - **Mode B (EMBER Vector Inference Demo)**: Labeled `EMBER FEATURE VECTOR DEMO — NOT RAW PE CLASSIFICATION`. Evaluates validated 2,351-dimensional vectors via `/predict`.
- **Page 3: Prediction Results**: Calibrated probability gauge chart ("Model score: 0.XX"), classification status card, and an interactive threshold slider that **re-evaluates classification locally** using received probabilities without re-running inference.
- **Page 4: Local SHAP Explainability**: Integrates `/explain` to display diverging horizontal Plotly bar charts (positive contributions in red, negative in cyan/blue), hover tooltip details, top-$K$ selector, and exact additivity error.
- **Page 5: Global Feature Analysis**: Top 20 global features by mean absolute SHAP, attribution across the 8 EMBER schema groups, and side-by-side metric comparison (SHAP vs Gain vs Split).
- **Page 6: Simulated Threshold Analysis**: Operational trade-off curves (TPR, FPR, FNR, F1 vs $t$) and dynamic posture classifier (High Security / Balanced / Low Interruption).
- **Page 7: Model Information**: Complete architecture metadata, hyperparameters, transparent breakdown of synthetic limitations, and future enterprise roadmap.

### Launch the Streamlit Dashboard:

```bash
# Ensure FastAPI is running on port 8000:
uvicorn threatlens.api.main:app --host 127.0.0.1 --port 8000 --reload

# In another terminal, launch the Streamlit console (port 8501):
streamlit run threatlens/dashboard/app.py --server.port 8501
```

See [docs/dashboard.md](file:///C:/Users/ALPHA_JUNIOR/MalExplain/docs/dashboard.md) for full dashboard architecture and page documentation.

---

## 🔒 Stage 7: Final Integration, Verified Dataset Tooling, Testing & Production Hardening

Stage 7 completes the scientific, architectural, and security hardening of ThreatLens, preparing the system for genuine benchmark dataset acquisition while transparently preserving and communicating research limitations:

### Key Hardening Deliverables:
- **Resumable Dataset Acquisition Utility** ([`threatlens/data/download_ember.py`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/data/download_ember.py)):
  - Cryptographic verification against the official EMBER 2018 SHA-256 (`b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812`).
  - Detects truncated partial downloads (such as the existing 1.46 GB partial archive on disk) and resumes via HTTP `Range` headers.
  - Requires explicit user confirmation (`--confirm` flag) to prevent accidental multi-gigabyte bandwidth consumption.
  - Mitigates Tar Slip / Zip Slip path traversal vulnerabilities during decompression.
- **Dynamic Model Provenance Tri-State**:
  - The API (`GET /health`, `GET /model/info`) and Dashboard automatically propagate the operational model state:
    - `SYNTHETIC — RESEARCH ONLY` (active status)
    - `GENUINE DATASET — EVALUATION MODEL`
    - `MODEL UNAVAILABLE`
- **Security & Pre-Parse Hardening**:
  - Ingestion checks verify PE DOS signature (`b"MZ"` / `0x5A4D`) before allocating memory or disk space.
  - Sanitizes upload filenames against path traversal.
  - Suppression of internal file system paths and stack traces in API error responses.
- **Comprehensive Documentation Suite**:
  - [`docs/stage7_audit.md`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/docs/stage7_audit.md) – Full repository and architectural audit.
  - [`docs/dataset_setup.md`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/docs/dataset_setup.md) – Official EMBER acquisition and validation guide.
  - [`docs/security_audit.md`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/docs/security_audit.md) – Cybersecurity threat model and defensive controls.
  - [`docs/model_card.md`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/docs/model_card.md) – Ethical considerations, limitations, and operational trade-offs.
  - [`docs/architecture.md`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/docs/architecture.md) – Component flowcharts and pipeline isolation design.
  - [`artifacts/dataset_report.json`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/artifacts/dataset_report.json) – Verified split counts, zero missing values, and leakage-free validation.

### Check EMBER Archive Status:

```bash
python -m threatlens.data.download_ember --check
```

---

## 🧪 Testing

The repository includes a comprehensive 211-test regression test suite:

```bash
# Run all 211 tests
pytest -v

# Run Stage 7 integration and security tests specifically
pytest tests/test_stage7_integration.py -v
```

### Test Coverage Summary:

| Module | Test File | Tests | Focus |
|--------|-----------|-------|-------|
| `config` | `test_config.py` | 15 | Configuration, paths, EMBER constants, calibration/API/dashboard configs |
| `validator` | `test_dataset_validation.py` | 20 | NaN/Inf checks, distribution, corruption |
| `ember_loader` | `test_ember_loader.py` | 14 | Dataset downloads, JSONL parsing, synthetic data |
| `preprocessing` | `test_preprocessing.py` | 15 | Stratified splitting, imputation, scaling |
| `extractor` | `test_pe_extractor.py` | 14 | Valid/invalid PE, empty, truncated, entropy, perms |
| `schema` | `test_feature_schema.py` | 12 | Schema v1.0.0, strict rejection, ordering |
| `adapter` | `test_model_adapter.py` | 12 | Vectorization, pipeline isolation, EMBER rejection |
| `trainer` | `test_model_training.py` | 6 | LightGBM fit, early stopping, serialization, dim checks |
| `classifier` | `test_classifier.py` | 9 | Loading, single/batch inference, dimension checks, rejection |
| `evaluator` | `test_evaluator.py` | 3 | Metric math, FPR/FNR, feature importance, PNG generation |
| `explainability` | `test_explainability.py` | 24 | TreeExplainer, local/batch, additivity, ranking, direction, JSON |
| `calibration` | `test_calibration.py` | 14 | Sigmoid/isotonic fit, ECE/MCE, save/load, threshold analysis |
| `api` | `test_api.py` | 14 | FastAPI endpoints, schema validation, NaN/Inf rejection, batch limits |
| `dashboard` | `test_dashboard.py` | 27 | API client, vector validation, PE upload cleanup, Plotly charts, state |
| `stage7_integration` | `test_stage7_integration.py` | 12 | Acquisition check, Tar Slip mitigation, provenance propagation, leakage |
| **Total** | | **211** | **100% Passing** |

---

## 🛣️ Roadmap

| Stage | Description | Status |
|-------|-------------|--------|
| **Stage 1** | Project Foundation & Dataset Preparation | ✅ Complete |
| **Stage 2** | Static PE Feature Extraction & Model Adapter Interface | ✅ Complete |
| **Stage 3** | LightGBM Model Training & Evaluation Suite | ✅ Complete |
| **Stage 4** | Explainable AI Engine Using SHAP (Global & Local) | ✅ Complete |
| **Stage 5** | Probability Calibration & FastAPI Backend Microservice | ✅ Complete |
| **Stage 6** | Streamlit Security Analyst Dashboard / Frontend UI | ✅ Complete |
| **Stage 7** | Final Integration, Verified Dataset Tooling, Testing & Hardening | ✅ Complete |

---

## 📜 License

MIT License.
