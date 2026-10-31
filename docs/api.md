# ThreatLens REST API Documentation (Stage 5)

## 1. Architecture Overview

ThreatLens exposes a production-ready, modular **FastAPI** backend for explainable static malware detection and analysis. The service connects the 2,351-dimensional EMBER 2018 LightGBM classification engine with post-hoc probability calibration and SHAP TreeExplainer attributions.

### Service Architecture

```
                          HTTP Clients / SOC Consoles / CI/CD
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │      FastAPI Router Gateway     │
                         │    (CORS, Validation, Errors)   │
                         └────────────────┬────────────────┘
                                          │
            ┌─────────────────────────────┼────────────────────────────┐
            ▼                             ▼                            ▼
   ┌─────────────────┐           ┌─────────────────┐          ┌─────────────────┐
   │ /health, /info  │           │     /predict    │          │     /explain    │
   │    & /metrics   │           │   (Single/Batch)│          │  (Local SHAP)   │
   └────────┬────────┘           └────────┬────────┘          └────────┬────────┘
            │                             │                            │
            │                             ▼                            ▼
            │                    ┌─────────────────┐          ┌─────────────────┐
            │                    │MalwareClassifier│          │  ShapExplainer  │
            │                    │  (LightGBM GBDT)│          │  (TreeExplainer)│
            │                    └────────┬────────┘          └────────┬────────┘
            │                             │                            │
            ▼                             ▼                            ▼
   ┌─────────────────┐           ┌─────────────────┐          ┌─────────────────┐
   │ Evaluation &    │           │   Probability   │          │     Feature     │
   │ Model Metadata  │           │   Calibrator    │          │   Interpreter   │
   │ (Stage 3 JSON)  │           │ (Platt Scaling) │          │ (Canonical Spec)│
   └─────────────────┘           └─────────────────┘          └─────────────────┘
```

---

## 2. API Endpoints

### Summary of Endpoints

| Method | Endpoint | Description | Auth (Future) |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service readiness, model loading state, and calibration status | Public |
| `GET` | `/model/info` | Architecture, training hyperparameters, and feature schema | Public |
| `GET` | `/metrics/model` | Test set evaluation metrics from Stage 3 evaluation report | Public |
| `POST` | `/predict` | Single binary prediction with raw & calibrated probabilities | Token |
| `POST` | `/predict/batch` | High-throughput batch prediction (up to 100 samples) | Token |
| `POST` | `/explain` | Local SHAP attribution breakdown for an individual prediction | Token |

---

### Detailed Endpoint Specifications

#### 1. `GET /health`
Returns service operational health, model loading status, and active calibration method.

**Response Example (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "3.0.0",
  "model_status": "SYNTHETIC — RESEARCH ONLY",
  "schema_version": "1.0.0",
  "calibration_status": "calibrated",
  "calibration_method": "sigmoid",
  "disclaimer": "DISCLAIMER: Explanations are produced from a model trained on synthetic data for validation purposes..."
}
```

---

#### 2. `GET /model/info`
Returns comprehensive architecture specifications, training hyperparameters, and known limitations.

**Response Example (200 OK):**
```json
{
  "algorithm": "LightGBM Gradient Boosted Decision Trees (GBDT)",
  "feature_dimension": 2351,
  "feature_schema": "EMBER_2018",
  "dataset_provenance": "Stage 3 LightGBM binary classifier trained on Stage 1 EMBER 2018 synthetic schema dataset.",
  "model_status": "SYNTHETIC — RESEARCH ONLY",
  "model_version": "3.0.0",
  "training_metadata": {
    "hyperparameters": {
      "boosting_type": "gbdt",
      "objective": "binary",
      "learning_rate": 0.05,
      "num_leaves": 31
    }
  },
  "calibration_info": {
    "is_calibrated": true,
    "method": "sigmoid",
    "status": "calibrated"
  },
  "limitations": [
    "SYNTHETIC DATA: The current model was trained on synthetic data for pipeline validation.",
    "FEATURE HASHING: EMBER uses MurmurHash for imports and exports. Hashed bins cannot be uniquely mapped back to raw API names.",
    "NON-CAUSALITY: Feature importance indicates tree split usage, not causal malware mechanisms."
  ]
}
```

---

#### 3. `POST /predict`
Executes validated binary inference on a single 2,351-dimensional EMBER feature vector.

**Request Payload:**
```json
{
  "features": [0.0, 0.012, -0.045, "... 2,351 float values ..."],
  "threshold": 0.50
}
```

**Response Example (200 OK):**
```json
{
  "prediction": 1,
  "prediction_label": "MALICIOUS",
  "raw_probability": 0.5296,
  "calibrated_probability": 0.5312,
  "threshold": 0.50,
  "confidence": 0.5296,
  "model_version": "3.0.0",
  "warning": "EXPERIMENTAL: Probability calibrator was fitted on synthetic data distributions..."
}
```

---

#### 4. `POST /predict/batch`
Performs batch classification for up to 100 samples in a single request.

**Request Payload:**
```json
{
  "samples": [
    [0.0, "... 2351 floats ..."],
    [0.1, "... 2351 floats ..."]
  ],
  "threshold": 0.50
}
```

**Response Example (200 OK):**
```json
{
  "count": 2,
  "predictions": [0, 1],
  "prediction_labels": ["BENIGN", "MALICIOUS"],
  "raw_probabilities": [0.4689, 0.5296],
  "calibrated_probabilities": [0.5162, 0.5312],
  "threshold": 0.50,
  "model_version": "3.0.0",
  "warning": "EXPERIMENTAL: Model outputs reflect synthetic distributions..."
}
```

---

#### 5. `POST /explain`
Calculates exact SHAP values for an individual sample using `shap.TreeExplainer`.

**Request Payload:**
```json
{
  "features": [0.0, "... 2351 floats ..."],
  "top_k": 5,
  "threshold": 0.50
}
```

**Response Example (200 OK):**
```json
{
  "prediction": 1,
  "prediction_label": "MALICIOUS",
  "raw_score": 0.118611,
  "raw_probability": 0.5296,
  "calibrated_probability": 0.5312,
  "baseline": -0.001288,
  "threshold": 0.50,
  "additivity_error": 7.81e-09,
  "top_positive_contributors": [
    {
      "rank": 1,
      "feature_name": "ember_byteentropy_317",
      "feature_group": "byteentropy",
      "shap_value": 0.081177,
      "direction": "INCREASES_MALWARE_RISK",
      "feature_description": "2D Byte-entropy histogram joint distribution bin 61."
    }
  ],
  "top_negative_contributors": [
    {
      "rank": 3,
      "feature_name": "ember_header_789",
      "feature_group": "header",
      "shap_value": -0.030530,
      "direction": "DECREASES_MALWARE_RISK",
      "feature_description": "COFF or Optional PE header field (offset 11)."
    }
  ],
  "warning": "DISCLAIMER: Explanations are produced from a model trained on synthetic data..."
}
```

---

#### 6. `GET /metrics/model`
Returns test evaluation metrics from the Stage 3 evaluation report without evaluating calibration data.

**Response Example (200 OK):**
```json
{
  "threatlens_stage": 3,
  "metrics": {
    "accuracy": 0.5042,
    "precision": 0.5041,
    "recall": 0.5167,
    "specificity": 0.4917,
    "f1": 0.5103,
    "roc_auc": 0.4703,
    "pr_auc": 0.4812,
    "false_positive_rate": 0.5083,
    "false_negative_rate": 0.4833
  },
  "confusion_matrix": {
    "tn": 59,
    "fp": 61,
    "fn": 58,
    "tp": 62
  },
  "dataset_type": "synthetic_ember_test_split",
  "disclaimer": "DISCLAIMER: Model was evaluated on synthetic test split."
}
```

---

## 3. Launching the API Server

### Development Server (Uvicorn):
```bash
uvicorn threatlens.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Production Execution:
```bash
python -m threatlens.api.main
```

### Interactive Documentation & Testing:
* **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
* **OpenAPI Schema:** [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

---

## 4. Probability Calibration Methodology

LightGBM outputs uncalibrated probabilities via its logistic objective. In moderate sample regimes, tree probabilities can be overconfident or poorly calibrated.

ThreatLens provides post-hoc probability calibration evaluated strictly on the **dedicated calibration split** (`X_cal.npy`, `y_cal.npy`, $N=360$):

1. **Platt Scaling (Sigmoid Calibration — Default):**
   Fits a scalar logistic regression over uncalibrated probabilities:
   $$\hat{P}_{\text{cal}}(y=1 | s) = \frac{1}{1 + \exp(A \cdot s + B)}$$
   * **Advantage:** Has only 2 parameters ($A, B$). Highly regularized, preventing overfitting on small-to-moderate sample cohorts ($N=360$).
2. **Isotonic Regression:**
   Fits a non-parametric, piece-wise constant non-decreasing step function.
   * **Limitation:** Can overfit or create flat prediction ties when sample sizes are small ($N < 1000$).
3. **Evaluation Metrics:**
   * **Brier Score:** Mean squared probability error.
   * **Log Loss:** Binary cross-entropy.
   * **Expected Calibration Error (ECE):** Bin-weighted difference between accuracy and confidence across 10 equal-width bins.
   * **Maximum Calibration Error (MCE):** Maximum absolute calibration discrepancy across any bin.

---

## 5. Decision Thresholds & Operational Trade-Offs

Malware detection requires balancing operational disruption against catastrophic breach risk:

| Posture | Threshold ($\tau$) | Target Objective | Best Suited For |
| :--- | :---: | :--- | :--- |
| **High Security (Low FNR)** | `0.40 - 0.45` | Minimize False Negatives ($\text{FNR} \to 0$) | High-assurance gateways, air-gapped networks, critical servers. |
| **Balanced (Max F1)** | `0.50` | Harmonize Precision and Recall | Standard endpoint detection, SOC automated triage queues. |
| **Low Interruption (Low FPR)** | `0.55 - 0.60` | Minimize False Positives ($\text{FPR} \to 0$) | Developer workstations, business-critical servers, automated quarantine. |

> [!IMPORTANT]
> The threshold $\tau$ is fully configurable via the `threshold` field in API requests.

---

## 6. Security Considerations & Hardening

1. **Strict Input Validation:**
   * Every feature vector must contain **exactly 2,351 numeric values**.
   * Feature vectors containing `NaN` or `Infinity` are rejected immediately with HTTP 422.
2. **Batch Size Enforcing:**
   * Batch inference requests are capped at 100 samples to prevent resource exhaustion or Denial of Service (DoS).
3. **Information Disclosure Prevention:**
   * Internal exception stack traces are withheld from HTTP responses. Internal exceptions are logged securely on the server with structured logging.
4. **CORS Governance:**
   * CORS origins are restricted and configurable via the `ALLOWED_ORIGINS` environment variable.
5. **No File Execution:**
   * ThreatLens static analysis extracts structural PE metadata without executing binary code.
6. **Future Authentication & Rate Limiting:**
   * In public production environments, attach an API Gateway or FastAPI middleware providing JWT/OAuth2 Bearer token verification and token-bucket rate limiting (e.g., via `slowapi` or Redis).

---

## 7. Dataset Provenance & Limitations

> [!WARNING]
> **Experimental Synthetic Data Notice:**  
> The active model artifact (`models/lightgbm_ember.txt`) and calibrator (`models/calibrator.joblib`) were trained on a synthetic data distribution for infrastructure validation.  
> Predictions, probabilities, and SHAP attributions demonstrate technical pipeline correctness and **must not be relied upon as verified real-world threat intelligence**.
