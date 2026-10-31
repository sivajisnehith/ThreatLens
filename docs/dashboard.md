# ThreatLens — Streamlit Security Analyst Dashboard (Stage 6)

## 1. Overview

The **ThreatLens Security Analyst Console** is a production-grade, dark-themed SOC interface built with Streamlit and Plotly. It connects to the ThreatLens asynchronous FastAPI backend to deliver explainable static malware analysis, non-executing PE binary structural decomposition, calibrated probability scoring, additive SHAP attributions, and simulated operational threshold trade-off analysis.

---

## 2. Architecture & Design

The dashboard is structured into modular layers adhering to separation of concerns:

```
threatlens/dashboard/
├── __init__.py               # Exports ThreatLensAPIClient and custom exceptions
├── app.py                   # Main Streamlit application entrypoint & routing
├── api_client.py            # Resilient HTTP client for FastAPI endpoints
├── state.py                 # Safe session state lifecycle manager (no raw binary caching)
├── components/              # Reusable dark SOC UI widgets and Plotly charts
│   ├── __init__.py
│   ├── styles.py            # Dark SOC theme CSS, monospace styling, warning banners
│   ├── metrics.py           # Status cards, classification badges, metric rows
│   └── charts.py            # Dark Plotly chart builders (gauge, SHAP bar, trade-offs)
└── pages/                   # Dedicated page views
    ├── __init__.py
    ├── security_overview.py # Page 1: Health, status cards, test split metrics
    ├── analyze_sample.py    # Page 2: PE static extractor (85-dim) & EMBER demo (2,351-dim)
    ├── prediction_results.py# Page 3: Calibrated risk gauge, interactive threshold slider
    ├── local_shap.py        # Page 4: TreeSHAP local attribution bars & additivity
    ├── global_analysis.py   # Page 5: Macro feature importance, 8 schema groups, gain/split
    ├── threshold_analysis.py# Page 6: Simulated operating postures & sensitivity curves
    └── model_information.py # Page 7: Architecture specs, limitations, future roadmap
```

### Theme & Visual Direction
* **Color Palette**: Dark charcoal / navy canvas (`#0b0f19`, `#131b2e`, `#162036`).
* **Accents**: Electric cyan (`#38bdf8`) for controls and highlights.
* **Status Signaling**:
  - `Malicious`: High-contrast crimson (`#ef4444`, `rgba(239, 68, 68, 0.15)`)
  - `Benign`: Emerald green (`#10b981`, `rgba(16, 185, 129, 0.15)`)
  - `Experimental / Warning`: Amber (`#f59e0b`, `rgba(245, 158, 11, 0.12)`)
* **Typography**: Monospace font (`JetBrains Mono`, `SF Mono`, `Consolas`) for hashes, memory addresses, feature vectors, and numerical outputs.

---

## 3. Page Functional Reference

### Page 1 — Security Overview
* **Status Grid**: Real-time status cards displaying FastAPI service health, model load state, model release (`v1.0.0`), feature dimension (`2,351 EMBER`), and calibration method (`Platt Scaling / Sigmoid`).
* **Test Split Metrics**: Evaluated exclusively on untouched test data (`X_test`, `y_test`). Displays Accuracy, Precision, Recall, Specificity, F1-Score, FPR, and FNR.
* **Interactive Visualizations**:
  - Metric summary bar chart comparing primary performance indicators.
  - Styled $2 \times 2$ Confusion Matrix heatmap (TP, TN, FP, FN counts and cohort percentages).
* **Provenance Disclosure**: Highlights the `synthetic-ember-2018` dataset origin and explains that chance-level test metrics (~50%) reflect pipeline verification on synthetic distributions.

### Page 2 — Analyze Sample (Dual-Mode Interface)
* **Mode A: Custom PE Static Feature Extractor (85 Features)**:
  - Accepts `.exe`, `.dll`, `.sys` binaries ($\le 50\text{ MB}$).
  - Strictly non-executing static structural parsing via `pefile`.
  - Temporary files are immediately wiped upon extraction.
  - Displays file size, architecture (`PE32` vs `PE32+`), timestamp, section entropy table, and import statistics.
  - **Explicit Disclosure**: States that no compatible classifier exists for these 85 features. No prediction or malware score is computed.
  - Enables JSON download of the extracted 85-dimensional feature dictionary.
* **Mode B: EMBER 2018 Classifier Demonstration (2,351 Features)**:
  - Labeled `EMBER FEATURE VECTOR DEMO — NOT RAW PE CLASSIFICATION`.
  - Supports loading pre-extracted benchmark test vectors (`X_test[i]`), `.npy` files, or `.json` payloads.
  - Strictly validates vector shape (must equal 2,351), rejects non-numeric types, `NaN`, and infinite values.
  - Executes inference via `POST /predict` and caches the sample for cross-page explanation.

### Page 3 — Prediction Results
* **Calibrated Risk Gauge**: Plotly indicator displaying effective probability score ($0.00 \to 1.00$) with colored operational zones and a dynamic threshold pointer.
* **Classification Badge**: High-visibility status card declaring `🛡️ BENIGN` or `☣️ MALICIOUS`.
* **Instant Threshold Adjustment**: Interactive slider ($0.05 \le t \le 0.95$). Adjusting the slider **re-evaluates classification locally** using the cached probability without sending unnecessary HTTP requests to the backend.

### Page 4 — Local SHAP Explainability
* **FastAPI Integration**: Calls `POST /explain` to fetch TreeSHAP attributions and the baseline expected value $E[f(x)]$.
* **Plotly Diverging Bar Chart**:
  - Horizontal bar chart sorting features by absolute attribution magnitude $|\phi_i|$.
  - Positive attributions (red) indicate features pushing the prediction toward Malicious.
  - Negative attributions (cyan/blue) indicate features reducing the malware score.
  - Rich hover data reveals canonical feature name, EMBER group, observed value, SHAP contribution, and description.
* **Additivity & Top-K**: Displays exact additivity verification ($|E[f(x)] + \sum \phi_i - f(x)| < 10^{-6}$) and allows configuring $5 \le K \le 40$.
* **Mandatory Caveat**: Explains that SHAP explains internal model logic rather than physical ground truth.

### Page 5 — Global Feature Analysis
* **Macro Explainability**: Visualizes cohort-wide feature importance across the evaluation set using `reports/shap_global_importance.json`.
* **Top 20 Features**: Horizontal Plotly bar chart displaying the 20 most impactful features across the model.
* **Schema Group Attribution**: Aggregated attribution across the 8 EMBER schema groups (`histogram`, `byteentropy`, `strings`, `general`, `header`, `section`, `imports`, `exports`).
* **Metric Distinctions**: Detailed side-by-side comparison of **SHAP Global Importance** vs **LightGBM Gain** vs **LightGBM Split Count**, emphasizing why these metrics are not interchangeable.

### Page 6 — Simulated Threshold Analysis
* **Operational Trade-Off Curves**: Interactive visualization of TPR (Recall), FPR (Alert Fatigue), FNR (Missed Malware), and F1-Score across thresholds $0.05 \le t \le 0.95$.
* **Posture Classification**:
  - **High Security ($t \le 0.45$)**: Low FNR ($\le 15\%$), high TPR, higher alert triage requirements.
  - **Balanced ($t \approx 0.50$)**: Optimal F1 trade-off.
  - **Low Interruption ($t \ge 0.55$)**: Low FPR, suppressed alert fatigue, increased false negatives.
* **Clear Notice**: Labeled `SIMULATED THRESHOLD ANALYSIS` to ensure synthetic trade-offs are not misconstrued as live SOC guidance.

### Page 7 — Model Information & Architecture
* **Technical Metadata**: Reflects responses from `GET /model/info`, including boosting type (`GBDT`), tree depth, learning rate, and feature counts.
* **Calibration Parameters**: Platt scaling logistic coefficients ($A = 0.5898, B = -0.2974$), Brier score improvement, and Expected Calibration Error.
* **Documented Limitations**: Transparent breakdown of synthetic training, chance-level discrimination, hashing collisions in EMBER, and pipeline incompatibilities.
* **Development Roadmap**: Planned steps for authentic EMBER ingestion, native EMBER extraction, conformal prediction, and API security.

---

## 4. FastAPI Client (`ThreatLensAPIClient`)

[`ThreatLensAPIClient`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/dashboard/api_client.py) communicates with the backend:

| Method | Target Endpoint | Description |
| :--- | :--- | :--- |
| `health()` | `GET /health` | Validates backend readiness and model status |
| `get_model_info()` | `GET /model/info` | Fetches model architecture, calibration status, and limits |
| `get_model_metrics()` | `GET /metrics/model` | Loads untouched test split performance metrics |
| `predict(features, threshold)` | `POST /predict` | Runs single-vector classification (2,351 dims) |
| `predict_batch(samples, threshold)` | `POST /predict/batch` | Runs batch classification ($N \le 100$) |
| `explain(features, top_k, threshold)` | `POST /explain` | Requests local SHAP attributions and baseline |

### Resilience & Error Handling:
* **Timeouts**: Enforces 3.0-second connection timeout and 10.0-second read/write timeout.
* **Custom Exceptions**: Wraps raw network errors into `APIConnectionError`, `APITimeoutError`, and `APIResponseError`.
* **Zero Binary Caching**: Uploaded PE files and raw vectors are never retained in persistent caches.

---

## 5. Security & Operational Hardening

1. **Non-Executing Static Analysis**: Uploaded PE files are passed strictly to `pefile` and are never invoked, executed, or loaded into executable memory.
2. **Immediate Temporary File Cleanup**: Temporary binary buffers created during extraction are deleted in a `try...finally` block.
3. **No Binary Content in Logs**: Application logs never write binary payloads, headers, or raw executable content.
4. **Strict Upload Sanitization**: Enforces file extension limits (`.exe`, `.dll`, `.sys`) and a 50 MB size ceiling.
5. **No Hardcoded Secrets**: Backend URLs are configurable via the `THREATLENS_API_URL` environment variable with a localhost default.

---

## 6. Launch Instructions

### 1. Start the FastAPI Backend:
```bash
# Start FastAPI service on port 8000
uvicorn threatlens.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Launch the Streamlit Dashboard:
```bash
# In a separate terminal session
streamlit run threatlens/dashboard/app.py --server.port 8501
```

Once running, navigate to `http://localhost:8501` in your browser.
