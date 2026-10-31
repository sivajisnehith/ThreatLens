# ThreatLens Explainability Engine (SHAP)

## 1. Overview & Theoretical Foundations

ThreatLens integrates **SHAP (SHapley Additive exPlanations)** to provide transparent, mathematically grounded interpretations of static malware detection decisions. The system operates on the **2,351-dimensional EMBER 2018** feature representation, explaining predictions from the gradient-boosted decision tree ensemble trained with LightGBM.

### What is SHAP?

SHAP grounds machine learning explanation in cooperative game theory (Lloyd Shapley, 1953). In the ThreatLens context:
* The **players** in the cooperative game are the 2,351 static features extracted from a PE binary.
* The **payout** or coalition value function is the model output margin (log-odds of being malicious).
* The **Shapley value** ($\phi_i$) assigned to feature $i$ represents its average marginal contribution across all possible coalitions of features $S \subseteq F \setminus \{i\}$:

$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$

SHAP is the unique attribution method that simultaneously satisfies four fundamental axiomatic properties:
1. **Efficiency (Local Accuracy / Additivity):** The sum of feature contributions plus the baseline expected value exactly equals the model output:
   $$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$
   where $\phi_0 = E[f(x)]$ is the expected model output across the training population.
2. **Symmetry:** If two features contribute equally to all possible feature subsets, their attributions are identical.
3. **Dummy Player (Null Effect):** A feature that does not change the model prediction when added to any coalition receives $\phi_i = 0$.
4. **Additivity (Linearity):** When combining independent model ensembles, the combined attribution is the sum of the individual attributions.

---

## 2. How TreeExplainer Works

Computing exact Shapley values naively requires exponential time ($O(M \cdot 2^{|F|})$), which is intractable for 2,351 features ($2^{2351}$).

ThreatLens leverages **SHAP TreeExplainer** (Lundberg et al., *Nature Machine Intelligence*, 2020), which exploits the tree structure of decision trees to evaluate conditional expectations in polynomial time:

$$O(T \cdot L \cdot D^2)$$

where:
* $T$ is the number of boosting iterations (trees),
* $L$ is the maximum number of leaves per tree (default: 31),
* $D$ is the maximum tree depth.

### Model Output Margin vs. Probability

In binary classification with LightGBM, predictions are made on the raw margin (log-odds):

$$\text{Margin } f(x) = \log\left(\frac{P(\text{Malicious})}{1 - P(\text{Malicious})}\right)$$

The probability is recovered via the standard logistic sigmoid:

$$P(\text{Malicious}) = \sigma(f(x)) = \frac{1}{1 + e^{-f(x)}}$$

> [!IMPORTANT]
> **Exact additivity holds on the log-odds margin ($f(x)$).**  
> ThreatLens evaluates TreeExplainer with `model_output="raw"`. The sum of the baseline expected value $\phi_0$ and all 2,351 SHAP values $\sum \phi_i$ matches the raw margin score to machine precision ($< 10^{-6}$ error). Because the sigmoid transformation is strictly monotonic, positive log-odds attributions directly correspond to increased malicious probability.

---

## 3. Global versus Local Explanations

| Characteristic | Local Explanations (`explain_single`) | Global Explanations (`calculate_global_importance`) |
| :--- | :--- | :--- |
| **Scope** | One individual PE binary prediction | Representative dataset population ($N$ binaries) |
| **Question Answered** | *"Why was this specific sample classified as Malicious with 94.2% probability?"* | *"Which features does the model rely upon overall across benign and malicious software?"* |
| **Metric** | Signed attribution vector $\phi(x) \in \mathbb{R}^{2351}$ | Mean absolute attribution $E[\|\phi_i\|] = \frac{1}{N} \sum_{j=1}^N \|\phi_{ij}\|$ |
| **Output Format** | Structured `SingleExplanation` JSON, top pushing/pulling features | `GlobalImportanceResult` JSON, beeswarm summary, global bar chart, group plot |
| **Use Case** | Incident response, SOC triage, analyst verification | Model validation, feature pruning, bias detection, compliance reporting |

---

## 4. Directional Influence in Binary Classification

In ThreatLens, the target variable is encoded as:
* **Class 0:** `BENIGN`
* **Class 1:** `MALICIOUS`

Therefore, the sign of the SHAP attribution value ($\phi_i$) has a rigorous, unambiguous meaning:

* **Positive Contribution ($\phi_i > 0$):**
  * Pushes the model score toward the positive class (Class 1).
  * **Interpretation:** `INCREASES_MALWARE_RISK`.
  * The observed feature value provides evidence supporting a malware classification.

* **Negative Contribution ($\phi_i < 0$):**
  * Pulls the model score toward the negative class (Class 0).
  * **Interpretation:** `DECREASES_MALWARE_RISK`.
  * The observed feature value provides evidence supporting a benign classification.

* **Neutral Contribution ($\phi_i \approx 0$):**
  * The feature was not traversed in the decision trees or had zero marginal impact on this binary.
  * **Interpretation:** `NEUTRAL`.

---

## 5. EMBER Feature Group Aggregation

The 2,351 EMBER features are grouped into 8 official schema groups. ThreatLens aggregates attributions at both the individual feature level and the group level:

1. **Byte Histogram (Indices 0–255, 256 features):** Normalized frequencies of raw byte values $0\times00$ to $0\times\text{FF}$.
2. **Byte-Entropy Histogram (Indices 256–511, 256 features):** 2D joint distribution of byte values and localized sliding-window Shannon entropy.
3. **Strings (Indices 512–767, 256 features):** Printable ASCII string statistics (count, average length, entropy, paths, URLs, registry keys, embedded MZ markers, and character distributions).
4. **General File Info (Indices 768–777, 10 features):** File size, virtual image size, debug presence, export count, import count, relocations, resources, digital signature, TLS, and symbols.
5. **Header Features (Indices 778–838, 61 features):** COFF header attributes (timestamp, machine, sections) and Optional header fields (entry point, image base, subsystem, linker versions).
6. **Section Characteristics (Indices 839–1034, 196 features):** Section sizes, memory entropy, access permissions, and hashed section names.
7. **Imports (Indices 1035–2282, 1,248 features):** 2D MurmurHash projection mapping imported DLL names and API function symbols.
8. **Exports (Indices 2283–2350, 68 features):** Feature-hashed export symbols via MurmurHash.

---

## 6. Metric Distinctions: SHAP vs. Gain vs. Split Importance

ThreatLens clearly distinguishes three metrics that must **not** be used interchangeably:

```
+---------------------------------------------------------------------------------------------------------+
|                                  FEATURE IMPORTANCE METRICS COMPARISON                                  |
+----------------------+--------------------------------------------------+-------------------------------+
| Metric               | What It Measures                                 | Evaluation Context            |
+----------------------+--------------------------------------------------+-------------------------------+
| SHAP Global          | Mean absolute Shapley attribution (E[|phi_i|])   | Test-time population impact   |
| (E[|phi_i|])         | on model margin across evaluation samples.       | on actual predictions.        |
+----------------------+--------------------------------------------------+-------------------------------+
| LightGBM Gain        | Total reduction in training loss (binary         | Training-time objective       |
|                      | logloss) contributed by splits on this feature.  | optimization efficiency.      |
+----------------------+--------------------------------------------------+-------------------------------+
| LightGBM Split       | Raw count of times a feature was selected to     | Structural tree topology      |
|                      | split nodes across all trees.                    | frequency.                    |
+----------------------+--------------------------------------------------+-------------------------------+
```

> [!WARNING]
> A feature can have a high split count but low gain if it makes many superficial splits that barely reduce loss. Conversely, a feature may appear in only one or two splits early in the trees with massive gain and decisive test-time SHAP impact. Always report SHAP attributions when explaining model behavior on data cohorts.

---

## 7. Critical Limitations & Governance

### 1. Feature Hashing Non-Invertibility (No Fabricated Semantics)
EMBER hashes library names and API functions into 1,248 import bins and 68 export bins using MurmurHash.
* **Limitation:** MurmurHash is a one-way hashing function with inherent bucket collisions. Multiple different APIs map to the exact same feature index.
* **Policy:** ThreatLens strictly refrains from inventing specific API names (e.g., claiming bin 1100 is "CreateRemoteThread"). The explanation explicitly identifies it as a *feature-hashed import representation* subject to collision limitations.

### 2. Feature Correlation & Multicollinearity
In static PE analysis, features often correlate strongly (e.g., file size and virtual size, or byte entropy and packed section flags).
* **Limitation:** Tree ensembles can split on either of two correlated features arbitrarily. When correlated features exist, TreeExplainer divides marginal credit along the observed tree paths, which may dilute individual feature ranking.

### 3. Importance Does Not Establish Causality
* **Limitation:** SHAP values identify associative patterns utilized by the decision trees, **not real-world causal mechanisms**.
* **Example:** If `ember_general_has_signature` receives a high negative SHAP value, it does not mean that stripping a signature *causes* a file to turn into malware. Malware analysts must treat SHAP attributions as triage pointers, not causal proofs.

### 4. Synthetic Training Data Limitation (Stage 3 Artifact)
* **Dataset Notice:** The current Stage 3 model artifact (`models/lightgbm_ember.txt`) was trained on a synthetic data distribution for infrastructure validation because the full EMBER archive (1.7 GB) was truncated locally.
* **Disclaimer:** Attributions generated from this model reflect the synthetic statistical generator. They **must not** be cited as real-world malware discoveries or threat intelligence findings. The explainability engine is fully model-agnostic and will seamlessly explain real malware once retrained on full EMBER data.

---

## 8. CLI and Programmatic Usage

### Global SHAP Report Generation
```bash
python scripts/explain.py --mode global --data-path data/processed/X_test.npy --output-dir reports
```
Generated artifacts:
* `reports/shap_global_importance.json`
* `reports/shap_summary.png`
* `reports/shap_bar.png`
* `reports/shap_group_importance.png`

### Local Prediction Explanation
```bash
# Explain sample 0 from test split
python scripts/explain.py --mode local --sample-index 0 --output reports/sample_0_explanation.json

# Explain external numpy feature vector
python scripts/explain.py --mode local --input-path path/to/sample.npy
```

### Python API Integration
```python
import numpy as np
from threatlens.explainability import ShapExplainer

# Initialize explainer
explainer = ShapExplainer()

# Explain single 2,351-dimensional feature vector
sample = np.load("data/processed/X_test.npy")[0]
explanation = explainer.explain_single(sample, top_k=20)

print(f"Prediction: {explanation.prediction_label} (p = {explanation.probability:.4f})")
print(f"Additivity Error: {explanation.additivity_error:.2e}")

for contrib in explanation.top_positive_contributors[:5]:
    print(f"[+{contrib.shap_value:.4f}] {contrib.feature_name} ({contrib.feature_group}): {contrib.feature_description}")
```
