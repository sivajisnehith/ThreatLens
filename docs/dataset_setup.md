# ThreatLens — Genuine EMBER 2018 Dataset Setup & Validation Guide

**Document Version:** 1.0.0 (Stage 7)  
**Dataset Reference:** Elastic Malware Benchmark for Empowering Researchers (EMBER 2018 v2)  
**Official Repository:** [https://github.com/elastic/ember](https://github.com/elastic/ember)  

---

## 1. Dataset Specifications & Provenance

The official **EMBER 2018** dataset contains benchmark static feature representations extracted from 1,100,000 Windows PE binaries (900,000 training, 200,000 test), categorized as:
* **Benign**: Label `0` (clean software samples)
* **Malicious**: Label `1` (confirmed malware samples)
* **Unlabeled**: Label `-1` (excluded from supervised classification)

### Cryptographic & Storage Specifications:

| Property | Value |
| :--- | :--- |
| **Official Distribution URL** | `https://ember.elastic.co/ember_dataset_2018_2.tar.bz2` |
| **Official Archive SHA-256** | `b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812` |
| **Exact Archive File Size** | `1,696,539,273` bytes (~1.58 GiB / ~1.70 GB) |
| **Archive Format** | Bzip2-compressed POSIX Tar (`tar.bz2`) |
| **Feature Dimensionality** | Exactly 2,351 numerical dimensions |
| **Schema Version** | EMBER 2018 (v2 format) |

---

## 2. Truncated Archive Audit & Detection

> [!WARNING]
> Prior to Stage 7, an incomplete download existed at `data/raw/ember_dataset_2018_2.tar.bz2` with:
> - **Actual Size on Disk**: `1,572,003,840` bytes (1,499.18 MB)
> - **Computed SHA-256**: `fb2213e9aeb8a18b8bc47d3f4b6d9dd67ed8b897548aa6c65e500f996857851e`
> 
> Because this file is truncated at 92.7% of its total byte stream, it cannot be unpacked without corrupting JSONL feature records. ThreatLens **strictly rejects** this partial file and marks genuine retraining as **BLOCKED** until full cryptographic verification succeeds.

### Check Status via CLI:
```bash
python -m threatlens.data.download_ember --check
```

Output:
```text
Archive Path:  C:\Users\ALPHA_JUNIOR\MalExplain\data\raw\ember_dataset_2018_2.tar.bz2
File Exists:   True
Size on Disk:  1,572,003,840 bytes (1499.18 MB)
Expected Size: 1,696,539,273 bytes (1617.95 MB)
Truncated:     True
Status:        TRUNCATED_INCOMPLETE
SHA-256:       fb2213e9aeb8a18b8bc47d3f4b6d9dd67ed8b897548aa6c65e500f996857851e
Expected Hash: b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812
Valid:         False
Message:       Archive is truncated: 1,572,003,840 of 1,696,539,273 bytes (92.7%). Checksum does not match.
```

---

## 3. Resumable Acquisition Protocol

To prevent wasted bandwidth and support interrupted connections, [`EmberDatasetAcquisition`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/data/download_ember.py) implements HTTP `Range` requests:

1. **Explicit User Confirmation**: The utility refuses to initiate large downloads without the `--confirm` flag, preventing automated CI/CD runners from exhausting network quotas.
2. **Resumption**: Detects the existing byte offset on disk and requests only the remaining 124.5 MB.
3. **Cryptographic Check**: Once transfer completes, computes SHA-256 over 1 MB buffers and verifies exact equality with `b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812`.
4. **Path Traversal Protection**: During extraction, every archive member destination path is validated to prevent Zip Slip / Tar Slip vulnerabilities.

### Download Command:
```bash
python -m threatlens.data.download_ember --download --confirm
```

### Extraction Command:
```bash
python -m threatlens.data.download_ember --extract
```

---

## 4. Dataset Validation & Leakage Prevention Pipeline

Once extracted, files are processed through the strict Stage 1–3 pipeline:

```
data/raw/
├── train_features.jsonl  (900k samples)
├── train_labels.csv
├── test_features.jsonl   (200k samples)
└── test_labels.csv
```

### Split Isolation Protocol:
* **Train Set (60%)**: Used solely for fitting imputer, scaler, and LightGBM tree structures.
* **Validation Set (15%)**: Monitored strictly for early stopping and hyperparameter tuning.
* **Calibration Set (15%)**: Kept isolated for Platt scaling and isotonic regression.
* **Test Set (10%)**: Held out untouched until final reporting.

### Dataset Preparation Command:
```bash
python scripts/prepare_dataset.py --verbose
```
