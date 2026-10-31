# ThreatLens — Comprehensive Security Audit & Production Hardening Report

**Date:** 2026-10-31  
**Auditor:** Senior Machine Learning & Cybersecurity Security Architect  
**Scope:** ThreatLens FastAPI Microservice, Streamlit Dashboard, PE Static Parsers, and Machine Learning Inference Pipeline  

---

## 1. Executive Summary

As a static malware detection system, ThreatLens processes untrusted, potentially malicious Windows Portable Executable (PE) binaries. A focused security audit was performed covering:
- Binary file ingestion and parse-time isolation
- Denial of Service (DoS) and Zip/Tar Slip vulnerabilities
- Input sanitization and memory limits
- Information disclosure and internal path leakage
- Cross-Origin Resource Sharing (CORS) and secrets management

All identified vulnerabilities have been hardened with targeted, lightweight mitigations that do not introduce heavy infrastructure (such as Redis or Kubernetes) while maintaining strict local and intranet operational security.

---

## 2. Threat Modeling & Vulnerability Assessment

### 2.1 Untrusted Binary Execution Risk
* **Threat**: Malicious binaries uploaded for inspection could potentially execute code on the analysis server.
* **Mitigation Verified**:
  - The extraction pipeline uses strictly non-executing static structural parsing via [`pefile`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/core/extractor.py#L90-L140).
  - Binaries are never invoked, loaded via Windows `LoadLibraryEx` with execution flags, or submitted to dynamic execution harnesses.
  - Parsing occurs within user-space memory without elevated administrative privileges.

### 2.2 Malicious PE Header Traversal & Buffer Overflows
* **Threat**: Specially crafted, corrupted, or malformed PE headers designed to trigger buffer overflows, integer wrap-arounds, or infinite loops in PE parsers.
* **Hardening Implemented**:
  - Pre-validation of DOS header magic bytes (`b"MZ"` / `0x5A4D`) and PE signature offset (`e_lfanew` pointing to `b"PE\0\0"`).
  - Explicit file size boundary checks ($\le 50\text{ MB}$).
  - All parsing logic is wrapped inside explicit `pefile.PEFormatError` and [`PEParseError`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/core/extractor.py#L28-L32) exception handlers, preventing unhandled runtime crashes.

### 2.3 Archive Traversal (Tar Slip / Zip Slip)
* **Threat**: Unpacking untrusted tar/zip archives could extract files outside the intended target directory using relative paths (e.g. `../../etc/passwd`).
* **Hardening Implemented**:
  - In [`threatlens/data/download_ember.py`](file:///C:/Users/ALPHA_JUNIOR/MalExplain/threatlens/data/download_ember.py#L205-L225), `safe_extract()` resolves every member path against the destination directory and raises `EmberAcquisitionError` if any path escapes the root directory.

### 2.4 Denial of Service via Resource Exhaustion
* **Threat**: Excessive batch sizes or oversized JSON vectors could exhaust server RAM.
* **Hardening Implemented**:
  - Maximum upload size capped at **50 MB** (`DASHBOARD_CONFIG.MAX_UPLOAD_SIZE_MB`).
  - Batch inference limit strictly enforced at **100 samples** (`APIConfig.MAX_BATCH_SIZE`).
  - Strict 2,351-dimensional float vector validation with rejection of `NaN` and `Inf` via Pydantic validators.

### 2.5 Information Disclosure & Error Sanitization
* **Threat**: Stack traces returned to API clients could disclose internal file system paths, virtual environment layouts, or third-party dependency versions.
* **Hardening Implemented**:
  - Custom FastAPI global exception handler catches uncaught exceptions and returns standardized JSON `{"error": "Internal Server Error", "detail": "...", "status_code": 500}` without raw Python tracebacks.
  - Binary file contents are never written to disk logs or echoed back in response payloads.

---

## 3. Security Checklist Summary

| Security Boundary | Status | Implemented Mechanism |
| :--- | :---: | :--- |
| **Static-Only Parsing** | ✅ Verified | Only static parsing (`pefile`); no execution or dynamic linking. |
| **Pre-Parse Magic Check** | ✅ Hardened | Verifies `b"MZ"` header magic before allocating parse memory. |
| **File Size Ceiling** | ✅ Hardened | 50 MB hard ceiling enforced at API and Streamlit upload widgets. |
| **Path Traversal Guards** | ✅ Hardened | Destination path normalization in tar extractors and temp files. |
| **Immediate Temp Cleanup** | ✅ Verified | Files cleaned in `try...finally` blocks; zero persistent binary retention. |
| **Input Sanitization** | ✅ Verified | 2,351 dimension check, finite float check (`NaN`/`Inf` rejected). |
| **Batch Size Cap** | ✅ Verified | Maximum 100 samples per request enforced with HTTP 422. |
| **No Hardcoded Secrets** | ✅ Verified | URLs and settings driven by `THREATLENS_API_URL` and config dataclasses. |
| **Safe Error Handling** | ✅ Hardened | Stack traces suppressed in external HTTP responses. |
| **CORS Policy** | ✅ Configurable | Controlled via `APIConfig.CORS_ORIGINS`. |

---

## 4. Operational Recommendations for Live Production

1. **Network Demarcation**: If deployed beyond localhost, host FastAPI behind a reverse proxy (e.g. Nginx or Cloudflare) with TLS 1.3 termination.
2. **API Authentication**: Integrate API Key headers (`X-API-Key`) or OAuth2 bearer tokens before exposing endpoints to multi-tenant environments.
3. **Container Sandboxing**: Run the backend in a non-root, read-only Docker container with seccomp profile restrictions.
