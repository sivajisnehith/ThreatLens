"""
ThreatLens Dashboard Page 2 — Analyze Sample

Provides two explicitly isolated analysis workflows:
1. Mode A: Custom PE Static Feature Extraction (85-dim static structural parsing via pefile).
   NO classification or malware score is computed (no compatible model exists).
2. Mode B: EMBER 2018 Classifier Demonstration (2,351-dim pre-extracted feature vector inference).
   Strict validation (2,351 dim, finite floats), calling FastAPI /predict endpoint.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from threatlens.config import DASHBOARD_CONFIG
from threatlens.core.extractor import extract_pe_features
from threatlens.dashboard.components.metrics import render_classification_card
from threatlens.dashboard.components.styles import (
    render_disclaimer_card,
    render_experimental_warning,
    render_header,
)
from threatlens.dashboard.state import (
    get_api_client,
    set_active_vector,
    set_custom_pe_result,
    set_prediction,
)


def _get_test_split_size() -> int:
    """Return total number of samples in X_test.npy."""
    test_path = Path("data/processed/X_test.npy")
    if test_path.exists():
        try:
            arr = np.load(test_path, mmap_mode="r")
            return len(arr)
        except Exception:
            pass
    return 2500


def _load_sample_from_test_split(index: int) -> Optional[np.ndarray]:
    """Load a sample vector from data/processed/X_test.npy if available."""
    test_path = Path("data/processed/X_test.npy")
    if test_path.exists():
        try:
            arr = np.load(test_path, mmap_mode="r")
            if 0 <= index < len(arr):
                return np.array(arr[index], dtype=np.float32)
        except Exception:
            return None
    return None


def _validate_ember_vector(vector: Any) -> tuple[bool, str, Optional[List[float]]]:
    """
    Validate that an input sequence is a valid 2,351-dimensional EMBER vector.

    Returns:
        Tuple of (is_valid, error_message, cleaned_float_list).
    """
    try:
        arr = np.asarray(vector, dtype=np.float64)
    except Exception as exc:
        return False, f"Could not parse features into numeric array: {exc}", None

    if arr.ndim != 1 or arr.shape[0] != 2351:
        return (
            False,
            f"Expected exactly 2,351 features for EMBER 2018 schema, but received {arr.shape[0] if arr.ndim == 1 else arr.shape}.",
            None,
        )

    if np.any(np.isnan(arr)):
        return False, "Feature vector contains NaN (Not a Number) values.", None

    if np.any(np.isinf(arr)):
        return False, "Feature vector contains infinite values.", None

    return True, "", arr.tolist()


def render_page_analyze_sample() -> None:
    """Render the dual-mode sample analysis page."""
    render_header(
        title="Analyze Sample",
        subtitle="Perform non-executing PE static extraction or evaluate EMBER 2018 feature vectors.",
        icon="🔬",
    )

    render_experimental_warning()

    tab_pe, tab_ember = st.tabs([
        "📁 Mode A: Custom PE Static Feature Extractor (85 features)",
        "📊 Mode B: EMBER 2018 Classifier Demo (2,351 features)",
    ])

    # =========================================================================
    # TAB 1: MODE A - CUSTOM PE STATIC FEATURE EXTRACTION
    # =========================================================================
    with tab_pe:
        st.markdown("### Mode A: Custom PE Static Structural Extraction")
        render_disclaimer_card(
            title="Extraction Only — Pipeline Isolation Notice",
            text=(
                "This mode parses Windows PE binaries using ThreatLens' custom static structural extractor (85 features). "
                "CRITICAL: There is currently NO compatible classifier trained on these 85 features. "
                "This mode performs static feature extraction ONLY. No malware score or classification is assigned."
            ),
        )

        uploaded_pe = st.file_uploader(
            "Upload Windows Portable Executable (.exe, .dll, .sys)",
            type=["exe", "dll", "sys"],
            key="pe_file_uploader",
            help="Maximum file size: 50 MB. Files are analyzed in-memory/temp only and never executed.",
        )

        if uploaded_pe is not None:
            max_bytes = DASHBOARD_CONFIG.MAX_UPLOAD_SIZE_MB * 1024 * 1024
            safe_name = Path(uploaded_pe.name).name.replace("..", "").replace("/", "").replace("\\", "")
            buffer_bytes = uploaded_pe.getbuffer()

            if uploaded_pe.size > max_bytes:
                st.error(f"File size ({uploaded_pe.size / (1024*1024):.1f} MB) exceeds maximum allowed limit ({DASHBOARD_CONFIG.MAX_UPLOAD_SIZE_MB} MB).")
            elif len(buffer_bytes) < 64 or buffer_bytes[0:2] != b"MZ":
                st.error(f"Uploaded file '{safe_name}' is not a valid Windows PE binary (missing 'MZ' DOS signature).")
            else:
                st.info(f"Loaded: `{safe_name}` ({uploaded_pe.size:,} bytes). Processing static PE headers...")

                # Save to secure temporary file and clean up immediately
                with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    tmp_file.write(buffer_bytes)

                try:
                    pe_features = extract_pe_features(file_path=tmp_path)
                    set_custom_pe_result(safe_name, pe_features)
                    st.success("Static features extracted successfully. No compatible trained classifier is currently available for this feature schema.")

                except Exception as exc:
                    st.error(f"Static PE extraction failed: {exc}")
                    pe_features = None

                finally:
                    if tmp_path.exists():
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass

                if pe_features:
                    # Display Extracted PE Metadata
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Binary Name", safe_name)
                    with col2:
                        file_sz = pe_features.get("file_size", uploaded_pe.size)
                        st.metric("File Size", f"{file_sz:,} bytes")
                    with col3:
                        file_t = pe_features.get("file_type", "PE32")
                        st.metric("Architecture", "PE32+ (64-bit)" if file_t == "PE32+" else "PE32 (32-bit)")
                    with col4:
                        st.metric("Total Sections", len(pe_features.get("sections", [])))

                    # Sections Table
                    sections = pe_features.get("sections", [])
                    if sections:
                        st.markdown("#### PE Section Metrics & Shannon Entropy")
                        sec_rows = []
                        for sec in sections:
                            sec_rows.append({
                                "Section": sec.get("name", ""),
                                "Virtual Size": sec.get("virtual_size", 0),
                                "Raw Size": sec.get("raw_size", 0),
                                "Entropy (0-8)": round(sec.get("entropy", 0.0), 3),
                                "Readable": sec.get("permissions", {}).get("readable", False),
                                "Writable": sec.get("permissions", {}).get("writable", False),
                                "Executable": sec.get("permissions", {}).get("executable", False),
                            })
                        st.dataframe(pd.DataFrame(sec_rows), use_container_width=True)

                    # Import Summary
                    imports = pe_features.get("imports", {})
                    st.markdown("#### Import Table Summary")
                    col_imp1, col_imp2, col_imp3 = st.columns(3)
                    with col_imp1:
                        st.metric("Imported DLLs", imports.get("total_dll_count", 0))
                    with col_imp2:
                        st.metric("Total Imported Functions", imports.get("total_function_count", 0))
                    with col_imp3:
                        st.metric("Suspicious API Hits", len(imports.get("suspicious_api_hits", [])))

                    # Raw 85-Feature JSON Download
                    json_str = json.dumps(pe_features, indent=2)
                    st.download_button(
                        label="📥 Download Extracted Custom PE Features (JSON)",
                        data=json_str,
                        file_name=f"{uploaded_pe.name}_custom_pe_features.json",
                        mime="application/json",
                    )

    # =========================================================================
    # TAB 2: MODE B - EMBER 2018 CLASSIFIER DEMO
    # =========================================================================
    with tab_ember:
        st.markdown("### Mode B: EMBER 2018 Feature Vector Inference Demo")
        st.markdown(
            "<span class='badge-info' style='font-size:0.85rem;'>EMBER FEATURE VECTOR DEMO — NOT RAW PE CLASSIFICATION</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            "The LightGBM classifier requires a pre-extracted 2,351-dimensional feature vector adhering to the EMBER 2018 schema. "
            "Select a pre-loaded sample from the test split or upload a custom .npy / .json vector below."
        )

        input_choice = st.radio(
            "Vector Input Method:",
            [
                "Use Benchmark Test Split Sample",
                "Upload .npy Feature Vector",
                "Upload .json Feature Vector",
            ],
            horizontal=True,
            key="ember_input_method",
        )

        loaded_vector: Optional[List[float]] = None
        source_label: str = ""

        if input_choice == "Use Benchmark Test Split Sample":
            col_sel, col_info = st.columns([2, 3])
            max_idx = max(0, _get_test_split_size() - 1)
            with col_sel:
                sample_idx = st.slider(f"Select Sample Index from X_test (0 - {max_idx}):", 0, max_idx, 0, key="ember_sample_slider")
            with col_info:
                sample_vec = _load_sample_from_test_split(sample_idx)
                if sample_vec is not None:
                    loaded_vector = sample_vec.tolist()
                    source_label = f"Benchmark X_test Sample #{sample_idx}"
                    st.success(f"Loaded {source_label} (2,351 features, norm: {np.linalg.norm(sample_vec):.2f})")
                else:
                    st.error("Could not load data/processed/X_test.npy. Run scripts/prepare_dataset.py first.")

        elif input_choice == "Upload .npy Feature Vector":
            uploaded_npy = st.file_uploader("Upload .npy Vector (shape: 2351,)", type=["npy"], key="npy_uploader")
            if uploaded_npy is not None:
                try:
                    arr = np.load(uploaded_npy)
                    is_valid, msg, clean_vec = _validate_ember_vector(arr)
                    if is_valid and clean_vec is not None:
                        loaded_vector = clean_vec
                        source_label = f"Uploaded .npy: {uploaded_npy.name}"
                        st.success(f"Validated {source_label} (2,351 features)")
                    else:
                        st.error(f"Validation failed: {msg}")
                except Exception as exc:
                    st.error(f"Failed to read .npy file: {exc}")

        elif input_choice == "Upload .json Feature Vector":
            uploaded_json = st.file_uploader("Upload .json Vector (list of 2,351 numbers)", type=["json"], key="json_uploader")
            if uploaded_json is not None:
                try:
                    data = json.load(uploaded_json)
                    raw_vec = data.get("features", data) if isinstance(data, dict) else data
                    is_valid, msg, clean_vec = _validate_ember_vector(raw_vec)
                    if is_valid and clean_vec is not None:
                        loaded_vector = clean_vec
                        source_label = f"Uploaded .json: {uploaded_json.name}"
                        st.success(f"Validated {source_label} (2,351 features)")
                    else:
                        st.error(f"Validation failed: {msg}")
                except Exception as exc:
                    st.error(f"Failed to parse JSON file: {exc}")

        # Classification Action Button
        if loaded_vector is not None:
            threshold = st.slider(
                "Operational Decision Threshold (t):",
                min_value=0.05,
                max_value=0.95,
                value=st.session_state.get("selected_threshold", 0.50),
                step=0.01,
                key="ember_threshold_slider",
            )
            st.session_state.selected_threshold = threshold

            if st.button("🚀 Run Malware Classification & SHAP Explanation", type="primary", use_container_width=True):
                client = get_api_client()
                with st.spinner("Calling FastAPI /predict endpoint..."):
                    try:
                        pred_resp = client.predict(features=loaded_vector, threshold=threshold)
                        set_active_vector(loaded_vector, source_label)
                        set_prediction(pred_resp)

                        st.markdown("---")
                        st.subheader("Classification Outcome")
                        render_classification_card(
                            prediction_label=pred_resp.get("prediction_label", "UNKNOWN"),
                            raw_probability=pred_resp.get("raw_probability", 0.5),
                            calibrated_probability=pred_resp.get("calibrated_probability"),
                            threshold=threshold,
                            model_version=pred_resp.get("model_version", "1.0.0"),
                        )

                        st.success(
                            f"Sample saved to active session as **{source_label}**. "
                            "You can now navigate to **Prediction Results** or **Local SHAP Explainability** from the sidebar!"
                        )

                    except Exception as exc:
                        st.error(f"FastAPI prediction request failed: {exc}")
