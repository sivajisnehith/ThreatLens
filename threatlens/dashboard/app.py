"""
ThreatLens — Streamlit Security Analyst Dashboard

Production-grade cybersecurity analyst interface for explainable static PE malware detection.
Connects to the FastAPI backend, provides dual-mode static extraction and EMBER vector inference,
interactive local and global SHAP explainability, and simulated operational threshold analysis.
"""

from __future__ import annotations

import streamlit as st

# Configure Streamlit page options safely
try:
    st.set_page_config(
        page_title="ThreatLens — SOC Analyst Console",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

from threatlens.dashboard.components.styles import apply_custom_css
from threatlens.dashboard.pages import (
    render_page_analyze_sample,
    render_page_global_analysis,
    render_page_local_shap,
    render_page_model_information,
    render_page_prediction_results,
    render_page_security_overview,
    render_page_threshold_analysis,
)
from threatlens.dashboard.state import get_api_client, init_session_state


def main() -> None:
    """Main application loop."""
    # Apply SOC Dark Theme styling
    apply_custom_css()
    init_session_state()

    # Sidebar Navigation & Telemetry
    with st.sidebar:
        st.markdown(
            """
            <div style="margin-bottom: 16px;">
                <div style="font-size: 1.4rem; font-weight: 800; color: #f8fafc; display: flex; align-items: center; gap: 8px;">
                    <span>🛡️</span> ThreatLens
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; letter-spacing: 0.05em; text-transform: uppercase;">
                    Explainable AI Malware Defense
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Backend Health Ping
        client = get_api_client()
        backend_connected = False
        health_info = {}
        try:
            health_info = client.health()
            backend_connected = True
        except Exception:
            backend_connected = False

        if backend_connected:
            v = health_info.get("model_version", "1.0.0")
            model_status = health_info.get("model_status", "SYNTHETIC — RESEARCH ONLY")
            st.session_state.active_model_status = model_status
            if model_status == "GENUINE DATASET — EVALUATION MODEL":
                status_class = "badge-info"
            elif model_status == "MODEL UNAVAILABLE":
                status_class = "badge-danger"
            else:
                status_class = "badge-warning"

            st.markdown(
                f"""
                <div style="margin-bottom: 8px;">
                    <span class="badge-healthy">🟢 Connected (API v{v})</span>
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="{status_class}">{model_status}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.session_state.active_model_status = "MODEL UNAVAILABLE"
            st.markdown(
                """
                <div style="margin-bottom: 8px;">
                    <span class="badge-danger">🔴 Backend Offline</span>
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="badge-danger">MODEL UNAVAILABLE</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Page Navigation
        PAGES = {
            "Security Overview": render_page_security_overview,
            "Analyze Sample": render_page_analyze_sample,
            "Prediction Results": render_page_prediction_results,
            "Local SHAP Explainability": render_page_local_shap,
            "Global Feature Analysis": render_page_global_analysis,
            "Simulated Threshold Analysis": render_page_threshold_analysis,
            "Model Information": render_page_model_information,
        }

        # Check for programmatic navigation request
        default_index = 0
        if "nav_target" in st.session_state and st.session_state.nav_target in PAGES:
            target_page = st.session_state.nav_target
            del st.session_state.nav_target
            default_index = list(PAGES.keys()).index(target_page)

        selected_page = st.radio(
            "Navigation Menu",
            list(PAGES.keys()),
            index=default_index,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Configurable Backend URL
        with st.expander("⚙️ Connection Settings", expanded=False):
            api_url_input = st.text_input(
                "FastAPI URL:",
                value=st.session_state.api_url,
                help="Configurable via THREATLENS_API_URL environment variable.",
            )
            if api_url_input != st.session_state.api_url:
                st.session_state.api_url = api_url_input.strip()
                st.rerun()

        # Telemetry & Specifications Summary
        is_genuine = "GENUINE" in st.session_state.get("active_model_status", "").upper()
        active_prov = "genuine-ember-2018" if is_genuine else "synthetic-ember-2018"
        cal_method = health_info.get("calibration_method", "Sigmoid") or "Platt Scaling (Sigmoid)"
        model_ver = health_info.get("model_version", "7.0.0" if is_genuine else "1.0.0")

        st.markdown(
            f"""
            <div style="font-size: 0.78rem; color: #64748b; line-height: 1.5; margin-bottom: 14px;">
                <div><strong>Classifier:</strong> LightGBM GBDT (v{model_ver})</div>
                <div><strong>Features:</strong> EMBER 2018 (2,351 dims)</div>
                <div><strong>Provenance:</strong> {active_prov}</div>
                <div><strong>Calibration:</strong> {cal_method}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Persistent Warning Badge in Sidebar
        if is_genuine:
            st.markdown(
                """
                <div class="soc-info-banner" style="font-size: 0.74rem; padding: 8px 10px;">
                    <strong>ℹ️ EVALUATION MODEL</strong><br>
                    Trained on genuine EMBER 2018 dataset. Static analysis research only.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="soc-warning-banner" style="font-size: 0.74rem; padding: 8px 10px;">
                    <strong>⚠️ DEMO PROVENANCE</strong><br>
                    Model trained on synthetic data. Never use for real-world quarantine decisions.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Dispatch to Selected Page
    page_func = PAGES.get(selected_page, render_page_security_overview)
    page_func()


if __name__ == "__main__":
    main()
