"""
ThreatLens Dashboard Session State Manager

Manages Streamlit session state across page transitions without caching
sensitive binary payloads or raw uploaded executable files.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import streamlit as st

from threatlens.config import DASHBOARD_CONFIG
from threatlens.dashboard.api_client import ThreatLensAPIClient


def init_session_state() -> None:
    """Initialize default keys in Streamlit session state if not already set."""
    if "api_url" not in st.session_state:
        st.session_state["api_url"] = os.getenv("THREATLENS_API_URL", DASHBOARD_CONFIG.DEFAULT_API_URL)

    if "active_vector" not in st.session_state:
        st.session_state["active_vector"] = None

    if "active_source_name" not in st.session_state:
        st.session_state["active_source_name"] = None

    if "active_prediction" not in st.session_state:
        st.session_state["active_prediction"] = None

    if "active_explanation" not in st.session_state:
        st.session_state["active_explanation"] = None

    if "selected_threshold" not in st.session_state:
        st.session_state["selected_threshold"] = 0.50

    if "selected_top_k" not in st.session_state:
        st.session_state["selected_top_k"] = DASHBOARD_CONFIG.DEFAULT_TOP_K_EXPLAIN

    if "custom_pe_result" not in st.session_state:
        st.session_state["custom_pe_result"] = None

    if "custom_pe_filename" not in st.session_state:
        st.session_state["custom_pe_filename"] = None


def get_api_client() -> ThreatLensAPIClient:
    """
    Get or create an API client targeting the current configured API URL.

    Returns:
        ThreatLensAPIClient instance.
    """
    init_session_state()
    return ThreatLensAPIClient(base_url=st.session_state["api_url"])


def set_active_vector(vector: List[float], source_name: str) -> None:
    """
    Store an active 2,351-dimensional EMBER feature vector for cross-page analysis.

    Args:
        vector: 2,351-dimensional numeric list.
        source_name: Description of vector origin (e.g., 'Test Split Sample #0').
    """
    init_session_state()
    st.session_state["active_vector"] = vector
    st.session_state["active_source_name"] = source_name
    st.session_state["active_prediction"] = None
    st.session_state["active_explanation"] = None


def clear_active_vector() -> None:
    """Clear active feature vector and associated inference results."""
    init_session_state()
    st.session_state["active_vector"] = None
    st.session_state["active_source_name"] = None
    st.session_state["active_prediction"] = None
    st.session_state["active_explanation"] = None


def set_prediction(prediction: Dict[str, Any]) -> None:
    """Save prediction result dictionary to session state."""
    init_session_state()
    st.session_state["active_prediction"] = prediction


def set_explanation(explanation: Dict[str, Any]) -> None:
    """Save SHAP explanation dictionary to session state."""
    init_session_state()
    st.session_state["active_explanation"] = explanation


def set_custom_pe_result(filename: str, result: Dict[str, Any]) -> None:
    """
    Save 85-dimensional custom PE extraction results to session state.
    Does NOT store raw binary executable content.
    """
    init_session_state()
    st.session_state["custom_pe_filename"] = filename
    st.session_state["custom_pe_result"] = result


def clear_custom_pe_result() -> None:
    """Clear custom PE static analysis results."""
    init_session_state()
    st.session_state["custom_pe_filename"] = None
    st.session_state["custom_pe_result"] = None
