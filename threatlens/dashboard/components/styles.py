"""
ThreatLens Dashboard Custom Styles & CSS Injection

Provides custom styling for a professional SOC analyst dashboard:
- Dark charcoal / deep navy palette
- Subtle cyan/blue accents
- High-contrast alert boxes and metric cards
- Clean typography and monospace data formatting
"""

import streamlit as st


DARK_THEME_CSS = """
<style>
/* Main app background and typography */
.stApp {
    background-color: #0b0f19;
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0d1322;
    border-right: 1px solid #1e293b;
}

/* Hide Streamlit auto-generated multi-page navigation menu completely */
[data-testid="stSidebarNav"],
div[data-testid="stSidebarNav"],
ul[data-testid="stSidebarNavItems"],
section[data-testid="stSidebarNav"],
div[data-testid="stSidebarNavSeparator"],
nav[data-testid="stSidebarNav"] {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    visibility: hidden !important;
}

/* Sleek Sidebar Navigation Radio Styling */
[data-testid="stSidebar"] div.stRadio > div[role="radiogroup"] {
    gap: 4px;
}

[data-testid="stSidebar"] div.stRadio > div[role="radiogroup"] > label {
    background-color: #131b2e;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 3px;
    cursor: pointer;
    transition: all 0.15s ease-in-out;
    width: 100%;
}

[data-testid="stSidebar"] div.stRadio > div[role="radiogroup"] > label:hover {
    background-color: #1a253c;
    border-color: #38bdf8;
}

[data-testid="stSidebar"] div.stRadio > div[role="radiogroup"] > label:has(input:checked) {
    background-color: rgba(56, 189, 248, 0.15);
    border-color: #38bdf8;
    color: #f8fafc;
    font-weight: 600;
}

/* Card container */
.soc-card {
    background-color: #131b2e;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
}

.soc-card:hover {
    border-color: #38bdf8;
    transition: border-color 0.2s ease-in-out;
}

/* Metric card */
.metric-box {
    background-color: #162036;
    border: 1px solid #22314d;
    border-radius: 6px;
    padding: 12px 16px;
    text-align: center;
}

.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    font-family: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
    color: #f8fafc;
}

.metric-sub {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 2px;
}

/* Status Badges */
.badge-healthy {
    display: inline-block;
    padding: 2px 8px;
    background-color: rgba(16, 185, 129, 0.15);
    border: 1px solid #10b981;
    color: #34d399;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

.badge-warning {
    display: inline-block;
    padding: 2px 8px;
    background-color: rgba(245, 158, 11, 0.15);
    border: 1px solid #f59e0b;
    color: #fbbf24;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

.badge-danger {
    display: inline-block;
    padding: 2px 8px;
    background-color: rgba(239, 68, 68, 0.15);
    border: 1px solid #ef4444;
    color: #f87171;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

.badge-info {
    display: inline-block;
    padding: 2px 8px;
    background-color: rgba(56, 189, 248, 0.15);
    border: 1px solid #38bdf8;
    color: #7dd3fc;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Experimental warning banner */
.soc-warning-banner {
    background-color: rgba(245, 158, 11, 0.12);
    border-left: 4px solid #f59e0b;
    border-top: 1px solid rgba(245, 158, 11, 0.3);
    border-right: 1px solid rgba(245, 158, 11, 0.3);
    border-bottom: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 4px;
    padding: 12px 16px;
    margin-bottom: 16px;
    color: #fbbf24;
    font-size: 0.85rem;
}

.soc-info-banner {
    background-color: rgba(56, 189, 248, 0.12);
    border-left: 4px solid #38bdf8;
    border-top: 1px solid rgba(56, 189, 248, 0.3);
    border-right: 1px solid rgba(56, 189, 248, 0.3);
    border-bottom: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 4px;
    padding: 12px 16px;
    margin-bottom: 16px;
    color: #7dd3fc;
    font-size: 0.85rem;
}

/* Page headers */
.page-header-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.page-header-subtitle {
    font-size: 0.9rem;
    color: #94a3b8;
    margin-bottom: 20px;
}

/* Code blocks / tables monospace */
code, pre {
    font-family: "JetBrains Mono", Consolas, Menlo, monospace !important;
}

/* StTables / DataFrames */
.stDataFrame {
    border: 1px solid #1e293b;
    border-radius: 6px;
}
</style>
"""


def apply_custom_css() -> None:
    """Inject SOC dark theme CSS into Streamlit page head."""
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, icon: str = "🛡️") -> None:
    """Render a unified SOC-styled page header."""
    st.markdown(
        f"""
        <div style="margin-bottom: 18px;">
            <div class="page-header-title">{icon} {title}</div>
            <div class="page-header-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_experimental_warning(custom_text: str | None = None, is_genuine: bool | None = None) -> None:
    """Render a high-visibility notice reflecting active model provenance."""
    if is_genuine is None:
        model_status = st.session_state.get("active_model_status", "")
        is_genuine = "GENUINE" in str(model_status).upper()

    if custom_text:
        message = custom_text
    elif is_genuine:
        message = (
            "GENUINE EMBER 2018 EVALUATION MODEL: The active classifier was retrained on genuine EMBER 2018 "
            "benchmark records. Model scores, calibrated probabilities, and SHAP attributions reflect real PE static "
            "feature distributions for cybersecurity research and evaluation."
        )
    else:
        message = (
            "EXPERIMENTAL/DEMO SYSTEM: The active model was trained on synthetic EMBER 2018 benchmark "
            "distributions. Model scores, classifications, and SHAP attributions demonstrate system pipeline mechanics "
            "and must NOT be used for real-world malware quarantine or incident response decisions."
        )

    badge_title = "ℹ️ EVALUATION MODEL NOTICE:" if is_genuine else "⚠️ EXPERIMENTAL PROVENANCE NOTICE:"
    banner_class = "soc-info-banner" if is_genuine else "soc-warning-banner"

    st.markdown(
        f"""
        <div class="{banner_class}">
            <strong>{badge_title}</strong><br>
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer_card(title: str, text: str) -> None:
    """Render an informational or security limitation card."""
    st.markdown(
        f"""
        <div class="soc-card" style="border-left: 3px solid #38bdf8;">
            <div style="font-weight: 600; color: #38bdf8; margin-bottom: 4px;">{title}</div>
            <div style="font-size: 0.85rem; color: #94a3b8;">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
