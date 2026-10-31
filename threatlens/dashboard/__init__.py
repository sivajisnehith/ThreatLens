"""
ThreatLens Dashboard Package

Streamlit-based SOC analyst console for explainable malware detection.
"""

from threatlens.dashboard.api_client import (
    ThreatLensAPIClient,
    APIClientError,
    APIConnectionError,
    APITimeoutError,
    APIResponseError,
)

__all__ = [
    "ThreatLensAPIClient",
    "APIClientError",
    "APIConnectionError",
    "APITimeoutError",
    "APIResponseError",
]
