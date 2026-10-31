"""
ThreatLens Dashboard FastAPI Client

Provides a synchronous, resilient HTTP client interface to the ThreatLens
FastAPI inference microservice. Handles connection timeouts, schema serialization,
HTTP error decoding, and user-friendly error abstraction for Streamlit.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import httpx

from threatlens.config import DASHBOARD_CONFIG


class APIClientError(Exception):
    """Base exception for all API client errors."""
    pass


class APIConnectionError(APIClientError):
    """Raised when the client cannot connect to the FastAPI backend."""
    pass


class APITimeoutError(APIClientError):
    """Raised when an API request times out."""
    pass


class APIResponseError(APIClientError):
    """Raised when the API returns an error response (HTTP 4xx/5xx)."""

    def __init__(self, message: str, status_code: int = 500, detail: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class ThreatLensAPIClient:
    """Client for communicating with the ThreatLens FastAPI microservice."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        connect_timeout: Optional[float] = None,
        client: Optional[httpx.Client] = None,
    ):
        """
        Initialize the API client.

        Args:
            base_url: Base URL for FastAPI server (defaults to env var or DASHBOARD_CONFIG).
            timeout: Read/write timeout in seconds.
            connect_timeout: Connection attempt timeout in seconds.
            client: Optional pre-configured httpx.Client (useful for testing and mocks).
        """
        env_url = os.getenv("THREATLENS_API_URL")
        self.base_url = (base_url or env_url or DASHBOARD_CONFIG.DEFAULT_API_URL).rstrip("/")
        self.timeout = timeout or DASHBOARD_CONFIG.API_TIMEOUT
        self.connect_timeout = connect_timeout or DASHBOARD_CONFIG.CONNECT_TIMEOUT

        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    self.timeout,
                    connect=self.connect_timeout,
                ),
                headers={"User-Agent": "ThreatLens-Dashboard/1.0.0"},
            )
            self._owns_client = True

    def close(self) -> None:
        """Close the underlying HTTP client session if owned."""
        if self._owns_client and hasattr(self, "_client"):
            self._client.close()

    def __enter__(self) -> ThreatLensAPIClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _handle_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute an HTTP request with error handling.

        Args:
            method: HTTP method ('GET', 'POST', etc.)
            endpoint: URL path relative to base_url (e.g. '/health')
            json_data: Optional JSON payload for POST
            params: Optional query parameters

        Returns:
            Decoded JSON response dictionary.
        """
        clean_endpoint = "/" + endpoint.lstrip("/")
        try:
            response = self._client.request(
                method=method,
                url=clean_endpoint,
                json=json_data,
                params=params,
            )
            response.raise_for_status()
            return response.json()

        except httpx.ConnectError as exc:
            raise APIConnectionError(
                f"Could not connect to ThreatLens backend at {self.base_url}. "
                "Ensure the FastAPI service is running."
            ) from exc

        except httpx.ConnectTimeout as exc:
            raise APITimeoutError(
                f"Connection attempt to {self.base_url} timed out after {self.connect_timeout}s."
            ) from exc

        except (httpx.ReadTimeout, httpx.WriteTimeout) as exc:
            raise APITimeoutError(
                f"Request to {clean_endpoint} timed out after {self.timeout}s."
            ) from exc

        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            try:
                error_body = exc.response.json()
                detail = error_body.get("detail", error_body)
            except Exception:
                detail = exc.response.text or str(exc)

            raise APIResponseError(
                f"API request failed with HTTP {status_code}: {detail}",
                status_code=status_code,
                detail=detail,
            ) from exc

        except httpx.RequestError as exc:
            raise APIConnectionError(f"Network error communicating with API: {exc}") from exc

    def health(self) -> Dict[str, Any]:
        """
        Check FastAPI backend health and model readiness.

        Returns:
            Dict containing status, model_loaded, model_version, calibration_status.
        """
        return self._handle_request("GET", "/health")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retrieve model metadata, architecture specs, and limitations.

        Returns:
            Dict containing algorithm, feature dimensions, provenance, hyperparameters.
        """
        return self._handle_request("GET", "/model/info")

    def get_model_metrics(self) -> Dict[str, Any]:
        """
        Retrieve untouched test split evaluation metrics and confusion matrix.

        Returns:
            Dict containing accuracy, roc_auc, pr_auc, precision, recall, f1, fpr, fnr.
        """
        return self._handle_request("GET", "/metrics/model")

    def predict(
        self,
        features: List[float],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Execute single-sample malware classification on a 2,351-dim EMBER vector.

        Args:
            features: 2,351-dimensional float vector.
            threshold: Classification decision threshold (0.0 to 1.0).

        Returns:
            Dict containing label, raw_probability, calibrated_probability, threshold.
        """
        payload = {
            "features": features,
            "threshold": threshold,
        }
        return self._handle_request("POST", "/predict", json_data=payload)

    def predict_batch(
        self,
        samples: List[List[float]],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Execute batch malware classification on multiple 2,351-dim EMBER vectors.

        Args:
            samples: List of 2,351-dimensional float vectors (max 100).
            threshold: Classification decision threshold.

        Returns:
            Dict containing predictions, probabilities, and count.
        """
        payload = {
            "samples": samples,
            "threshold": threshold,
        }
        return self._handle_request("POST", "/predict/batch", json_data=payload)

    def explain(
        self,
        features: List[float],
        top_k: int = 15,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Compute local SHAP feature attributions for a single 2,351-dim EMBER vector.

        Args:
            features: 2,351-dimensional float vector.
            top_k: Number of positive and negative features to return.
            threshold: Decision threshold.

        Returns:
            Dict containing raw_score, baseline, top_positive_contributors,
            top_negative_contributors, additivity_error, etc.
        """
        payload = {
            "features": features,
            "top_k": top_k,
            "threshold": threshold,
        }
        return self._handle_request("POST", "/explain", json_data=payload)
