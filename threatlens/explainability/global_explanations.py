"""
ThreatLens Global SHAP Explanations and Group Importance

Calculates dataset-level SHAP feature importance, aggregates attributions across
the 8 official EMBER feature groups, generates summary and bar visualizations,
and contrasts SHAP attribution against LightGBM gain and split importances.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import shap

from threatlens.config import DATA_CONFIG, FEATURE_GROUPS, EmberFeatureGroups
from threatlens.explainability.explainer import (
    SYNTHETIC_MODEL_DISCLAIMER,
    ShapExplainer,
)
from threatlens.explainability.feature_interpreter import FeatureInterpreter
from threatlens.utils.logger import get_logger

logger = get_logger("global_explanations")


METRIC_DISTINCTION_EXPLANATION: Dict[str, str] = {
    "shap_importance": (
        "SHAP Global Feature Importance represents the mean absolute Shapley attribution "
        "(E[|phi_i|]) across the evaluation dataset. It measures the expected magnitude by which "
        "a feature shifts the model prediction (in log-odds margin) relative to the baseline expected "
        "value. It satisfies efficiency, symmetry, and additivity."
    ),
    "lgbm_gain": (
        "LightGBM Gain Importance measures the total reduction in training loss (binary logloss) "
        "contributed by all splits on a given feature during tree construction. Features with high gain "
        "drive significant training error reduction, but may appear in few splits."
    ),
    "lgbm_split": (
        "LightGBM Split Importance counts the raw frequency with which a feature was selected to "
        "split nodes across all boosting iterations. It reflects structural utilization but does not "
        "indicate whether those splits produced large or marginal loss improvements."
    ),
    "non_interchangeability_warning": (
        "CRITICAL: SHAP importance, Gain, and Split metrics are mathematically distinct and NOT "
        "interchangeable. High split count does not imply high SHAP impact, and SHAP attribution "
        "evaluates test-time marginal contributions rather than tree-building loss reductions."
    ),
}


@dataclass
class GroupImportanceMetric:
    """Aggregated SHAP importance for an official EMBER feature group."""
    group_name: str
    feature_start_idx: int
    feature_end_idx: int
    num_features: int
    total_abs_shap: float
    mean_abs_shap_per_feature: float
    percentage_of_total: float
    top_feature_index: int
    top_feature_name: str
    top_feature_shap: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert group metric to dictionary."""
        return asdict(self)


@dataclass
class GlobalImportanceResult:
    """Container for complete global SHAP feature importance analysis."""
    total_samples: int
    total_features: int
    top_k: int
    top_features_by_shap: List[Dict[str, Any]]
    group_importance: Dict[str, Dict[str, Any]]
    top_features_by_gain: List[Dict[str, Any]]
    top_features_by_split: List[Dict[str, Any]]
    metric_distinctions: Dict[str, str]
    is_synthetic_model: bool = True
    disclaimer: str = SYNTHETIC_MODEL_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "total_samples": int(self.total_samples),
            "total_features": int(self.total_features),
            "top_k": int(self.top_k),
            "top_features_by_shap": self.top_features_by_shap,
            "group_importance": self.group_importance,
            "top_features_by_gain": self.top_features_by_gain,
            "top_features_by_split": self.top_features_by_split,
            "metric_distinctions": self.metric_distinctions,
            "is_synthetic_model": bool(self.is_synthetic_model),
            "disclaimer": str(self.disclaimer),
        }

    def to_json(self, filepath: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        """Serialize result to JSON, optionally saving to file."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=indent)
        if filepath is not None:
            out_path = Path(filepath)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            logger.info(f"Saved global importance JSON to {out_path}")
        return json_str


class GlobalExplainer:
    """
    Computes dataset-level SHAP importance, EMBER schema group aggregations,
    and publication-quality visualization figures.
    """

    def __init__(
        self,
        explainer: Optional[ShapExplainer] = None,
        reports_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the global explainer.

        Args:
            explainer: ShapExplainer instance (creates default if None)
            reports_dir: Directory where reports and figures are saved
        """
        self.explainer = explainer or ShapExplainer()
        self.interpreter = self.explainer.interpreter or FeatureInterpreter(
            feature_names=self.explainer.feature_names
        )
        self.reports_dir = Path(reports_dir) if reports_dir else DATA_CONFIG.REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def calculate_global_importance(
        self,
        X: np.ndarray,
        top_k: int = 20,
        max_samples: Optional[int] = 500,
        random_seed: int = 42,
    ) -> Tuple[GlobalImportanceResult, np.ndarray, np.ndarray]:
        """
        Compute mean absolute SHAP values and group aggregations.

        Args:
            X: Input dataset feature matrix (N, 2351)
            top_k: Number of top features to report
            max_samples: Maximum samples to evaluate for performance
            random_seed: Reproducible sampling seed

        Returns:
            Tuple of (GlobalImportanceResult, evaluated_X, shap_matrix)
        """
        X_validated = self.explainer.validate_features(X)
        n_total = len(X_validated)

        # Performance optimization: Subsample representative cohort if dataset is large
        if max_samples is not None and n_total > max_samples:
            rng = np.random.default_rng(random_seed)
            indices = rng.choice(n_total, size=max_samples, replace=False)
            X_eval = X_validated[indices]
            logger.info(f"Subsampled {max_samples} from {n_total} samples for global SHAP analysis.")
        else:
            X_eval = X_validated

        shap_matrix = self.explainer.compute_shap_values(X_eval)

        # 1. Mean absolute and signed SHAP values across evaluation set
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)  # (2351,)
        mean_signed_shap = np.mean(shap_matrix, axis=0)        # (2351,)
        total_global_shap = float(np.sum(mean_abs_shap))

        # 2. Top K features ranked by mean absolute SHAP
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        top_features_shap: List[Dict[str, Any]] = []

        for rank, idx in enumerate(sorted_indices[:top_k], start=1):
            idx_int = int(idx)
            name = self.explainer.feature_names[idx_int]
            group = self.interpreter.get_feature_group(idx_int)
            desc = self.interpreter.get_feature_description(idx_int, name)
            limitations = self.interpreter.get_feature_limitations(idx_int, group)

            top_features_shap.append({
                "rank": rank,
                "feature_index": idx_int,
                "feature_name": name,
                "feature_group": group,
                "mean_abs_shap": float(mean_abs_shap[idx_int]),
                "mean_signed_shap": float(mean_signed_shap[idx_int]),
                "description": desc,
                "limitations": limitations,
            })

        # 3. Aggregation by official EMBER feature groups
        group_results: Dict[str, Dict[str, Any]] = {}
        for group_name in FEATURE_GROUPS.get_all_group_names():
            start, end = FEATURE_GROUPS.get_group_indices(group_name)
            group_shap = mean_abs_shap[start:end]
            group_total = float(np.sum(group_shap))
            group_mean = float(np.mean(group_shap)) if len(group_shap) > 0 else 0.0
            group_pct = (group_total / total_global_shap * 100.0) if total_global_shap > 0 else 0.0

            # Find top feature within this specific group
            group_top_local_idx = int(np.argmax(group_shap))
            top_global_idx = start + group_top_local_idx

            metric = GroupImportanceMetric(
                group_name=group_name,
                feature_start_idx=start,
                feature_end_idx=end,
                num_features=end - start,
                total_abs_shap=round(group_total, 6),
                mean_abs_shap_per_feature=round(group_mean, 6),
                percentage_of_total=round(group_pct, 2),
                top_feature_index=top_global_idx,
                top_feature_name=self.explainer.feature_names[top_global_idx],
                top_feature_shap=round(float(group_shap[group_top_local_idx]), 6),
            )
            group_results[group_name] = metric.to_dict()

        # 4. Extract LightGBM Gain and Split importances for rigorous contrast
        top_gain: List[Dict[str, Any]] = []
        top_split: List[Dict[str, Any]] = []

        if self.explainer.booster is not None:
            gain_importances = self.explainer.booster.feature_importance(importance_type="gain")
            split_importances = self.explainer.booster.feature_importance(importance_type="split")

            gain_sorted = np.argsort(gain_importances)[::-1]
            split_sorted = np.argsort(split_importances)[::-1]

            for rank, idx in enumerate(gain_sorted[:top_k], start=1):
                i = int(idx)
                top_gain.append({
                    "rank": rank,
                    "feature_index": i,
                    "feature_name": self.explainer.feature_names[i],
                    "group": self.interpreter.get_feature_group(i),
                    "gain_importance": float(gain_importances[i]),
                })

            for rank, idx in enumerate(split_sorted[:top_k], start=1):
                i = int(idx)
                top_split.append({
                    "rank": rank,
                    "feature_index": i,
                    "feature_name": self.explainer.feature_names[i],
                    "group": self.interpreter.get_feature_group(i),
                    "split_count": int(split_importances[i]),
                })

        result = GlobalImportanceResult(
            total_samples=len(X_eval),
            total_features=self.explainer.expected_dim,
            top_k=top_k,
            top_features_by_shap=top_features_shap,
            group_importance=group_results,
            top_features_by_gain=top_gain,
            top_features_by_split=top_split,
            metric_distinctions=METRIC_DISTINCTION_EXPLANATION,
            is_synthetic_model=self.explainer.is_synthetic_model,
            disclaimer=SYNTHETIC_MODEL_DISCLAIMER,
        )

        return result, X_eval, shap_matrix

    def generate_summary_plot(
        self,
        shap_matrix: np.ndarray,
        X_eval: np.ndarray,
        output_path: Optional[Union[str, Path]] = None,
        max_display: int = 20,
    ) -> Path:
        """
        Generate publication-grade SHAP summary (beeswarm) plot.

        Args:
            shap_matrix: Calculated SHAP matrix of shape (N, 2351)
            X_eval: Evaluated feature values of shape (N, 2351)
            output_path: File path to save PNG
            max_display: Number of features to display

        Returns:
            Path to generated image
        """
        target = Path(output_path) if output_path else (self.reports_dir / "shap_summary.png")
        target.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(10, 8), dpi=150)
        shap.summary_plot(
            shap_matrix,
            X_eval,
            feature_names=self.explainer.feature_names,
            max_display=max_display,
            show=False,
        )
        plt.title(
            f"ThreatLens: SHAP Global Feature Summary (Top {max_display})\n"
            "[Evaluated on EMBER 2018 Schema]",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )
        plt.tight_layout()
        plt.savefig(target, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved SHAP summary plot to {target}")
        return target

    def generate_bar_plot(
        self,
        shap_matrix: np.ndarray,
        X_eval: np.ndarray,
        output_path: Optional[Union[str, Path]] = None,
        max_display: int = 20,
    ) -> Path:
        """
        Generate global mean |SHAP| feature importance bar chart.

        Args:
            shap_matrix: Calculated SHAP matrix of shape (N, 2351)
            X_eval: Evaluated feature values of shape (N, 2351)
            output_path: File path to save PNG
            max_display: Number of features to display

        Returns:
            Path to generated image
        """
        target = Path(output_path) if output_path else (self.reports_dir / "shap_bar.png")
        target.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(10, 8), dpi=150)
        shap.summary_plot(
            shap_matrix,
            X_eval,
            feature_names=self.explainer.feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        plt.title(
            f"ThreatLens: Global Feature Importance (Mean |SHAP Value|)\n"
            f"Top {max_display} Most Influential Features",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )
        plt.xlabel("Mean |SHAP Value| (Impact on Model Log-Odds Margin)", fontsize=10)
        plt.tight_layout()
        plt.savefig(target, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved SHAP global bar plot to {target}")
        return target

    def generate_group_importance_plot(
        self,
        group_importance: Dict[str, Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Render dual-panel bar plot comparing the 8 EMBER feature groups:
        Panel 1: Total Share of Global Attribution (%)
        Panel 2: Mean Attribution per Feature in Group

        Args:
            group_importance: Dict from GlobalImportanceResult.group_importance
            output_path: Output PNG path

        Returns:
            Path to generated image
        """
        target = Path(output_path) if output_path else (self.reports_dir / "shap_group_importance.png")
        target.parent.mkdir(parents=True, exist_ok=True)

        groups = list(group_importance.keys())
        pcts = [group_importance[g]["percentage_of_total"] for g in groups]
        means = [group_importance[g]["mean_abs_shap_per_feature"] for g in groups]
        counts = [group_importance[g]["num_features"] for g in groups]

        # Order by total percentage descending
        sort_order = np.argsort(pcts)
        sorted_groups = [groups[i] for i in sort_order]
        sorted_pcts = [pcts[i] for i in sort_order]
        sorted_means = [means[i] for i in sort_order]
        sorted_counts = [counts[i] for i in sort_order]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=150)

        # Panel 1: Percentage of total SHAP attribution
        bars1 = ax1.barh(sorted_groups, sorted_pcts, color="#2b5c8f", edgecolor="#1a3654")
        ax1.set_xlabel("Percentage of Total Global Attribution (%)", fontsize=10, fontweight="bold")
        ax1.set_title("Total Attribution Share by EMBER Group", fontsize=11, fontweight="bold")
        ax1.grid(axis="x", linestyle="--", alpha=0.5)

        for bar, count, pct in zip(bars1, sorted_counts, sorted_pcts):
            ax1.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}% (N={count})",
                va="center",
                fontsize=8.5,
                color="#333333",
            )
        ax1.set_xlim(0, max(sorted_pcts) * 1.25 if sorted_pcts else 10)

        # Panel 2: Mean attribution per feature (density)
        bars2 = ax2.barh(sorted_groups, sorted_means, color="#008080", edgecolor="#004d4d")
        ax2.set_xlabel("Mean |SHAP| per Individual Feature", fontsize=10, fontweight="bold")
        ax2.set_title("Feature Attribution Density (Per-Feature Mean)", fontsize=11, fontweight="bold")
        ax2.grid(axis="x", linestyle="--", alpha=0.5)

        for bar, mean_val in zip(bars2, sorted_means):
            ax2.text(
                bar.get_width() + (max(sorted_means) * 0.02 if sorted_means else 0.001),
                bar.get_y() + bar.get_height() / 2,
                f"{mean_val:.4f}",
                va="center",
                fontsize=8.5,
                color="#333333",
            )
        ax2.set_xlim(0, max(sorted_means) * 1.25 if sorted_means else 0.05)

        fig.suptitle(
            "ThreatLens: SHAP Attribution by EMBER 2018 Feature Group\n"
            "[8 Official Feature Groups, 2,351 Dimensions Total]",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()
        plt.savefig(target, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved EMBER group importance plot to {target}")
        return target

    def generate_full_report(
        self,
        X: np.ndarray,
        output_dir: Optional[Union[str, Path]] = None,
        top_k: int = 20,
        max_samples: Optional[int] = 500,
    ) -> Dict[str, Path]:
        """
        Execute full end-to-end global explainability pipeline and export all 4 artifacts:
        1. shap_global_importance.json
        2. shap_summary.png
        3. shap_bar.png
        4. shap_group_importance.png

        Args:
            X: Evaluation feature matrix
            output_dir: Target output directory (defaults to reports/)
            top_k: Top features count
            max_samples: Max samples for evaluation

        Returns:
            Dictionary mapping artifact names to their file paths
        """
        target_dir = Path(output_dir) if output_dir else self.reports_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initiating global SHAP report generation into {target_dir}...")

        # 1. Compute global importances
        result, X_eval, shap_matrix = self.calculate_global_importance(
            X, top_k=top_k, max_samples=max_samples
        )

        # 2. Save JSON report
        json_path = target_dir / "shap_global_importance.json"
        result.to_json(json_path)

        # 3. Generate summary plot
        summary_path = target_dir / "shap_summary.png"
        self.generate_summary_plot(shap_matrix, X_eval, output_path=summary_path, max_display=top_k)

        # 4. Generate bar plot
        bar_path = target_dir / "shap_bar.png"
        self.generate_bar_plot(shap_matrix, X_eval, output_path=bar_path, max_display=top_k)

        # 5. Generate group importance plot
        group_path = target_dir / "shap_group_importance.png"
        self.generate_group_importance_plot(result.group_importance, output_path=group_path)

        logger.info("Successfully generated all global SHAP artifacts.")
        return {
            "json_report": json_path,
            "summary_plot": summary_path,
            "bar_plot": bar_path,
            "group_plot": group_path,
        }


__all__ = [
    "GlobalExplainer",
    "GlobalImportanceResult",
    "GroupImportanceMetric",
    "METRIC_DISTINCTION_EXPLANATION",
]
