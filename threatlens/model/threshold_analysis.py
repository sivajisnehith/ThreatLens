"""
ThreatLens Threshold Analysis and Operating Point Evaluation

Evaluates binary classification metrics across a continuum of decision thresholds,
identifies key operational trade-offs (low-FPR vs low-FNR operational postures),
and exports comprehensive JSON reports and dual-panel visual trade-off curves.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from threatlens.config import DATA_CONFIG
from threatlens.utils.logger import get_logger

logger = get_logger("threshold_analysis")


@dataclass
class ThresholdPoint:
    """Metrics evaluated at a specific decision threshold."""
    threshold: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    recall: float          # TPR
    specificity: float     # TNR
    false_positive_rate: float  # FPR
    false_negative_rate: float  # FNR
    precision: float       # PPV
    f1_score: float
    balanced_accuracy: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert point to dictionary."""
        return asdict(self)


@dataclass
class OperationalPosture:
    """Recommended threshold and trade-off profile for a specific operational deployment."""
    name: str
    description: str
    target_metric: str
    threshold: float
    metrics: Dict[str, float]


@dataclass
class ThresholdAnalysisResult:
    """Complete evaluation report across all evaluated thresholds."""
    total_samples: int
    positive_samples: int
    negative_samples: int
    default_threshold: float
    default_metrics: ThresholdPoint
    points: List[ThresholdPoint]
    postures: Dict[str, OperationalPosture]
    cybersecurity_tradeoff_analysis: str
    is_synthetic: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis result to JSON-serializable dictionary."""
        return {
            "total_samples": int(self.total_samples),
            "positive_samples": int(self.positive_samples),
            "negative_samples": int(self.negative_samples),
            "default_threshold": float(self.default_threshold),
            "default_metrics": self.default_metrics.to_dict(),
            "threshold_points": [p.to_dict() for p in self.points],
            "operational_postures": {k: asdict(v) for k, v in self.postures.items()},
            "cybersecurity_tradeoff_analysis": self.cybersecurity_tradeoff_analysis,
            "is_synthetic": bool(self.is_synthetic),
        }

    def to_json(self, filepath: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        """Serialize analysis to JSON, optionally saving to file."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=indent)
        if filepath is not None:
            p = Path(filepath)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(json_str)
            logger.info(f"Saved threshold analysis report to {p}")
        return json_str


CYBERSECURITY_TRADEOFF_EXPLANATION: str = (
    "OPERATIONAL TRADE-OFF IN STATIC MALWARE DETECTION:\n"
    "1. High-Security / Low-FNR Posture: Low decision thresholds (e.g., t = 0.30 - 0.45) prioritize "
    "detecting all malicious binaries, driving False Negative Rate (FNR) toward 0. In critical "
    "infrastructure or quarantine gateways, missing a zero-day or ransomware payload carries catastrophic "
    "cost. However, this increases False Positives (FPR), requiring higher analyst triage bandwidth.\n"
    "2. Low-Interruption / Low-FPR Posture: High decision thresholds (e.g., t = 0.55 - 0.70) prioritize "
    "minimizing false alarms on legitimate business software, commercial installers, or proprietary tools. "
    "This prevents disruption to business workflows, but risks allowing evasive malware to execute undetected.\n"
    "3. Balanced Posture: Mid-range thresholds (e.g., t = 0.50) maximize F1 score or balanced accuracy, "
    "providing an equal compromise when class costs are roughly symmetric."
)


class ThresholdEvaluator:
    """
    Evaluator for classification thresholds across probability distributions.
    """

    def __init__(
        self,
        default_threshold: float = 0.5,
        reports_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize threshold evaluator.

        Args:
            default_threshold: Primary operating threshold
            reports_dir: Directory where reports and figures are saved
        """
        self.default_threshold = default_threshold
        self.reports_dir = Path(reports_dir) if reports_dir else DATA_CONFIG.REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def calculate_point(probs: np.ndarray, y_true: np.ndarray, threshold: float) -> ThresholdPoint:
        """Calculate metrics at a single threshold."""
        preds = (probs >= threshold).astype(int)

        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))

        pos_count = tp + fn
        neg_count = tn + fp

        recall = float(tp / pos_count) if pos_count > 0 else 0.0
        specificity = float(tn / neg_count) if neg_count > 0 else 0.0
        fpr = float(fp / neg_count) if neg_count > 0 else 0.0
        fnr = float(fn / pos_count) if pos_count > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        bal_acc = float((recall + specificity) / 2.0)

        return ThresholdPoint(
            threshold=round(float(threshold), 4),
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            recall=round(recall, 4),
            specificity=round(specificity, 4),
            false_positive_rate=round(fpr, 4),
            false_negative_rate=round(fnr, 4),
            precision=round(precision, 4),
            f1_score=round(f1, 4),
            balanced_accuracy=round(bal_acc, 4),
        )

    def evaluate(
        self,
        probabilities: np.ndarray,
        y_true: np.ndarray,
        thresholds: Optional[List[float]] = None,
    ) -> ThresholdAnalysisResult:
        """
        Evaluate performance across multiple decision thresholds.

        Args:
            probabilities: 1D array of predicted probabilities
            y_true: 1D array of binary labels {0, 1}
            thresholds: Optional list of thresholds to test

        Returns:
            ThresholdAnalysisResult
        """
        probs = np.asarray(probabilities, dtype=np.float32).flatten()
        y = np.asarray(y_true, dtype=np.int32).flatten()

        if len(probs) != len(y):
            raise ValueError(f"Length mismatch: {len(probs)} probabilities vs {len(y)} labels.")

        if thresholds is None:
            # Default continuum from 0.05 to 0.95 in 0.05 steps, including default_threshold
            t_set = set(np.round(np.arange(0.05, 0.96, 0.05), 2))
            t_set.add(round(self.default_threshold, 2))
            eval_thresholds = sorted(list(t_set))
        else:
            eval_thresholds = sorted(list(set(thresholds)))

        points: List[ThresholdPoint] = []
        for t in eval_thresholds:
            points.append(self.calculate_point(probs, y, t))

        # Default point
        default_point = self.calculate_point(probs, y, self.default_threshold)

        # Identify operational postures based on metrics
        # 1. Best F1
        best_f1_point = max(points, key=lambda p: p.f1_score)
        # 2. Low FNR (e.g. minimal FNR while keeping precision > 0)
        low_fnr_point = min([p for p in points if p.recall > 0], key=lambda p: (p.false_negative_rate, -p.precision))
        # 3. Low FPR (e.g. minimal FPR while maintaining recall > 0)
        low_fpr_point = min([p for p in points if p.specificity > 0], key=lambda p: (p.false_positive_rate, -p.recall))

        postures = {
            "balanced_f1": OperationalPosture(
                name="Balanced (Max F1)",
                description="Optimal harmonic balance between precision and recall.",
                target_metric="f1_score",
                threshold=best_f1_point.threshold,
                metrics=best_f1_point.to_dict(),
            ),
            "high_security_low_fnr": OperationalPosture(
                name="High-Security (Low FNR)",
                description="Minimizes missed malware payloads at the cost of higher analyst triage.",
                target_metric="false_negative_rate",
                threshold=low_fnr_point.threshold,
                metrics=low_fnr_point.to_dict(),
            ),
            "low_interruption_low_fpr": OperationalPosture(
                name="Low-Interruption (Low FPR)",
                description="Minimizes false alarms to avoid disrupting business workflows.",
                target_metric="false_positive_rate",
                threshold=low_fpr_point.threshold,
                metrics=low_fpr_point.to_dict(),
            ),
        }

        pos_count = int(np.sum(y == 1))
        neg_count = int(np.sum(y == 0))

        return ThresholdAnalysisResult(
            total_samples=len(y),
            positive_samples=pos_count,
            negative_samples=neg_count,
            default_threshold=self.default_threshold,
            default_metrics=default_point,
            points=points,
            postures=postures,
            cybersecurity_tradeoff_analysis=CYBERSECURITY_TRADEOFF_EXPLANATION,
            is_synthetic=True,
        )

    def generate_plot(
        self,
        result: ThresholdAnalysisResult,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Generate publication-grade dual-panel threshold analysis visualization.

        Panel 1: Precision, Recall, Specificity, and F1 curves across thresholds.
        Panel 2: FPR vs FNR Trade-Off Curve (Operating Characteristic).
        """
        target = Path(output_path) if output_path else (self.reports_dir / "threshold_analysis.png")
        target.parent.mkdir(parents=True, exist_ok=True)

        t_vals = [p.threshold for p in result.points]
        recalls = [p.recall for p in result.points]
        precisions = [p.precision for p in result.points]
        specificities = [p.specificity for p in result.points]
        f1s = [p.f1_score for p in result.points]
        fprs = [p.false_positive_rate for p in result.points]
        fnrs = [p.false_negative_rate for p in result.points]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

        # Panel 1: Core Performance Metrics
        ax1.plot(t_vals, recalls, label="Recall / TPR", color="#d9534f", linewidth=2.2, marker="o", markersize=4)
        ax1.plot(t_vals, precisions, label="Precision", color="#337ab7", linewidth=2.0, marker="s", markersize=4)
        ax1.plot(t_vals, specificities, label="Specificity (TNR)", color="#5cb85c", linewidth=2.0, linestyle="--")
        ax1.plot(t_vals, f1s, label="F1 Score", color="#f0ad4e", linewidth=2.2, linestyle="-.")
        ax1.axvline(result.default_threshold, color="#888888", linestyle=":", linewidth=1.5, label=f"Default ({result.default_threshold:.2f})")

        ax1.set_title("ThreatLens: Classification Metrics vs. Decision Threshold", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Decision Threshold (tau)", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Metric Value", fontsize=10, fontweight="bold")
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="lower left", fontsize=9)

        # Panel 2: Error Trade-Off (FPR vs FNR)
        ax2.plot(t_vals, fprs, label="False Positive Rate (FPR)", color="#c9302c", linewidth=2.2, marker="^", markersize=4)
        ax2.plot(t_vals, fnrs, label="False Negative Rate (FNR)", color="#ec971f", linewidth=2.2, marker="v", markersize=4)
        ax2.axvline(result.default_threshold, color="#888888", linestyle=":", linewidth=1.5, label=f"Default ({result.default_threshold:.2f})")

        # Highlight low FNR and low FPR postures
        low_fnr = result.postures["high_security_low_fnr"].threshold
        low_fpr = result.postures["low_interruption_low_fpr"].threshold
        ax2.axvline(low_fnr, color="#d9534f", linestyle="--", alpha=0.6, label=f"High-Security ({low_fnr:.2f})")
        ax2.axvline(low_fpr, color="#5cb85c", linestyle="--", alpha=0.6, label=f"Low-Interruption ({low_fpr:.2f})")

        ax2.set_title("Operational Error Trade-Off: False Alarm (FPR) vs. Missed Malware (FNR)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Decision Threshold (tau)", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Error Rate", fontsize=10, fontweight="bold")
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="center right", fontsize=9)

        fig.suptitle(
            "ThreatLens Threshold Analysis: Trade-Off Evaluation for Security Operations",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout()
        plt.savefig(target, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved threshold analysis plot to {target}")
        return target

    def generate_full_report(
        self,
        probabilities: np.ndarray,
        y_true: np.ndarray,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Path]:
        """Generate JSON report and PNG visualization."""
        target_dir = Path(output_dir) if output_dir else self.reports_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        result = self.evaluate(probabilities, y_true)

        json_path = target_dir / "threshold_analysis.json"
        result.to_json(json_path)

        png_path = target_dir / "threshold_analysis.png"
        self.generate_plot(result, png_path)

        return {"json_report": json_path, "plot": png_path}


__all__ = [
    "ThresholdEvaluator",
    "ThresholdAnalysisResult",
    "ThresholdPoint",
    "OperationalPosture",
    "CYBERSECURITY_TRADEOFF_EXPLANATION",
]
