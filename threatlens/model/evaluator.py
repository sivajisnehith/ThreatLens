"""
ThreatLens Model Evaluation Suite

Evaluates trained models on the untouched test split, calculating accuracy,
precision, recall, F1, ROC-AUC, PR-AUC, FPR, FNR, confusion matrix, and feature
importances. Generates JSON reports, PNG plots (via Pillow), and interactive Plotly figures.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from threatlens.config import DATA_CONFIG
from threatlens.utils.logger import get_logger

logger = get_logger("model_evaluator")


@dataclass
class EvaluationMetrics:
    """Complete evaluation metrics for binary classification on test set."""
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    roc_auc: float
    pr_auc: float
    false_positive_rate: float
    false_negative_rate: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    total_samples: int
    benign_samples: int
    malicious_samples: int
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


def _draw_confusion_matrix_image(cm: np.ndarray, output_path: Path) -> None:
    """Render publication-quality confusion matrix PNG using Pillow."""
    width, height = 650, 500
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    total = tn + fp + fn + tp

    # Title
    draw.text((30, 20), "ThreatLens Malware Classification Confusion Matrix", fill=(20, 20, 20))
    draw.text((30, 45), f"Untouched Test Set Evaluation (N = {total})", fill=(100, 100, 100))

    # Grid box coordinates
    x0, y0 = 150, 100
    cw, ch = 200, 140

    # Colors
    c_tn = (220, 245, 235)  # Light green/teal
    c_fp = (255, 230, 230)  # Light red
    c_fn = (255, 220, 220)  # Darker red (critical)
    c_tp = (210, 240, 255)  # Light blue

    # Top-Left: TN
    draw.rectangle([x0, y0, x0 + cw, y0 + ch], fill=c_tn, outline=(180, 180, 180), width=2)
    draw.text((x0 + 15, y0 + 15), "True Negative (TN)", fill=(20, 60, 40))
    draw.text((x0 + 15, y0 + 45), f"{tn:,}", fill=(10, 80, 40))
    draw.text((x0 + 15, y0 + 80), f"{tn / total * 100:.1f}% of total", fill=(60, 60, 60))
    draw.text((x0 + 15, y0 + 105), "Correct Benign", fill=(60, 100, 70))

    # Top-Right: FP
    draw.rectangle([x0 + cw, y0, x0 + 2 * cw, y0 + ch], fill=c_fp, outline=(180, 180, 180), width=2)
    draw.text((x0 + cw + 15, y0 + 15), "False Positive (FP)", fill=(100, 30, 30))
    draw.text((x0 + cw + 15, y0 + 45), f"{fp:,}", fill=(180, 30, 30))
    draw.text((x0 + cw + 15, y0 + 80), f"{fp / total * 100:.1f}% of total", fill=(60, 60, 60))
    draw.text((x0 + cw + 15, y0 + 105), "False Alarm (Benign as Mal)", fill=(120, 60, 60))

    # Bottom-Left: FN
    draw.rectangle([x0, y0 + ch, x0 + cw, y0 + 2 * ch], fill=c_fn, outline=(180, 180, 180), width=2)
    draw.text((x0 + 15, y0 + ch + 15), "False Negative (FN)", fill=(140, 20, 20))
    draw.text((x0 + 15, y0 + ch + 45), f"{fn:,}", fill=(200, 20, 20))
    draw.text((x0 + 15, y0 + ch + 80), f"{fn / total * 100:.1f}% of total", fill=(60, 60, 60))
    draw.text((x0 + 15, y0 + ch + 105), "CRITICAL: Missed Malware!", fill=(160, 20, 20))

    # Bottom-Right: TP
    draw.rectangle([x0 + cw, y0 + ch, x0 + 2 * cw, y0 + 2 * ch], fill=c_tp, outline=(180, 180, 180), width=2)
    draw.text((x0 + cw + 15, y0 + ch + 15), "True Positive (TP)", fill=(20, 40, 80))
    draw.text((x0 + cw + 15, y0 + ch + 45), f"{tp:,}", fill=(10, 50, 140))
    draw.text((x0 + cw + 15, y0 + ch + 80), f"{tp / total * 100:.1f}% of total", fill=(60, 60, 60))
    draw.text((x0 + cw + 15, y0 + ch + 105), "Correct Malware Caught", fill=(30, 70, 120))

    # Axis Labels
    draw.text((x0 + 70, y0 - 25), "Predicted: Benign", fill=(50, 50, 50))
    draw.text((x0 + cw + 60, y0 - 25), "Predicted: Malware", fill=(50, 50, 50))
    draw.text((30, y0 + 55), "Actual: Benign", fill=(50, 50, 50))
    draw.text((30, y0 + ch + 55), "Actual: Malware", fill=(50, 50, 50))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")


def _draw_roc_curve_image(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, output_path: Path) -> None:
    """Render ROC curve PNG using Pillow."""
    width, height = 650, 500
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((30, 20), f"ROC Curve (AUC = {roc_auc:.4f})", fill=(20, 20, 20))
    draw.text((30, 45), "Receiver Operating Characteristic on Test Split", fill=(100, 100, 100))

    # Plot bounding box
    px0, py0 = 80, 80
    px1, py1 = 580, 420
    pw, ph = px1 - px0, py1 - py0

    # Draw grid and background
    draw.rectangle([px0, py0, px1, py1], fill=(250, 250, 252), outline=(200, 200, 200), width=1)
    for step in [0.2, 0.4, 0.6, 0.8]:
        gx = px0 + int(step * pw)
        gy = py1 - int(step * ph)
        draw.line([(gx, py0), (gx, py1)], fill=(230, 230, 235), width=1)
        draw.line([(px0, gy), (px1, gy)], fill=(230, 230, 235), width=1)
        draw.text((gx - 10, py1 + 8), f"{step:.1f}", fill=(100, 100, 100))
        draw.text((px0 - 35, gy - 7), f"{step:.1f}", fill=(100, 100, 100))

    # Ticks at 0 and 1
    draw.text((px0 - 5, py1 + 8), "0.0", fill=(100, 100, 100))
    draw.text((px1 - 10, py1 + 8), "1.0", fill=(100, 100, 100))
    draw.text((px0 - 35, py1 - 7), "0.0", fill=(100, 100, 100))
    draw.text((px0 - 35, py0 - 7), "1.0", fill=(100, 100, 100))

    # Diagonal chance line
    draw.line([(px0, py1), (px1, py0)], fill=(180, 180, 180), width=1)

    # Map FPR and TPR points to screen pixels
    pts = []
    for f_val, t_val in zip(fpr, tpr):
        sx = px0 + int(f_val * pw)
        sy = py1 - int(t_val * ph)
        pts.append((sx, sy))

    # Draw ROC curve line
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=(14, 116, 144), width=3)

    # Axis titles
    draw.text((px0 + pw // 2 - 60, py1 + 30), "False Positive Rate (FPR)", fill=(40, 40, 40))
    draw.text((15, py0 + ph // 2 - 10), "TPR", fill=(40, 40, 40))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")


def _draw_pr_curve_image(recall: np.ndarray, precision: np.ndarray, pr_auc: float, output_path: Path) -> None:
    """Render Precision-Recall curve PNG using Pillow."""
    width, height = 650, 500
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((30, 20), f"Precision-Recall Curve (PR-AUC = {pr_auc:.4f})", fill=(20, 20, 20))
    draw.text((30, 45), "Precision vs. Recall Tradeoff on Test Split", fill=(100, 100, 100))

    # Plot bounding box
    px0, py0 = 80, 80
    px1, py1 = 580, 420
    pw, ph = px1 - px0, py1 - py0

    # Draw grid and background
    draw.rectangle([px0, py0, px1, py1], fill=(250, 250, 252), outline=(200, 200, 200), width=1)
    for step in [0.2, 0.4, 0.6, 0.8]:
        gx = px0 + int(step * pw)
        gy = py1 - int(step * ph)
        draw.line([(gx, py0), (gx, py1)], fill=(230, 230, 235), width=1)
        draw.line([(px0, gy), (px1, gy)], fill=(230, 230, 235), width=1)
        draw.text((gx - 10, py1 + 8), f"{step:.1f}", fill=(100, 100, 100))
        draw.text((px0 - 35, gy - 7), f"{step:.1f}", fill=(100, 100, 100))

    # Ticks at 0 and 1
    draw.text((px0 - 5, py1 + 8), "0.0", fill=(100, 100, 100))
    draw.text((px1 - 10, py1 + 8), "1.0", fill=(100, 100, 100))
    draw.text((px0 - 35, py1 - 7), "0.0", fill=(100, 100, 100))
    draw.text((px0 - 35, py0 - 7), "1.0", fill=(100, 100, 100))

    # Map Recall and Precision points to screen pixels
    pts = []
    for r_val, p_val in zip(recall, precision):
        sx = px0 + int(r_val * pw)
        sy = py1 - int(p_val * ph)
        pts.append((sx, sy))

    # Draw PR curve line
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=(124, 58, 237), width=3)

    # Axis titles
    draw.text((px0 + pw // 2 - 30, py1 + 30), "Recall", fill=(40, 40, 40))
    draw.text((15, py0 + ph // 2 - 10), "Precision", fill=(40, 40, 40))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")


class ModelEvaluator:
    """
    Evaluates trained LightGBM models on the untouched test split and exports reports.
    """

    def __init__(self, reports_dir: Optional[Path] = None):
        """Initialize ModelEvaluator with destination reports directory."""
        self.reports_dir = reports_dir or DATA_CONFIG.REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        booster: lgb.Booster,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5,
    ) -> EvaluationMetrics:
        """
        Evaluate model on test dataset.

        Args:
            booster: Trained LightGBM Booster
            X_test: Test features of shape (N, 2351)
            y_test: Test labels of shape (N,)
            threshold: Decision threshold for positive (malicious) class

        Returns:
            EvaluationMetrics dataclass instance
        """
        y_prob = booster.predict(X_test)
        y_pred = (y_prob >= threshold).astype(int)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        try:
            roc_auc = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            roc_auc = 0.5

        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = float(auc(rec_curve, prec_curve))

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        return EvaluationMetrics(
            accuracy=acc,
            precision=prec,
            recall=rec,
            specificity=spec,
            f1=f1,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            total_samples=len(y_test),
            benign_samples=int(np.sum(y_test == 0)),
            malicious_samples=int(np.sum(y_test == 1)),
            threshold=threshold,
        )

    def extract_feature_importance(
        self,
        booster: lgb.Booster,
        feature_names: Optional[List[str]] = None,
        top_n: int = 50,
    ) -> Dict[str, Any]:
        """
        Extract feature importance rankings (split and gain).

        Args:
            booster: Trained LightGBM booster
            feature_names: List of feature names
            top_n: Number of top features to report in detail

        Returns:
            Dictionary with feature importance rankings
        """
        names = feature_names or booster.feature_name()
        gain_imp = booster.feature_importance(importance_type="gain")
        split_imp = booster.feature_importance(importance_type="split")

        ranked_gain_indices = np.argsort(gain_imp)[::-1]
        top_gain = []
        for idx in ranked_gain_indices[:top_n]:
            top_gain.append({
                "rank": len(top_gain) + 1,
                "feature_index": int(idx),
                "feature_name": names[idx] if idx < len(names) else f"feature_{idx}",
                "gain_importance": float(gain_imp[idx]),
                "split_count": int(split_imp[idx]),
            })

        return {
            "total_features": len(gain_imp),
            "top_n": top_n,
            "top_by_gain": top_gain,
        }

    def generate_evaluation_artifacts(
        self,
        booster: lgb.Booster,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> Dict[str, Path]:
        """
        Generate and persist all required Stage 3 reports and visualizations.

        Returns:
            Dictionary mapping artifact names to file paths
        """
        y_prob = booster.predict(X_test)
        y_pred = (y_prob >= threshold).astype(int)
        metrics = self.evaluate(booster, X_test, y_test, threshold=threshold)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        # 1. Classification report JSON
        sk_report = classification_report(
            y_test,
            y_pred,
            target_names=["Benign", "Malicious"],
            output_dict=True,
            zero_division=0,
        )
        report_payload = {
            "threatlens_stage": 3,
            "metrics": metrics.to_dict(),
            "sklearn_classification_report": sk_report,
            "confusion_matrix": {
                "tn": metrics.true_negatives,
                "fp": metrics.false_positives,
                "fn": metrics.false_negatives,
                "tp": metrics.true_positives,
            },
        }
        report_json_path = self.reports_dir / "classification_report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)

        # 2. Confusion matrix PNG
        cm_png_path = self.reports_dir / "confusion_matrix.png"
        _draw_confusion_matrix_image(cm, cm_png_path)

        # 3. ROC Curve PNG
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_png_path = self.reports_dir / "roc_curve.png"
        _draw_roc_curve_image(fpr, tpr, metrics.roc_auc, roc_png_path)

        # 4. Precision-Recall Curve PNG
        prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
        pr_png_path = self.reports_dir / "precision_recall_curve.png"
        _draw_pr_curve_image(rec_c, prec_c, metrics.pr_auc, pr_png_path)

        # 5. Feature Importance JSON
        importance_payload = self.extract_feature_importance(booster, feature_names=feature_names)
        importance_json_path = self.reports_dir / "feature_importance.json"
        with open(importance_json_path, "w", encoding="utf-8") as f:
            json.dump(importance_payload, f, indent=2)

        # 6. Interactive Plotly Figures (HTML)
        try:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={metrics.roc_auc:.4f})"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Chance"))
            fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
            fig_roc.write_html(str(self.reports_dir / "roc_curve.html"))

            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=rec_c, y=prec_c, mode="lines", name=f"PR (AUC={metrics.pr_auc:.4f})"))
            fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
            fig_pr.write_html(str(self.reports_dir / "precision_recall_curve.html"))
        except Exception as e:
            logger.warning(f"Could not generate Plotly HTML reports: {e}")

        logger.info(f"Evaluation artifacts generated in {self.reports_dir}")

        return {
            "classification_report": report_json_path,
            "confusion_matrix_png": cm_png_path,
            "roc_curve_png": roc_png_path,
            "precision_recall_curve_png": pr_png_path,
            "feature_importance_json": importance_json_path,
        }


__all__ = [
    "EvaluationMetrics",
    "ModelEvaluator",
]
