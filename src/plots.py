from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def save_confusion_matrix(y_true, y_pred, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_roc_curve(y_true, y_proba, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_classification_report_text(report: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path
