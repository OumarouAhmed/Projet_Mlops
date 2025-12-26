import argparse
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from src.config import ARTIFACTS_DIR, PROCESSED_DATA_PATH
from src.data import download_dataset, split_and_save
from src.plots import save_classification_report_text, save_confusion_matrix, save_roc_curve


def load_processed_data(path: Path = PROCESSED_DATA_PATH):
    data = np.load(path)
    return data["X_train"], data["X_test"], data["y_train"], data["y_test"]


def build_model(model_type: str, params: dict) -> Pipeline:
    if model_type == "logreg":
        clf = LogisticRegression(
            C=params.get("C", 1.0),
            solver=params.get("solver", "liblinear"),
            max_iter=params.get("max_iter", 200),
        )
    elif model_type == "svm":
        clf = SVC(
            C=params.get("C", 1.0),
            kernel=params.get("kernel", "rbf"),
            gamma=params.get("gamma", "scale"),
            probability=True,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def ensure_data():
    if not PROCESSED_DATA_PATH.exists():
        download_dataset()
        split_and_save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logreg", "svm"], default="logreg")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--solver", type=str, default="liblinear")
    parser.add_argument("--kernel", type=str, default="rbf")
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--experiment", type=str, default="breast_cancer_classification")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_model", type=str, default=None)
    args = parser.parse_args()

    ensure_data()
    X_train, X_test, y_train, y_test = load_processed_data()

    params = {
        "C": args.C,
        "solver": args.solver,
        "kernel": args.kernel,
        "gamma": args.gamma,
        "model": args.model,
    }

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name):
        model = build_model(args.model, params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "f1": f1_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        artifact_dir = ARTIFACTS_DIR / "mlflow"
        cm_path = save_confusion_matrix(y_test, y_pred, artifact_dir / "confusion_matrix.png")
        roc_path = save_roc_curve(y_test, y_proba, artifact_dir / "roc_curve.png")
        report = classification_report(y_test, y_pred)
        report_path = save_classification_report_text(
            report, artifact_dir / "classification_report.txt"
        )

        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(roc_path))
        mlflow.log_artifact(str(report_path))

        mlflow.sklearn.log_model(model, "model")

        if args.save_model:
            output_path = Path(args.save_model)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, output_path)

        print(metrics)


if __name__ == "__main__":
    main()
