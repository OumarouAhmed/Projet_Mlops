import argparse
from pathlib import Path

import joblib
import mlflow
import optuna
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import ARTIFACTS_DIR
from src.plots import save_classification_report_text, save_confusion_matrix, save_roc_curve
from src.train import build_model, ensure_data, load_processed_data


def objective(trial, X_train, X_test, y_train, y_test):
    params = {
        "C": trial.suggest_float("C", 0.01, 10.0, log=True),
        "solver": trial.suggest_categorical("solver", ["liblinear", "lbfgs"]),
        "model": "logreg",
    }
    model = build_model("logreg", params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "f1": f1_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

    return metrics["f1"], y_pred, y_proba, params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=8)
    parser.add_argument("--experiment", type=str, default="breast_cancer_classification")
    parser.add_argument("--save_model", type=str, default=None)
    args = parser.parse_args()

    ensure_data()
    X_train, X_test, y_train, y_test = load_processed_data()

    mlflow.set_experiment(args.experiment)
    best = {"score": -1.0}

    def optuna_objective(trial):
        score, y_pred, y_proba, params = objective(
            trial, X_train, X_test, y_train, y_test
        )
        if score > best["score"]:
            best.update(
                {"score": score, "y_pred": y_pred, "y_proba": y_proba, "params": params}
            )
        return score

    with mlflow.start_run(run_name="optuna_study"):
        study = optuna.create_study(direction="maximize")
        study.optimize(optuna_objective, n_trials=args.n_trials)

        best_model = build_model("logreg", best["params"])
        best_model.fit(X_train, y_train)

        mlflow.log_params(best["params"])
        mlflow.log_metric("best_f1", best["score"])
        mlflow.log_metrics(
            {
                "best_accuracy": accuracy_score(y_test, best["y_pred"]),
                "best_precision": precision_score(y_test, best["y_pred"]),
                "best_recall": recall_score(y_test, best["y_pred"]),
            }
        )

        artifact_dir = ARTIFACTS_DIR / "mlflow_optuna"
        cm_path = save_confusion_matrix(
            y_test, best["y_pred"], artifact_dir / "confusion_matrix.png"
        )
        roc_path = save_roc_curve(
            y_test, best["y_proba"], artifact_dir / "roc_curve.png"
        )
        report = classification_report(y_test, best["y_pred"])
        report_path = save_classification_report_text(
            report, artifact_dir / "classification_report.txt"
        )

        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(roc_path))
        mlflow.log_artifact(str(report_path))
        mlflow.sklearn.log_model(best_model, "model")

        if args.save_model:
            output_path = Path(args.save_model)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_model, output_path)

        print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
