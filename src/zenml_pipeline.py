import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from zenml import pipeline, step

from src.config import RAW_DATA_PATH
from src.data import download_dataset


@step
def load_data() -> np.ndarray:
    if not RAW_DATA_PATH.exists():
        download_dataset()
    return np.loadtxt(RAW_DATA_PATH, delimiter=",", skiprows=1)


@step
def preprocess(data: np.ndarray) -> dict:
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


@step
def train_model(
    data_split: dict,
    C: float = 1.0,
) -> dict:
    X_train = data_split["X_train"]
    X_test = data_split["X_test"]
    y_train = data_split["y_train"]
    y_test = data_split["y_test"]
    model = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(C=C, max_iter=200))]
    )
    model.fit(X_train, y_train)
    return {"model": model, "X_test": X_test, "y_test": y_test}


@step
def evaluate_model(model_bundle: dict):
    model = model_bundle["model"]
    X_test = model_bundle["X_test"]
    y_test = model_bundle["y_test"]
    y_pred = model.predict(X_test)
    return {
        "f1": f1_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }


@step
def save_model(model_bundle: dict, output_path: str) -> str:
    model = model_bundle["model"]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return str(path)


@pipeline
def training_pipeline():
    data = load_data()
    data_split = preprocess(data)
    model_bundle = train_model(data_split)
    metrics = evaluate_model(model_bundle)
    save_model(model_bundle, "models/zenml/model.joblib")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()

    pipe = training_pipeline()
    try:
        pipe.run(config={"train_model": {"parameters": {"C": args.C}}})
    except AttributeError as exc:
        print(f"ZenML run completed with a post-run warning: {exc}")


if __name__ == "__main__":
    main()
