import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH


def download_dataset(output_path: Path = RAW_DATA_PATH) -> Path:
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    df.to_csv(output_path, index=False)
    return output_path


def split_and_save(
    raw_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Path:
    df = pd.read_csv(raw_path)
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split", action="store_true")
    args = parser.parse_args()

    if args.download:
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        path = download_dataset()
        print(f"Saved raw data to {path}")

    if args.split:
        if not RAW_DATA_PATH.exists():
            raise FileNotFoundError(
                f"{RAW_DATA_PATH} not found. Run with --download first."
            )
        path = split_and_save()
        print(f"Saved processed data to {path}")


if __name__ == "__main__":
    main()
