from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "breast_cancer.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "train_test.npz"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
PRODUCTION_MODEL_PATH = MODELS_DIR / "production" / "model.joblib"
