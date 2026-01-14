# Mini-projet MLOps â€“ Classification cancer du sein


## Auteurs
- Oumekelthoum Mohamed
- Ahmed Oumarou
Objectif: pipeline complet (data â†’ train â†’ suivi dâ€™expÃ©riences â†’ dÃ©ploiement) avec DVC, MLflow, ZenML et Docker.

## Structure
- `src/`: scripts ML, Optuna, pipeline ZenML, API FastAPI
- `data/`: donnÃ©es versionnÃ©es par DVC
- `models/`: modÃ¨les (v1, v2, production)
- `artifacts/`: graphiques (confusion matrix, ROC)
- `docker-compose.yml`: MLflow + API

## PrÃ©requis
- Python 3.10+
- Docker + Docker Compose
- DVC

## Installation
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## DonnÃ©es + DVC
1) TÃ©lÃ©charger et prÃ©parer les donnÃ©es:
```bash
python -m src.data --download --split
```
Ou via script:
```bash
.\scripts\setup_data.ps1
```

2) Initialiser DVC et tracker les donnÃ©es:
```bash
dvc init
dvc add data/raw/breast_cancer.csv data/processed/train_test.npz
```
Ou via pipeline DVC:
```bash
dvc repro
```

3) Configurer un remote DVC (exemple local):
```bash
dvc remote add -d storage dvc_storage
dvc push
```

Pour reproduire sur une autre machine:
```bash
dvc pull
```

## EntraÃ®nement + MLflow
Basline v1 (Logistic Regression):
```bash
python -m src.train --model logreg --C 1.0 --solver liblinear --run_name baseline --save_model models/v1/model.joblib
```
Ou via script:
```bash
.\scripts\train_baseline.ps1
```

Variations:
```bash
python -m src.train --model logreg --C 0.5 --solver lbfgs --run_name variation_lr
python -m src.train --model svm --C 1.0 --kernel rbf --run_name variation_svm
```
Ou via script:
```bash
.\scripts\train_variations.ps1
```

Optuna (v2):
```bash
python -m src.optuna_tune --n_trials 8 --save_model models/v2/model.joblib
```
Ou via script:
```bash
.\scripts\run_optuna.ps1
```

Lancer MLflow UI:
```bash
mlflow ui
```

## Pipeline ZenML
```bash
python -m src.zenml_pipeline --C 1.0
```
Ou via script:
```bash
.\scripts\run_zenml.ps1
```
Version recommandÃ©e (Windows): utiliser le venv et le script local:
```bash
.\scripts\run_zenml_local.ps1
```
Run variation ZenML (C=0.5):
```bash
.\scripts\zenml_run_variation.ps1
```
Si la crÃ©ation du user/stack bloque, lancer le login local (dans un autre terminal):
```bash
.\scripts\zenml_login_blocking.ps1
```
Pour afficher les runs dans le dashboard local, garde le login en mode blocking
et ouvre lâ€™URL indiquÃ©e.
Exporter la liste des runs (JSON) pour preuve:
```bash
.\scripts\zenml_runs_json.ps1
```
Si tu as lâ€™erreur `ModuleNotFoundError: zenml.zen_stores`, rÃ©installe ZenML dans un venv propre:
```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```
Si la config ZenML locale est corrompue, supprime `.zenml/` et relance:
```bash
Remove-Item .\\.zenml -Recurse -Force
.\.venv\Scripts\python -m src.zenml_pipeline --C 1.0
```

## API dâ€™infÃ©rence (local)
```bash
uvicorn src.api:app --reload
```

Exemple de requÃªte:
```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"features\":[17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.0787,1.095,0.9053,8.589,153.4,0.0064,0.049,0.0537,0.0159,0.030,0.0062,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}"
```

## Docker Compose (API + MLflow)
Lâ€™image API utilise `requirements-api.txt` pour limiter les dÃ©pendances Docker.

1) Mettre en production un modÃ¨le:
```bash
Copy-Item .\models\v1\model.joblib .\models\production\model.joblib
```
Ou via script:
```bash
.\scripts\promote_v1.ps1
```

2) Lancer la stack:
```bash
docker-compose up --build
```
Ou via script:
```bash
.\scripts\compose_up.ps1
```

- MLflow: http://localhost:5000
- API: http://localhost:8000

## Captures Ã  fournir
- MLflow: liste des runs + comparaison (baseline/variations/Optuna) + artefacts (confusion/ROC).
- ZenML: dashboard avec pipeline + exÃ©cutions (au moins 2 runs).
- DVC: commande `dvc pull` (ou `dvc repro`) + preuve des fichiers rÃ©cupÃ©rÃ©s.

## DÃ©mo v1 â†’ v2 â†’ rollback
1) v1:
```bash
Copy-Item .\models\v1\model.joblib .\models\production\model.joblib
docker-compose up --build
```
Test /predict (capture).

2) v2:
```bash
Copy-Item .\models\v2\model.joblib .\models\production\model.joblib
```
RedÃ©marrer le conteneur API et retester /predict.
```bash
.\scripts\restart_api.ps1
```

3) Rollback v1:
```bash
Copy-Item .\models\v1\model.joblib .\models\production\model.joblib
```
RedÃ©marrer lâ€™API et retester /predict.
```bash
.\scripts\restart_api.ps1
```

Script de test API:
```bash
.\scripts\test_api.ps1
```
Script dÃ©mo v1 â†’ v2 â†’ rollback:
```bash
.\scripts\demo_rollback.ps1
```

Le champ `model_version` est lu depuis `models/production/VERSION`. Les scripts de promotion lâ€™Ã©crivent automatiquement.
