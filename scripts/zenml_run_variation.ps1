$env:ZENML_CONFIG_PATH = "$PWD\.zenml\config\config.yaml"
$env:ZENML_LOCAL_STORES_PATH = "$PWD\.zenml\local_stores"
$env:ZENML_DEFAULT_USER_NAME = "default"
$env:ZENML_DEFAULT_USER_PASSWORD = "default123"
$env:ZENML_DEFAULT_PROJECT_NAME = "default"

.\.venv\Scripts\python -m src.zenml_pipeline --C 0.5
