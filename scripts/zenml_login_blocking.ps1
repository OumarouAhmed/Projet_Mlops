$env:ZENML_CONFIG_PATH = "$PWD\.zenml\config\config.yaml"
$env:ZENML_LOCAL_STORES_PATH = "$PWD\.zenml\local_stores"

.\.venv\Scripts\zenml login --local --blocking
