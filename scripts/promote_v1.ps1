Copy-Item .\models\v1\model.joblib .\models\production\model.joblib -Force
'v1' | Set-Content .\models\production\VERSION
