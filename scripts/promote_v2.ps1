Copy-Item .\models\v2\model.joblib .\models\production\model.joblib -Force
'v2' | Set-Content .\models\production\VERSION
