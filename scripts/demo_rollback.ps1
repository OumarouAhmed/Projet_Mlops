.\scripts\promote_v1.ps1
.\scripts\restart_api.ps1
Start-Sleep -Seconds 3
.\scripts\test_api.ps1

.\scripts\promote_v2.ps1
.\scripts\restart_api.ps1
Start-Sleep -Seconds 3
.\scripts\test_api.ps1

.\scripts\promote_v1.ps1
.\scripts\restart_api.ps1
Start-Sleep -Seconds 3
.\scripts\test_api.ps1
