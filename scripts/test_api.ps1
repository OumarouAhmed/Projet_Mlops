$body = @{
    features = @(
        17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.0787,
        1.095,0.9053,8.589,153.4,0.0064,0.049,0.0537,0.0159,0.030,0.0062,
        25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189
    )
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/predict -Method POST -Body $body -ContentType "application/json"
