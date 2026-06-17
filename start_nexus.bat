@echo off
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak >nul
start "NEXUS Rocketry" /MIN "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -m streamlit run "%~dp0ballistics_app.py" --server.port 8502 --server.headless true
timeout /t 6 /nobreak >nul
start "" "http://localhost:8502"
