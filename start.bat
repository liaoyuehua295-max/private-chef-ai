@echo off
cd /d "%~dp0"

if not exist ".env" (
    copy .env.example .env
    echo [INFO] .env created, please fill in your API keys and restart.
    pause
    start notepad .env
    exit /b 1
)

echo [1/2] Installing dependencies...
pip install -r requirements.txt -q

echo [2/2] Starting app at http://localhost:8501
streamlit run app.py --server.port 8501 --browser.gatherUsageStats false

pause
