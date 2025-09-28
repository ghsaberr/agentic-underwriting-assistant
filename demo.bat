@echo off
echo ================================================================================
echo 🤖 AGENTIC UNDERWRITING ASSISTANT - DEMO
echo ================================================================================
echo.

echo 🔍 Starting services...
docker-compose up -d

echo.
echo ⏳ Waiting for services to start (30 seconds)...
timeout /t 30 /nobreak > nul

echo.
echo 🚀 Running demo...
python demo.py

echo.
echo 🛑 Stopping services...
docker-compose down

echo.
echo ✅ Demo completed!
pause
