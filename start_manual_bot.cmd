@echo off
rem Start the dc-sandbox manual Telegram bot on this PC.
rem 1. Copy manual.env.example to manual.env and fill in your values.
rem 2. Run this file. Ctrl+C stops it. /stop also stops the current fixer run.
cd /d "%~dp0"
if not exist manual.env (
  echo MISSING manual.env -- copy manual.env.example to manual.env and fill it in.
  exit /b 1
)
for /f "usebackq delims=" %%i in ("manual.env") do set "%%i"
python beacon\manual_bot.py