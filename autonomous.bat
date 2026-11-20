@echo off
cd /d "%~dp0"
"%LOCALAPPDATA%\miniforge3\envs\lerobot\python.exe" autonomous.py %*
pause
