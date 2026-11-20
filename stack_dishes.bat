@echo off
cd /d "%~dp0"
"%LOCALAPPDATA%\miniforge3\envs\lerobot\python.exe" stack_dishes.py %*
pause
