@echo off
cd /d "%~dp0"
"%LOCALAPPDATA%\miniforge3\envs\lerobot\python.exe" check_arm.py %*
pause
