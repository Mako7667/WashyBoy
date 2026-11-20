@echo off
cd /d "%~dp0"
"%LOCALAPPDATA%\miniforge3\envs\lerobot\python.exe" move_dishes.py teach --leader auto %*
pause
