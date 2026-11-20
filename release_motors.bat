@echo off
cd /d "%~dp0"
echo HOLD THE ARM - it goes limp now.
"%LOCALAPPDATA%\miniforge3\envs\lerobot\python.exe" release_motors.py %*
pause
