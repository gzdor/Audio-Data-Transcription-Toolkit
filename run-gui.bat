@echo off
setlocal
cd /d "%~dp0"
title Audio Data Transcription Toolkit
chcp 65001 >nul 2>&1
mode con: cols=96 lines=42
color 0B

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0splash.ps1" -Mode boot

echo.
echo    [*]  Locating Python runtime ...
set "PYTHON_CMD="

python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=python"

if "%PYTHON_CMD%"=="" (
    py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=py -3"
)

if "%PYTHON_CMD%"=="" (
    if exist "%UserProfile%\miniconda3\python.exe" (
        "%UserProfile%\miniconda3\python.exe" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
        if not errorlevel 1 set "PYTHON_CMD=""%UserProfile%\miniconda3\python.exe"""
    )
)

if "%PYTHON_CMD%"=="" (
    color 0C
    echo.
    echo    [X]  Python was not found.
    echo         Install Python 3.12 or 3.13 from the Microsoft Store, then try again.
    echo         See docs\WINDOWS_11_NO_ADMIN_README.md
    echo.
    pause
    exit /b 1
)
echo    [OK] Python runtime online (%PYTHON_CMD%).

echo    [*]  Verifying ASR ^& report engines (faster-whisper, python-docx, PyAV, numpy) ...
%PYTHON_CMD% -c "import faster_whisper, docx, av, numpy, lxml" >nul 2>&1
if errorlevel 1 (
    echo    [*]  Installing required packages from requirements.txt ^(first-run setup^) ...
    %PYTHON_CMD% -m pip install --disable-pip-version-check -r "%~dp0requirements.txt"
    if errorlevel 1 (
        color 0C
        echo.
        echo    [X]  Could not install dependencies from requirements.txt.
        echo         Check your internet connection and try again.
        echo.
        pause
        exit /b 1
    )
)
echo    [OK] ASR ^& Conversation Report engines online.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0splash.ps1" -Mode open
if errorlevel 1 (
    echo    Opening Audio Data Transcription Toolkit GUI ...
)

%PYTHON_CMD% "%~dp0gui.py" %*
if errorlevel 1 (
    color 0C
    echo.
    echo    [X]  The application exited with an error.
    echo.
    pause
    exit /b 1
)
exit /b 0

