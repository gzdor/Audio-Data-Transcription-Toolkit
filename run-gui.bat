@echo off
setlocal
cd /d "%~dp0"
title Audio Data Transcription Toolkit
chcp 65001 >nul 2>&1
mode con: cols=96 lines=40
color 0B

echo.
echo           _   _   _ ____ ___ ___    ____    _  _____  _
echo          / \ ^| ^| ^| ^|  _ \_ _/ _ \  ^|  _ \  / \^|_   _^|/ \
echo         / _ \^| ^| ^| ^| ^| ^| ^| ^| ^| ^| ^| ^| ^| ^| ^|/ _ \ ^| ^| / _ \
echo        / ___ \ ^|_^| ^| ^|_^| ^| ^| ^|_^| ^| ^| ^|_^| / ___ \^| ^|/ ___ \
echo       /_/   \_\___/^|____/___\___/  ^|____/_/   \_\_/_/   \_\
echo.
echo            T R A N S C R I P T I O N   ^&   R E P O R T   T O O L K I T
echo            A U D I O   ^&   V I D E O   ( . W A V / . M P 3 / . M P 4 )
echo.
echo     +------------------------------------------------------------------+
echo     ^|  100%% Local Whisper ASR   Multi-CPU / NVIDIA CUDA / AMD ROCm     ^|
echo     ^|  WAV / MP3 / MP4 / M4A / FLAC / MKV  -^>  .TXT / .DOCX / .SRT     ^|
echo     +------------------------------------------------------------------+
echo.
echo     [*]  Locating Python 3.10+ runtime ...
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
    echo     [X]  Python 3.10+ was not found on PATH.
    echo          Install Python 3.12 or 3.13 from the Microsoft Store, then try again.
    echo          See docs\WINDOWS_11_NO_ADMIN_README.md
    echo.
    pause
    exit /b 1
)
echo     [OK] Python runtime online (%PYTHON_CMD%).

echo     [*]  Checking required Python packages (faster-whisper, python-docx, av, numpy, lxml) ...
%PYTHON_CMD% -c "import faster_whisper, docx, av, numpy, lxml" >nul 2>&1
if errorlevel 1 (
    echo.
    echo     [!]  One or more required Python libraries from requirements.txt are not installed.
    echo          Command to run: %PYTHON_CMD% -m pip install -r requirements.txt
    echo.
    choice /C YN /M "    Install required Python packages from requirements.txt now"
    if errorlevel 2 (
        echo.
        echo     [!]  Cancelled package installation. Run "pip install -r requirements.txt" manually.
        pause
        exit /b 1
    )
    echo.
    echo     [*]  Installing packages from requirements.txt ...
    %PYTHON_CMD% -m pip install --disable-pip-version-check -r "%~dp0requirements.txt"
    if errorlevel 1 (
        color 0C
        echo.
        echo     [X]  Could not install dependencies from requirements.txt.
        echo          Check your internet connection and try again.
        echo.
        pause
        exit /b 1
    )
)
echo     [OK] ASR ^& Executive Summary Report (.docx) engines online.
echo.
echo     [OK] Launching Audio Data Transcription Toolkit GUI ...
echo.

%PYTHON_CMD% "%~dp0gui.py" %*
if errorlevel 1 (
    color 0C
    echo.
    echo     [X]  The application exited with an error.
    echo.
    pause
    exit /b 1
)
exit /b 0
