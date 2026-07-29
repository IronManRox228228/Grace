@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   Project Grace - Complete Collaborator Environment Setup
echo ============================================================
echo.

:: 1. Python Virtual Environment Setup
echo [1/6] Setting up Python virtual environment...
if not exist "venv" (
    echo Creating virtual environment in .\venv ...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create Python virtual environment.
        echo Please make sure Python 3.10+ is installed and available on system PATH.
        exit /b 1
    )
) else (
    echo Virtual environment already exists at .\venv.
)

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    exit /b 1
)

echo Upgrading pip and installing Python dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install Python dependencies from requirements.txt.
    exit /b 1
)

pip install -e .
if errorlevel 1 (
    echo [ERROR] Failed to install Grace package in editable mode.
    exit /b 1
)

:: 2. Environment Configuration Setup (.env)
echo.
echo [2/6] Setting up local environment configuration (.env)...
if not exist ".env" (
    if exist ".env.example" (
        echo Creating .env from .env.example ...
        copy .env.example .env
    ) else (
        echo [WARNING] Neither .env nor .env.example found.
    )
) else (
    echo .env configuration file already exists.
)

:: 3. Custom Llama CPP Setup (TurboQuant tqp-v0.3.0)
echo.
echo [3/6] Setting up custom Llama CPP (TurboQuant v0.3.0)...
python scripts/setup_llama_cpp.py
if errorlevel 1 (
    echo [WARNING] Llama CPP setup encountered an error. You can retry running: python scripts/setup_llama_cpp.py
)

:: 4. Speech and Audio Models Setup
echo.
echo [4/6] Downloading required speech/audio models (Vosk, Whisper)...
python scripts/download_models.py
if errorlevel 1 (
    echo [WARNING] Model download script encountered an error. You can retry running: python scripts/download_models.py
)

:: 5. OculiX Java Bridge & JARs Setup
echo.
echo [5/6] Downloading OculiX and OpenCV Java JARs (libs/*.jar)...
python scripts/setup_oculix.py
if errorlevel 1 (
    echo [WARNING] OculiX setup encountered an error. Pure OpenCV fallback will be used.
)

:: 6. Frontend Node.js Dependencies Setup
echo.
echo [6/6] Installing Frontend Node.js dependencies (node_modules)...
if exist "frontend\package.json" (
    cd frontend
    call npm install
    cd ..
) else (
    echo [SKIP] frontend\package.json not found, skipping npm install.
)

echo.
echo ============================================================
echo   [SUCCESS] Full setup complete!
echo   All ignored dependencies and binaries (.env, venv, llama cpp, models, libs/*.jar, node_modules) are ready.
echo   To activate the environment: call venv\Scripts\activate.bat
echo   To launch Grace: python scripts/launch_grace.py or python -m grace
echo ============================================================
pause
