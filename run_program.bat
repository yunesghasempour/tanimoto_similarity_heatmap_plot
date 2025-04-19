@echo off
echo Starting Molecular Tanimoto Calculator...

:: Change to the directory containing this batch file
cd /d "%~dp0"

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python first.
    echo You can download Python from: https://www.python.org/downloads/
    pause
    exit /b
)

:: Check if pip is installed and get its version
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo pip is not installed! Installing pip...
    python -m ensurepip --default-pip
    if errorlevel 1 (
        echo Failed to install pip. Please install it manually.
        pause
        exit /b
    )
)

:: Update pip to the latest version
echo Updating pip to the latest version...
python -m pip install --upgrade pip

:: Check if tonimoto_calculator.py exists
if not exist "tonimoto_calculator.py" (
    echo Error: tonimoto_calculator.py not found in the current directory!
    echo Please make sure tonimoto_calculator.py is in the same folder as this batch file.
    pause
    exit /b
)

:: Check if requirements.txt exists
if not exist "requirements.txt" (
    echo Error: requirements.txt not found in the current directory!
    echo Please make sure requirements.txt is in the same folder as this batch file.
    pause
    exit /b
)

:: Check and install required packages
echo.
echo Checking and installing required packages...
echo This may take a few minutes...
echo.

:: Install required packages
python -m pip install -r requirements.txt --upgrade
if errorlevel 1 (
    echo Failed to install required packages.
    echo Please check your internet connection and try again.
    pause
    exit /b
)

:: Run the program
echo.
echo Starting the program...
echo.
python tonimoto_calculator.py

echo.
echo Press any key to exit...
pause >nul