@echo off
setlocal

:: Configura la directory del progetto
set PROJECT_DIR=%cd%
set OPENPOSE_DIR=%PROJECT_DIR%\openpose
set BUILD_DIR=%OPENPOSE_DIR%\build
set VENV_DIR=%PROJECT_DIR%\.venv

:: Configura Python dal virtual environment
set PYTHON_EXECUTABLE=%VENV_DIR%\Scripts\python.exe
set PYTHON_INCLUDE_DIR=%VENV_DIR%\Include
set PYTHON_LIBRARY=%VENV_DIR%\libs\python38.lib

:: Verifica se il virtual environment esiste
if not exist "%PYTHON_EXECUTABLE%" (
    echo "Virtual environment non trovato in: %VENV_DIR%"
    echo "Crea il virtual environment con: python -m venv .venv"
    exit /b 1
)

:: Verifica se CMake è installato
where cmake >nul 2>nul
if errorlevel 1 (
    echo "CMake non trovato. Assicurati che CMake sia nel PATH."
    exit /b 1
)

:: Crea la directory di build se non esiste
if not exist "%BUILD_DIR%" (
    mkdir "%BUILD_DIR%"
)

:: Passa alla directory di build
cd /d "%BUILD_DIR%"

:: Configura la build con CMake per Visual Studio 2022
cmake -G "Visual Studio 17 2022" ^
      -DBUILD_PYTHON=ON ^
      -DPYTHON_EXECUTABLE=%PYTHON_EXECUTABLE% ^
      -DPYTHON_INCLUDE_DIR=%PYTHON_INCLUDE_DIR% ^
      -DBUILD_EXAMPLES=ON ^
      -DGPU_MODE=CPU_ONLY ^
      %OPENPOSE_DIR% ^
      -Ax64 ^
      -Wno-dev

if errorlevel 1 (
    echo "Errore durante la configurazione con CMake."
    exit /b 1
)

:: Compila OpenPose
cmake --build . --config Release

if errorlevel 1 (
    echo "Errore durante la compilazione."
    exit /b 1
)

copy x64\Release\*  bin\

echo "Compilazione completata con successo."
endlocal
