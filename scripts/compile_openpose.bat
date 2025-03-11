@echo off
setlocal

:: Configura la directory del progetto
set PROJECT_DIR=%cd%
set THIRDPARTY_LIB_DIR=%PROJECT_DIR%\3rdparty
set OPENPOSE_DIR=%THIRDPARTY_LIB_DIR%\openpose
set BUILD_DIR=%OPENPOSE_DIR%\build
set VENV_DIR=%PROJECT_DIR%\.venv
set OPENPOSE_INSTALL_DIR=%PROJECT_DIR%\openpose

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

:: Crea la cartella 3rdparty
if not exist "%THIRDPARTY_LIB_DIR%" (
    mkdir "%THIRDPARTY_LIB_DIR%"
)

:: Scarica openpose dal repository git ed inizializza submoduli
if not exist "%OPENPOSE_DIR%" (
    cd /d "%THIRDPARTY_LIB_DIR%"
    git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
    cd openpose/
    git submodule update --init --recursive --remote
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
cmake .. -G "Visual Studio 17 2022" ^
    -DBUILD_PYTHON=ON ^
    -DPYTHON_EXECUTABLE=%PYTHON_EXECUTABLE% ^
    -DPYTHON_INCLUDE_DIR=%PYTHON_INCLUDE_DIR% ^
    -DCMAKE_INSTALL_PREFIX=%OPENPOSE_INSTALL_DIR% ^
    -DBUILD_EXAMPLES=OFF ^
    -A x64 ^
    -T v143 ^
    -Wno-deprecated-gpu-targets

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

copy x64\Release\* bin\

echo "Compilazione completata con successo."

:: Copia i file nella cartella locale di openpose

:: Crea la directory di build se non esiste
if not exist "%OPENPOSE_INSTALL_DIR%" (
    mkdir "%OPENPOSE_INSTALL_DIR%"
)

copy bin\* %OPENPOSE_INSTALL_DIR%
copy python\openpose\__init__.py %OPENPOSE_INSTALL_DIR%
copy python\openpose\Release\* %OPENPOSE_INSTALL_DIR%

echo "Installazione completata con successo."
endlocal
