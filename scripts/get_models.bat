@echo off
setlocal

:: Configura la directory del progetto
set PROJECT_DIR=%cd%
set APP_DIR=%PROJECT_DIR%\app
set MODELS_DIR=%APP_DIR%\models

REM Download dei modelli di openpose
gdown 1Yn03cKKfVOq4qXmgBMQD20UMRRRkd_tV -O %APP_DIR%\models.tar.gz

REM Unzip dei modelli di openpose
cd /d "%APP_DIR%"
tar -xvzf models.tar.gz
del models.tar.gz