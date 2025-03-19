#!/bin/bash

# Abilita l'uscita immediata in caso di errore
set -e

# Effettua download dei modelli di HPE e Falldetection da LFS
git lfs pull

# Configura la directory del progetto
PROJECT_DIR=$(pwd)
APP_DIR="$PROJECT_DIR/app"
MODELS_DIR="$APP_DIR/models"

# Crea la directory dei modelli se non esiste
mkdir -p "$MODELS_DIR"

# Download dei modelli di OpenPose
gdown 1Yn03cKKfVOq4qXmgBMQD20UMRRRkd_tV -O "$APP_DIR/models.tar.gz"

# Estrai i modelli
tar -xvzf "$APP_DIR/models.tar.gz" -C "$MODELS_DIR"

echo "Download e estrazione completati!"
