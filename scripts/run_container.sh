#!/bin/bash

# Abilita l'uscita immediata in caso di errore
set -e

# Configura la directory di output dell'applicazione
PROJECT_DIR=$(pwd)
OUTPUT_DIR="$PROJECT_DIR/output"
APP_DIR="$PROJECT_DIR/app"

# Crea la directory di output se non esiste
mkdir -p "$OUTPUT_DIR"

# Avvia il nuovo container con la build
xhost +local:docker
docker run --runtime=nvidia --rm -it \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /dev:/dev \
    -v "$APP_DIR:/app" \
    -v "$OUTPUT_DIR:/app/output" \
    --device-cgroup-rule "c 81:* rmw" \
    --device-cgroup-rule "c 189:* rmw" \
    openaiw
