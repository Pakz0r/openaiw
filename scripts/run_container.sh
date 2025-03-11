#!/bin/bash

# Abilita l'uscita immediata in caso di errore
set -e

# Configura la directory di output dell'applicazione
PROJECT_DIR=$(pwd)
OUTPUT_DIR="$PROJECT_DIR/output"

# Crea la directory di output se non esiste
mkdir -p "$OUTPUT_DIR"

# Avvia il nuovo container con la build
docker run --rm -it \
    -v /dev:/dev \
    -v "$OUTPUT_DIR:/app/output" \
    --device-cgroup-rule "c 81:* rmw" \
    --device-cgroup-rule "c 189:* rmw" \
    openaiw
