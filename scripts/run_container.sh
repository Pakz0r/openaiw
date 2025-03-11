#!/bin/bash

# Abilita l'uscita immediata in caso di errore
set -e

# Avvia il nuovo container con la build
docker run --rm -it \
    -v /dev:/dev \
    --device-cgroup-rule "c 81:* rmw" \
    --device-cgroup-rule "c 189:* rmw" \
    openaiw
