#!/bin/bash

# Abilita l'uscita immediata in caso di errore
set -e

# Cancella il container esistente se è in esecuzione
docker stop openaiw || true
docker rm openaiw || true

# (Opzionale) Rimuove l'immagine Docker precedente
docker rmi openaiw || true

# Costruisce la nuova immagine Docker
docker build -t openaiw .

# (Opzionale) Avvia il container basato sulla nuova immagine
docker run -d --name openaiw openaiw
