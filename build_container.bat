@echo off
REM Cancella il container esistente se è in esecuzione
docker stop ai-watch
docker rm ai-watch

REM (Opzionale) Rimuove l'immagine Docker precedente
docker rmi ai-watch

REM Costruisce la nuova immagine Docker
docker build -t ai-watch .
