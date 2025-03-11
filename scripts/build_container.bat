@echo off
REM Cancella il container esistente se è in esecuzione
docker stop openaiw
docker rm openaiw

REM (Opzionale) Rimuove l'immagine Docker precedente
docker rmi openaiw

REM Costruisce la nuova immagine Docker
docker build -t openaiw .
