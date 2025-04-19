@echo off

REM Avvia il nuovo container con la build
docker run --runtime=nvidia --gpus all --rm -it ^
    -v /dev:/dev ^
    --device-cgroup-rule "c 81:* rmw" ^
    --device-cgroup-rule "c 189:* rmw" ^
    --name openaiw ^
    --privileged ^
    openaiw
