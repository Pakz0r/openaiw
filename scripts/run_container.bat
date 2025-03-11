@echo off

REM Avvia il nuovo container con la build
docker run --runtime=nvidia --gpus all --rm -it ^
    -v /dev:/dev ^
    -v /tmp/.X11-unix/:/tmp/.X11-unix:rw ^
    --device /dev/video0:/dev/video0:mwr ^
    --device /dev/video1:/dev/video1:mwr ^
    --device /dev/video2:/dev/video2:mwr ^
    --device /dev/video3:/dev/video3:mwr ^
    --device /dev/video4:/dev/video4:mwr ^
    --device /dev/video5:/dev/video5:mwr ^
    --device-cgroup-rule "c 81:* rmw" ^
    --device-cgroup-rule "c 189:* rmw" ^
    --name ai-watch ^
    --net=host ^
    -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR ^
    -e DISPLAY=$DISPLAY ^
    --privileged ^
    ai-watch
