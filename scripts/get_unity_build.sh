#!/bin/bash

# Abilita l'uscita immediata in caso di errore
set -e

# Effettua il download del progetto da github
git clone https://github.com/Pakz0r/Openpose-Realsense unity || true

# Genera la build del progetto utilizzando docker
docker run --rm \
  -v $(pwd)/unity:/project \
  unityci/editor:ubuntu-2022.3.35f1-linux-il2cpp-3 \
  /opt/unity/Editor/Unity \
  -batchmode -nographics -quit \
  -logFile /dev/stdout \
  -projectPath /project \
  -buildTarget StandaloneLinux64

