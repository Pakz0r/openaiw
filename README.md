# Overview
AIWatch is a university research project with is intended to create a real-world video surveillance system that is capable of creating a digital twin for each human detected by the system and detect and report any abnormal behavior on them. Specifically, it wants to create a system in which from cameras placed in the real world, information about the humans in the scene is obtained, including:
- Position and orientation.
- Actions.
This information following processing, through which digital twins are created, is transmitted to the virtual environment within which it is then reproduced.

# VSCode Windows Setup
After creating the python environment in VS Code, create the file [.venv/Lib/site-packages/custom_paths.pth](.venv/Lib/site-packages/custom_paths.pth) and include all the custom paths for all the modules, like:
```
:: The openpose build path
C:\openpose\build
```
