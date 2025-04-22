ARG BASE_IMAGE=nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

#################################
#    Base Image Builder Stage   #
#################################
FROM $BASE_IMAGE AS base-builder

# To avoid waiting for input during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Builder dependencies installation
RUN apt-get update \
    && apt-get install -qq -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    libopencv-dev \
    libopenni2-dev \
    openni2-utils \
    python3 \
    python3-dev \
    python3-pip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install python dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install numpy opencv-python

#################################
#     Openpose Builder Stage    #
#################################
FROM base-builder AS openpose-builder

# To avoid waiting for input during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Builder dependencies installation
RUN apt-get update \
    && apt-get install -qq -y --no-install-recommends \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    libboost-all-dev \
    libhdf5-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Download sources
WORKDIR /usr/src
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git && \
    cd openpose && git submodule update --init --recursive --remote

# Build and install
RUN cd /usr/src/openpose \
    && mkdir build && cd build \
    && cmake \
    -DCMAKE_INSTALL_PREFIX=/opt/openpose \
    -DBUILD_PYTHON=ON \
    -DDOWNLOAD_BODY_25_MODEL=OFF \
    -DDOWNLOAD_FACE_MODEL=OFF \ 
    -DDOWNLOAD_HAND_MODEL=OFF \
    -DUSE_CUDNN=OFF \
    -DBUILD_EXAMPLES=OFF \
    .. && make -j$(($(nproc)-1)) && make install

#################################
#   Librealsense Builder Stage  #
#################################
FROM base-builder AS librealsense-builder

# To avoid waiting for input during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Builder dependencies installation
RUN apt-get update \
    && apt-get install -qq -y --no-install-recommends \
    curl \
    libssl-dev \
    libusb-1.0-0-dev \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Download sources
WORKDIR /usr/src
RUN curl https://codeload.github.com/IntelRealSense/librealsense/tar.gz/refs/tags/v2.56.3 -o librealsense.tar.gz 
RUN tar -zxf librealsense.tar.gz && rm librealsense.tar.gz 
RUN ln -s /usr/src/librealsense-2.56.3 /usr/src/librealsense

# Build and install
RUN cd /usr/src/librealsense \
    && mkdir build && cd build \
    && cmake \
    -DCMAKE_C_FLAGS_RELEASE="${CMAKE_C_FLAGS_RELEASE} -s" \
    -DCMAKE_CXX_FLAGS_RELEASE="${CMAKE_CXX_FLAGS_RELEASE} -s" \
    -DCMAKE_INSTALL_PREFIX=/opt/librealsense \    
    -DBUILD_GRAPHICAL_EXAMPLES=OFF \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DBUILD_PYTHON_BINDINGS:bool=true \
    -DBUILD_WITH_CUDA=TRUE \
    -DCMAKE_BUILD_TYPE=Release \
    ../ && make -j$(($(nproc)-1)) all && make install

ENV PATH=/opt/librealsense:$PATH
ENV LIBRARY_PATH=/opt/librealsense/lib:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/librealsense/lib:$LD_LIBRARY_PATH

# Apply CMake fix for linux and build OpenNI2 wrapper
WORKDIR /usr/src/librealsense/wrappers/openni2
RUN sed -i -e 's|link_directories (${REALSENSE2_DIR}/lib/x64)|link_directories(${REALSENSE2_DIR}/lib)|g' \
    -e 's|link_directories (${REALSENSE2_DIR}/lib/x86)|link_directories(${REALSENSE2_DIR}/lib)|g' \
    -e 's|set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/_out)|set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${REALSENSE2_DIR}/lib)|g' \
    -e 's|set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/_out)|set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${REALSENSE2_DIR}/lib)|g' \
    -e 's|set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/_out)|set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${REALSENSE2_DIR}/bin)|g' \
    CMakeLists.txt
RUN cmake -DOPENNI2_DIR=/usr/include/openni2 -DREALSENSE2_DIR=/opt/librealsense . && make

######################################
#      AIWatch Base Image Stage      #
######################################
FROM ${BASE_IMAGE} AS aiwatch

# Copy binaries from builders stage
COPY --from=openpose-builder /opt/openpose /opt/openpose
COPY --from=librealsense-builder /opt/librealsense /opt/librealsense
COPY --from=librealsense-builder /usr/lib/python3/dist-packages/pyrealsense2 /usr/lib/python3/dist-packages/pyrealsense2
COPY --from=librealsense-builder /usr/src/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
COPY --from=librealsense-builder /usr/src/librealsense/config/99-realsense-d4xx-mipi-dfu.rules /etc/udev/rules.d/

# To avoid waiting for input during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dep packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \	
    libusb-1.0-0 \
    udev \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    libboost-all-dev \
    libhdf5-dev \
    libatlas-base-dev \
    libopenni2-dev \
    openni2-utils \
    libopencv-dev \
    freeglut3 \
    freeglut3-dev \
    python3 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Setup ENV variables
ENV PATH=opt/librealsense:/opt/openpose:$PATH
ENV LIBRARY_PATH=/opt/librealsense/lib:/opt/openpose/lib:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/librealsense/lib:/opt/openpose/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH="/opt/openpose/python:${PYTHONPATH}"

# Copy Realsense driver into OpenNI2
COPY --from=librealsense-builder /opt/librealsense/lib/librs2driver.so /usr/lib/OpenNI2/Drivers/
COPY --from=librealsense-builder /opt/librealsense/lib/librealsense2.so /usr/lib/OpenNI2/Drivers/

# Upgrade pip and install python dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install numpy opencv-python primesense facenet-pytorch scipy && \
    python3 -m pip install PyOpenGL PyOpenGL_accelerate tensorflow-gpu keras keyboard

# Copy the AIWatch application code
WORKDIR /app
COPY /app /app

# Run AIWatch application
CMD ["python3", "app.py", \
    "-vision_driver", "realsense", \
    "-openni2_dll_directories", "/usr/lib/", \
    "-openpose_model_path", "/app/models/", \
    "-hpe_model_root", "/app/HPE/models/", \
    "-fall_model_root", "/app/fall_detection/models/"]