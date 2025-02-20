ARG BASE_IMAGE=erl/ubuntu-desktop:20.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# intel oneAPI
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \ | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list

RUN apt update && apt upgrade -y

RUN apt install -y intel-basekit-2024.1

RUN apt install -y \
    git \
    wget \
    curl \
    gcc \
    g++ \
    cmake \
    build-essential \
    ccache \
    python3 \
    python3-pip \
    python3-setuptools \
    software-properties-common \
    lsb-release \
    python3-pytest \
    python3-numpy \
    python3-scipy \
    libgtest-dev \
    liblapacke-dev \
    liblapack-dev \
    libomp-dev \
    libboost-all-dev \
    nlohmann-json3-dev \
    libopencv-dev \
    libpcl-dev \
    libyaml-cpp-dev \
    pybind11-dev \
    libepoxy-dev \
    libgl1-mesa-dev \
    libwayland-dev \
    libxkbcommon-dev \
    wayland-protocols \
    libegl1-mesa-dev \
    libc++-dev \
    libc++abi-dev \
    libglew-dev \
    ninja-build \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    xinput \
    libxi-dev \
    libtbb-dev \
    libudev-dev \
    autoconf \
    libtool \
    libcgal-dev

WORKDIR /root

RUN git clone https://gitlab.com/libeigen/eigen.git && \
    cd eigen && git checkout f679843dc2792349482af0c02e1daf7032d0c791 && \
    mkdir build && cd build && \
    cmake .. && cmake --build . --target install -j $(nproc) && \
    cd /root && rm -rf eigen

RUN git clone https://github.com/pybind/pybind11.git && \
    cd pybind11 && git checkout v2.13.4 && \
    mkdir build && cd build && \
    cmake .. && cmake --build . --target install -j $(nproc) && \
    cd /root && rm -rf pybind11

RUN git clone https://github.com/fmtlib/fmt.git &&  \
    cd fmt && git checkout 11.1.3 && \
    mkdir build && cd build && \
    cmake .. -DFMT_DOC=OFF -DFMT_TEST=OFF -DFMT_INSTALL=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON && \
    make -j`nproc` && make install && cd /root && rm -rf fmt

RUN git clone https://github.com/jlblancoc/nanoflann.git && \
    cd nanoflann && git checkout 518b2b97e484715d757c2960aefb26a25d0b7aab && \
    mkdir build && cd build && \
    cmake .. && cmake --build . --target install -j $(nproc) && \
    cd /root && rm -rf nanoflann

RUN wget https://github.com/abseil/abseil-cpp/releases/download/20240722.1/abseil-cpp-20240722.1.tar.gz && \
    tar -xf abseil-cpp-20240722.1.tar.gz && rm abseil-cpp-20240722.1.tar.gz && \
    cd abseil-cpp-20240722.1 && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="${CXXFLAGS} -DNDEBUG" \
        -DCMAKE_CXX_STANDARD=17 -DBUILD_SHARED_LIBS=ON -DABSL_PROPAGATE_CXX_STD=ON && \
    make -j`nproc` && make install && \
    cd /root && rm -rf abseil-cpp-20240722.1

RUN wget https://github.com/stevenlovegrove/Pangolin/archive/refs/tags/v0.9.2.tar.gz && \
    tar -xf v0.9.2.tar.gz && rm v0.9.2.tar.gz && \
    cd Pangolin-0.9.2 && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j`nproc` && make install && \
    cd /root && rm -rf Pangolin-0.9.2

RUN git clone --recursive https://github.com/qhull/qhull.git && \
    cd qhull && git checkout a22c735d6a8d1b5eac5773790aeae28f3b088655 && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_C_FLAGS="${CFLAGS} -ffat-lto-objects" -DCMAKE_CXX_FLAGS="${CXXFLAGS} -ffat-lto-objects" -DCMAKE_BUILD_TYPE=Release && \
    make -j`nproc` && make install && \
    cd /root && rm -rf qhull

RUN git clone --recursive https://github.com/isl-org/Open3D.git && cd Open3D && \
    git checkout 1e7b17438687a0b0c1e5a7187321ac7044afe275 && mkdir build && cd build && \
    cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release && \
    make -j`nproc` && make install && cd /root && rm -rf Open3D

# clean
RUN apt-get clean  && rm -rf /var/lib/apt/lists/*
# reset DEBIAN_FRONTEND
ENV DEBIAN_FRONTEND=""
