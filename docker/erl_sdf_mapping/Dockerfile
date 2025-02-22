ARG BASE_IMAGE=erl/ubuntu-desktop:20.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# intel oneAPI
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \ | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list

RUN apt update && apt upgrade -y

RUN apt install -y intel-basekit-2024.1

RUN apt install -y ssh locales-all dos2unix rsync tar
RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'Port 8022'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'X11Forwarding yes'; \
    echo 'X11DisplayOffset 10'; \
    echo 'X11UseLocalhost yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config.d/sshd_config_test_clion.conf \
  && mkdir -p /run/sshd && mkdir -p /var/run/sshd && ssh-keygen -A
EXPOSE 8022

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
    python3-dev \
    python3-pip \
    python3-setuptools \
    software-properties-common \
    lsb-release \
    python3-pytest \
    python3-numpy \
    python3-scipy \
    libeigen3-dev \
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
    libwayland-dev \
    wayland-protocols \
    libxkbcommon-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libglu1-mesa-dev \
    libsdl2-dev \
    libxcb-shm0 \
    xorg-dev \
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
    libcgal-dev \
    libfmt-dev \
    libnanoflann-dev \
    libabsl-dev \
    libqhull-dev \
    libjsoncpp-dev

WORKDIR /root

RUN git clone https://github.com/stevenlovegrove/Pangolin.git && \
    cd Pangolin && git checkout v0.9.3 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j`nproc` && make install && \
    cd /root && rm -rf Pangolin

COPY ./open3d-0.19.0.patch /root
RUN git clone --recursive https://github.com/isl-org/Open3D.git && cd Open3D && \
    git checkout v0.19.0 && \
    echo y | util/install_deps_ubuntu.sh && \
    git apply /root/open3d-0.19.0.patch && \
    mkdir build && cd build && \
    cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release \
        -DUSE_SYSTEM_EIGEN3=ON \
        -DUSE_SYSTEM_FMT=ON \
        -DUSE_SYSTEM_GOOGLETEST=ON \
        -DUSE_SYSTEM_JSONCPP=ON \
        -DUSE_SYSTEM_NANOFLANN=ON \
        -DUSE_SYSTEM_PYBIND11=ON \
        -DUSE_SYSTEM_QHULLCPP=ON \
        -DGLIBCXX_USE_CXX11_ABI=ON \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_UNIT_TESTS=OFF \
        -DBUILD_BENCHMARKS=OFF && \
    make -j`nproc` && make install

# clean
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# reset DEBIAN_FRONTEND
ENV DEBIAN_FRONTEND=""

RUN userdel ubuntu && rm -rf /home/ubuntu

CMD ["/usr/sbin/sshd", "-D", "-e"]
