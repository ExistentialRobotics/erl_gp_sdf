ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

COPY ./sources.list /etc/apt

ENV TZ="America/Los_Angeles"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /root
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y && apt install -y ubuntu-desktop-minimal  # time-consuming
RUN apt upgrade -y && apt install -y \
    software-properties-common lsb-release wget vim locales apt-utils build-essential \
    python3 python3-pip python3-dev python3-setuptools python3-wheel git \
    sudo zsh ca-certificates curl iproute2 libpcre3-dev libpcre2-dev libtinyxml2-dev

# set locale
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# clean
RUN apt-get clean  && rm -rf /var/lib/apt/lists/*

# reset DEBIAN_FRONTEND
ENV DEBIAN_FRONTEND=""
