#! /usr/bin/bash
set -e
set -x

#docker build --rm \
#    --build-arg BASE_IMAGE=ubuntu:24.04 \
#    -t erl/ubuntu-desktop:24.04 $@ .

docker build --rm \
    --build-arg BASE_IMAGE=nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 \
    -t erl/ubuntu-desktop:24.04-cuda $@ .
