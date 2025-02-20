#! /usr/bin/bash
set -e
set -x

#docker build --rm \
#    --build-arg BASE_IMAGE=erl/ubuntu-desktop:24.04 \
#    -t erl/erl_sdf_mapping $@ .

docker build --rm \
    --build-arg BASE_IMAGE=erl/ubuntu-desktop:24.04-cuda \
    -t erl/erl_sdf_mapping:cuda $@ .
