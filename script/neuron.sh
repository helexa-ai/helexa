#!/usr/bin/env bash

repo_path=${HOME}/git/helexa/helexa

${repo_path}/target/debug/helexa neuron \
    --control-socket 0.0.0.0:9050 \
    --api-socket     127.0.0.1:8060 \
    --cortex-control-endpoint ws://127.0.0.1:9040
