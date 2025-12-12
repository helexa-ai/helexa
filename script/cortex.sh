#!/usr/bin/env bash

repo_path=${HOME}/git/helexa/helexa
binary_path=${repo_path}/target/release/helexa

${binary_path} cortex \
    --dashboard-socket 0.0.0.0:8090 \
    --control-plane-socket 0.0.0.0:9040 \
    --orchestrator-socket 0.0.0.0:8040 \
    --gateway-socket 0.0.0.0:8080 \
    --spec ${repo_path}/asset/spec/default.json
