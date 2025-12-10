#!/usr/bin/env bash

repo_path=${HOME}/git/helexa/helexa

${repo_path}/target/debug/helexa cortex \
    --dashboard-socket 0.0.0.0:8090 \
    --control-plane-socket 0.0.0.0:9040 \
    --orchestrator-socket 0.0.0.0:8040 \
    --gateway-socket 0.0.0.0:8080 \
    --spec spec/default.json
