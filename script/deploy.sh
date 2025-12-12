#!/usr/bin/env bash

repo_path=${HOME}/git/helexa/helexa
target_binary_path=/usr/local/bin
target_binary_perm=755
target_unit_path=/etc/systemd/system
target_unit_perm=644
target_spec_path=/usr/local/share/helexa
target_spec_perm=755
target_spec_file_perm=644
local_staging_path=/tmp/helexa
target_orchestration=${1}

# Check if the target is provided
if [ -z "${target_orchestration}" ]; then
    echo "Usage: $0 <target_orchestration>"
    exit 1
fi

if ! mkdir -p ${local_staging_path}; then
    echo "Failed to create local staging path ${local_staging_path}"
    exit 1
fi

_decode_property() {
    echo ${1} | base64 --decode | jq -r ${2}
}

_deploy_file() {
    local sync_source=${1}
    local target_name=${2}
    local target_hostname=${3}
    local target_ip=${4}
    local target_ssh_username=${5}
    local target_ssh_port=${6}
    local target_api_port=${7}
    local target_service_username=${8}
    local target_service_home=${9}
    local cortex_ip=${10}
    local cortex_ws_port=${11}
    local cortex_dash_ws_port=${12}
    local cortex_spec_path=${13}
    local file_type
    local configured_sync_source
    local sync_type
    local sync_target
    local deploy_command

    if [[ $(file --brief ${sync_source}) == *"ELF"* ]]; then
        file_type="binary"
    elif [[ "${sync_source}" == *.service ]]; then
        file_type="unit"
        configured_sync_source=${local_staging_path}/${target_name}-$(basename ${sync_source})
        HELEXA_CORTEX_USERNAME=${target_service_username} \
        HELEXA_CORTEX_HOME=${target_service_home} \
        HELEXA_NEURON_USERNAME=${target_service_username} \
        HELEXA_NEURON_HOME=${target_service_home} \
        HELEXA_CORTEX_HOST=${cortex_ip} \
        HELEXA_CORTEX_DASH_WS_PORT=${cortex_dash_ws_port} \
        HELEXA_CORTEX_WS_PORT=${cortex_ws_port} \
        HELEXA_CORTEX_API_PORT=${target_api_port} \
        HELEXA_NEURON_API_PORT=${target_api_port} \
        HELEXA_CORTEX_SPEC_PATH=${cortex_spec_path} \
        envsubst < ${sync_source} > ${configured_sync_source}
    elif [[ "${sync_source}" == *.json ]]; then
        file_type="spec"
    else
        file_type="unknown"
    fi

    if [ "${target_hostname}" = "$(hostname -s)" ]; then
        if [ "${file_type}" = "binary" ]; then
            sync_type="local ${file_type}"
            sync_target=${target_binary_path}/$(basename ${sync_source})
            deploy_command="sudo install --owner root --group root --mode ${target_binary_perm} ${sync_source} ${sync_target}"
        elif [ "${file_type}" = "unit" ]; then
            sync_type="local ${file_type}"
            sync_target=${target_unit_path}/$(basename ${sync_source})
            deploy_command="(id ${target_service_username} &>/dev/null || sudo useradd --system --create-home --home ${target_service_home} ${target_service_username}) && sudo install --owner root --group root --mode ${target_unit_perm} ${configured_sync_source} ${sync_target} && sudo systemctl daemon-reload && sudo systemctl enable $(basename ${sync_source}) && sudo systemctl restart $(basename ${sync_source})"
        elif [ "${file_type}" = "spec" ]; then
            sync_type="local ${file_type}"
            sync_target=${target_spec_path}/$(basename ${sync_source})
            deploy_command="sudo mkdir -p --mode ${target_spec_perm} ${target_spec_path} && sudo install --owner root --group root --mode ${target_spec_file_perm} ${sync_source} ${sync_target}"
        fi
    else
        if [ "${file_type}" = "binary" ]; then
            sync_type="remote ${file_type}"
            sync_target=${target_ssh_username}@${target_ip}:${target_binary_path}/$(basename ${sync_source})
            deploy_command="rsync --archive --compress --rsync-path 'sudo rsync' --rsh 'ssh -p ${target_ssh_port}' --chown root:root --chmod ${target_binary_perm} ${sync_source} ${sync_target}"
        elif [ "${file_type}" = "unit" ]; then
            sync_type="remote ${file_type}"
            sync_target=${target_ssh_username}@${target_ip}:${target_unit_path}/$(basename ${sync_source})
            deploy_command="ssh ${target_ssh_username}@${target_ip} '((systemctl is-active --quiet $(basename ${sync_source}) && sudo systemctl stop $(basename ${sync_source})) || true) && (id ${target_service_username} &>/dev/null || sudo useradd --system --create-home --home ${target_service_home} ${target_service_username})' && rsync --archive --compress --rsync-path 'sudo rsync' --rsh 'ssh -p ${target_ssh_port}' --chown root:root --chmod ${target_unit_perm} ${configured_sync_source} ${sync_target} && ssh ${target_ssh_username}@${target_ip} 'sudo systemctl daemon-reload && sudo systemctl enable $(basename ${sync_source}) && sudo systemctl start $(basename ${sync_source})'"
        elif [ "${file_type}" = "spec" ]; then
            sync_type="remote ${file_type}"
            sync_target=${target_ssh_username}@${target_ip}:${target_spec_path}/$(basename ${sync_source})
            deploy_command="ssh ${target_ssh_username}@${target_ip} 'sudo mkdir -p --mode ${target_spec_perm} ${target_spec_path}' && rsync --archive --compress --rsync-path 'sudo rsync' --rsh 'ssh -p ${target_ssh_port}' --chown root:root --chmod ${target_spec_file_perm} ${sync_source} ${sync_target}"
        fi
    fi
    if eval ${deploy_command}; then
        echo "    ${sync_type} install success"
        echo "      command: ${deploy_command}"
        echo "      source: ${sync_source}"
        echo "      target: ${sync_target}"
    else
        echo "    ${sync_type} install failure"
        echo "      command: ${deploy_command}"
        echo "      source: ${sync_source}"
        echo "      target: ${sync_target}"
        exit 1
    fi
    unset configured_sync_source
}

if ! cargo build --release --manifest-path ${repo_path}/Cargo.toml; then
    echo "Cargo build failed"
    exit 1
fi
if ! strip ${repo_path}/target/release/helexa; then
    echo "Strip failed"
    exit 1
fi

base64_cortex=($(yq \
    --arg orchestration ${target_orchestration} \
    --raw-output '
        .orchestrations[]
        | select(.name == $orchestration)
        | .cortex
        | @base64' \
    ${repo_path}/asset/env/.${target_orchestration}.yml))
declare -a base64_neurons=($(yq \
    --arg orchestration ${target_orchestration} \
    --raw-output '
        .orchestrations[]
        | select(.name == $orchestration)
        | .neurons[]
        | @base64' \
    ${repo_path}/asset/env/.${target_orchestration}.yml))
neuron_count=${#base64_neurons[@]}
echo "orchestration: ${target_orchestration} (${neuron_count} neurons)"

# deploy one cortex
cortex_name=$(_decode_property ${base64_cortex} .name)
cortex_hostname=$(_decode_property ${base64_cortex} .hostname)
cortex_ip=$(_decode_property ${base64_cortex} .ip)
cortex_ws_port=$(_decode_property ${base64_cortex} .ws.port)
cortex_dash_ws_port=$(_decode_property ${base64_cortex} .dash.ws.port)
cortex_ssh_username=$(_decode_property ${base64_cortex} .superuser.username)
cortex_ssh_port=$(_decode_property ${base64_cortex} .ssh.port)
cortex_api_port=$(_decode_property ${base64_cortex} .api.port)
cortex_service_username=$(_decode_property ${base64_cortex} .helexa.username)
cortex_service_home=$(_decode_property ${base64_cortex} .helexa.home)
echo "  cortex: ${cortex_name} (${cortex_ip})"
for sync_source in ${repo_path}/target/release/helexa ${repo_path}/asset/spec/default.json ${repo_path}/asset/systemd/helexa-cortex.service; do
    _deploy_file \
        ${sync_source} \
        ${cortex_name} \
        ${cortex_hostname} \
        ${cortex_ip} \
        ${cortex_ssh_username} \
        ${cortex_ssh_port} \
        ${cortex_api_port} \
        ${cortex_service_username} \
        ${cortex_service_home} \
        ${cortex_ip} \
        ${cortex_ws_port} \
        ${cortex_dash_ws_port} \
        ${target_spec_path}/default.json
done

# deploy all neurons
for base64_neuron in ${base64_neurons[@]}; do
    neuron_name=$(_decode_property ${base64_neuron} .name)
    neuron_hostname=$(_decode_property ${base64_neuron} .hostname)
    neuron_ip=$(_decode_property ${base64_neuron} .ip)
    neuron_ssh_username=$(_decode_property ${base64_neuron} .superuser.username)
    neuron_ssh_port=$(_decode_property ${base64_neuron} .ssh.port)
    neuron_api_port=$(_decode_property ${base64_neuron} .api.port)
    neuron_service_username=$(_decode_property ${base64_neuron} .helexa.username)
    neuron_service_home=$(_decode_property ${base64_neuron} .helexa.home)
    echo "  neuron: ${neuron_name} (${neuron_ip})"
    for sync_source in ${repo_path}/target/release/helexa ${repo_path}/asset/systemd/helexa-neuron.service; do
        _deploy_file \
            ${sync_source} \
            ${neuron_name} \
            ${neuron_hostname} \
            ${neuron_ip} \
            ${neuron_ssh_username} \
            ${neuron_ssh_port} \
            ${neuron_api_port} \
            ${neuron_service_username} \
            ${neuron_service_home} \
            ${cortex_ip} \
            ${cortex_ws_port} \
            ${cortex_dash_ws_port} \
            ""
    done
done
