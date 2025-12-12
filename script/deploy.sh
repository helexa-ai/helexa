#!/usr/bin/env bash

repo_path=${HOME}/git/helexa/helexa
target_binary_path=/usr/local/bin
target_binary_perm=755
target_unit_path=/etc/systemd/system
target_unit_perm=644
target_orchestration=${1}

# Check if the target is provided
if [ -z "${target_orchestration}" ]; then
    echo "Usage: $0 <target_orchestration>"
    exit 1
fi

_decode_property() {
    echo ${1} | base64 --decode | jq -r ${2}
}

if ! cargo build --release --manifest-path ${repo_path}/Cargo.toml; then
    echo "Cargo build failed"
    exit 1
fi
if ! strip ${repo_path}/target/release/helexa; then
    echo "Strip failed"
    exit 1
fi

cortex_ip=$(yq \
    --arg orchestration ${target_orchestration} \
    --raw-output '
        .orchestrations[]
        | select(.name == $orchestration)
        | .cortex.ip' \
    ${repo_path}/asset/env/.${target_orchestration}.yml)
cortex_ws_port=$(yq \
    --arg orchestration ${target_orchestration} \
    --raw-output '
        .orchestrations[]
        | select(.name == $orchestration)
        | .cortex.ws.port' \
    ${repo_path}/asset/env/.${target_orchestration}.yml)
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
for base64_neuron in ${base64_neurons[@]}; do
    neuron_name=$(_decode_property ${base64_neuron} .name)
    neuron_hostname=$(_decode_property ${base64_neuron} .hostname)
    neuron_ip=$(_decode_property ${base64_neuron} .ip)
    neuron_ssh_username=$(_decode_property ${base64_neuron} .superuser.username)
    neuron_ssh_port=$(_decode_property ${base64_neuron} .ssh.port)
    neuron_api_port=$(_decode_property ${base64_neuron} .api.port)
    echo "  neuron: ${neuron_name} (${neuron_ip})"
    for sync_source in ${repo_path}/target/release/helexa ${repo_path}/asset/systemd/helexa-neuron.service; do
        if [ "${neuron_hostname}" = "$(hostname -s)" ]; then
            if [[ $(file --brief ${sync_source}) == *"ELF"* ]]; then
                sync_type="local binary"
                sync_target=${target_binary_path}/$(basename ${sync_source})
                deploy_command="sudo install --owner root --group root --mode ${target_binary_perm} ${sync_source} ${sync_target}"
            elif [[ "${sync_source}" == *.service ]]; then
                sync_type="local unit"
                sync_target=${target_unit_path}/$(basename ${sync_source})
                configured_sync_source=/tmp/${neuron_name}-$(basename ${sync_source})
                HELEXA_CORTEX_HOST=${cortex_ip} HELEXA_CORTEX_WS_PORT=${cortex_ws_port} HELEXA_NEURON_API_PORT=${neuron_api_port} envsubst < ${sync_source} > ${configured_sync_source}
                deploy_command="sudo install --owner root --group root --mode ${target_unit_perm} ${configured_sync_source} ${sync_target}"
            fi
        else
            if [[ $(file --brief ${sync_source}) == *"ELF"* ]]; then
                sync_type="remote binary"
                sync_target=${neuron_ssh_username}@${neuron_ip}:${target_binary_path}/$(basename ${sync_source})
                deploy_command="rsync --archive --compress --rsync-path 'sudo rsync' --rsh 'ssh -p ${neuron_ssh_port}' --chown root:root --chmod ${target_binary_perm} ${sync_source} ${sync_target}"
            elif [[ "${sync_source}" == *.service ]]; then
                sync_type="remote unit"
                sync_target=${neuron_ssh_username}@${neuron_ip}:${target_unit_path}/$(basename ${sync_source})
                configured_sync_source=/tmp/${neuron_name}-$(basename ${sync_source})
                HELEXA_CORTEX_HOST=${cortex_ip} HELEXA_CORTEX_WS_PORT=${cortex_ws_port} envsubst < ${sync_source} > ${configured_sync_source}
                deploy_command="rsync --archive --compress --rsync-path 'sudo rsync' --rsh 'ssh -p ${neuron_ssh_port}' --chown root:root --chmod ${target_unit_perm} ${configured_sync_source} ${sync_target}"
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
    done
done
