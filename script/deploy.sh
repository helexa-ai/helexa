#!/bin/env bash
#
# Rolling deploy across the helexa fleet, driven by asset/manifest.yml.
# Installs / upgrades cortex on the gateway host and the appropriate
# helexa-neuron-<flavour> package on each neuron host, then restarts
# their services.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST="${REPO_DIR}/asset/manifest.yml"

if [[ ! -f "${MANIFEST}" ]]; then
    echo "fatal: manifest not found at ${MANIFEST}" >&2
    exit 1
fi

# Parse the manifest with yq. NOTE: this expects the pip-installed yq
# (a jq wrapper using jq syntax) — `pip install yq`. The Fedora rpm
# `yq` is mikefarah/yq and uses different (yaml-native) syntax; if a
# host has that one instead these queries will fail.
cortex_host=$(yq -r '.cortex.host' "${MANIFEST}")

# Emit one TAB-separated 'host\tflavour' line per neuron.
mapfile -t neuron_entries < <(
    yq -r '.neurons[] | .host + "\t" + .flavour' "${MANIFEST}"
)

latest_helexa_version=$(git -C "${REPO_DIR}" describe --tags --abbrev=0 | sed 's/^v//')

# ---------------------------------------------------------------------------
# cortex (gateway)
# ---------------------------------------------------------------------------

observed_cortex_version=$(ssh "${cortex_host}" cortex --version | sed 's/^cortex //')
if [[ "${latest_helexa_version}" = "${observed_cortex_version}" ]]; then
    echo "[${cortex_host}] cortex is up to date (${observed_cortex_version})"
    if ssh "${cortex_host}" sudo systemctl stop cortex.service && rsync \
        --archive \
        --compress \
        --rsync-path 'sudo rsync' \
        --chown root:root \
        --chmod 644 \
        "${REPO_DIR}/cortex.toml" \
        "${cortex_host}:/etc/cortex/cortex.toml"; then
        echo "[${cortex_host}] sync'd cortex.toml"
        ssh "${cortex_host}" sudo systemctl daemon-reload
        ssh "${cortex_host}" sudo systemctl start cortex.service
    else
        echo "[${cortex_host}] failed to sync cortex.toml"
    fi
    if ssh "${cortex_host}" systemctl is-active --quiet cortex.service; then
        echo "[${cortex_host}] cortex service is active"
    elif ssh "${cortex_host}" sudo systemctl start cortex.service; then
        echo "[${cortex_host}] started cortex service"
    else
        echo "[${cortex_host}] failed to start cortex service"
    fi
else
    echo "[${cortex_host}] cortex is out of date (${observed_cortex_version} != ${latest_helexa_version})"
    if ssh "${cortex_host}" sudo systemctl stop cortex.service; then
        echo "[${cortex_host}] stopped cortex service"
        if ssh "${cortex_host}" sudo dnf upgrade --refresh -y cortex; then
            echo "[${cortex_host}] upgraded cortex"
            if rsync \
                --archive \
                --compress \
                --verbose \
                --rsync-path 'sudo rsync' \
                --chown root:root \
                --chmod 644 \
                "${REPO_DIR}/cortex.toml" \
                "${cortex_host}:/etc/cortex/cortex.toml"; then
                echo "[${cortex_host}] sync'd cortex.toml"
                ssh "${cortex_host}" sudo systemctl daemon-reload
                ssh "${cortex_host}" sudo systemctl start cortex.service
            else
                echo "[${cortex_host}] failed to sync cortex.toml"
            fi
        else
            echo "[${cortex_host}] failed to upgrade cortex"
        fi
    else
        echo "[${cortex_host}] failed to stop cortex service"
    fi
fi

# ---------------------------------------------------------------------------
# neuron (per-host, flavour from manifest)
# ---------------------------------------------------------------------------

for entry in "${neuron_entries[@]}"; do
    IFS=$'\t' read -r neuron_host neuron_flavour <<< "${entry}"
    package="helexa-neuron-${neuron_flavour}"

    observed_neuron_version=$(ssh "${neuron_host}" neuron --version 2> /dev/null | sed 's/^neuron //' || true)
    if [[ "${latest_helexa_version}" = "${observed_neuron_version}" ]]; then
        echo "[${neuron_host}] neuron is up to date (${observed_neuron_version}, ${package})"
        if ssh "${neuron_host}" systemctl is-active --quiet neuron.service; then
            echo "[${neuron_host}] neuron service is active"
        elif ssh "${neuron_host}" sudo systemctl start neuron.service; then
            echo "[${neuron_host}] started neuron service"
        else
            echo "[${neuron_host}] failed to start neuron service"
        fi
    else
        echo "[${neuron_host}] upgrading neuron from ${observed_neuron_version:-(absent)} to ${latest_helexa_version} (${package})"
        if ssh "${neuron_host}" "[ ! -f /usr/lib/systemd/system/neuron.service ] || sudo systemctl stop neuron.service"; then
            echo "[${neuron_host}] stopped neuron service"
            # --allowerasing lets dnf swap out a previously-installed
            # bare helexa-neuron or a different flavour without manual
            # intervention. The Conflicts: clauses in the spec ensure
            # only one flavour is ever resident.
            if ssh "${neuron_host}" sudo dnf install --refresh --allowerasing -y "${package}" &> /dev/null; then
                echo "[${neuron_host}] installed/upgraded ${package}"
                # Ensure firewalld allows neuron port
                ssh "${neuron_host}" "sudo firewall-cmd --query-service=helexa-neuron --quiet 2>/dev/null || sudo firewall-cmd --add-service=helexa-neuron --permanent && sudo firewall-cmd --reload" 2>/dev/null || true
                if ssh "${neuron_host}" "sudo systemctl daemon-reload && sudo systemctl start neuron.service"; then
                    echo "[${neuron_host}] started neuron service"
                else
                    echo "[${neuron_host}] failed to start neuron service"
                fi
            else
                echo "[${neuron_host}] failed to install ${package}"
            fi
        else
            echo "[${neuron_host}] failed to stop neuron service"
        fi
    fi
done
