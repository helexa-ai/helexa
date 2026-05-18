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

# Return the installed package's "version-release" string, or
# "(not installed)" when rpm reports the package as absent. Capture
# rpm's output into a variable so its "package X is not installed"
# stdout message (rpm writes that to stdout, not stderr, when -q fails)
# doesn't leak into the result.
installed_nvr() {
    local host="$1" pkg="$2"
    local nvr
    if nvr=$(ssh "${host}" "rpm -q --qf '%{version}-%{release}' ${pkg} 2>/dev/null"); then
        echo "${nvr}"
    else
        echo "(not installed)"
    fi
}

# Ensure the rpm.lair.cafe unstable repo is configured AND enabled on
# the remote host.
#
# The upstream .repo file at https://rpm.lair.cafe/lair-cafe-unstable.repo
# ships with `enabled=0` so a host that just fetched it won't start
# pulling unstable packages by accident. We have to explicitly flip
# enabled=1 via `dnf config-manager setopt`. Both addrepo and setopt
# are idempotent.
#
# Non-fatal — if either step fails the subsequent `dnf install` will
# surface a clearer diagnostic on its own.
ensure_lair_repo() {
    local host="$1"
    if ! ssh "${host}" "test -f /etc/yum.repos.d/lair-cafe-unstable.repo" 2>/dev/null; then
        echo "[${host}] adding rpm.lair.cafe unstable repo"
        if ! ssh "${host}" sudo dnf config-manager addrepo \
            --from-repofile=https://rpm.lair.cafe/lair-cafe-unstable.repo \
            >/dev/null 2>&1; then
            echo "[${host}] WARNING: failed to add lair.cafe repo file (proceeding anyway)"
            return 0
        fi
    fi
    # The .repo file ships enabled=0; flip it on. Cheap, idempotent.
    if ! ssh "${host}" sudo dnf config-manager setopt \
        lair-cafe-unstable.enabled=1 >/dev/null 2>&1; then
        echo "[${host}] WARNING: failed to enable lair-cafe-unstable (proceeding anyway)"
    fi
}

# True when the named package needs to be installed or upgraded on the
# remote host — either it's not present, or a newer version exists in
# the repo. False only when the installed version is current.
#
# `dnf check-update <pkg>` returns 0 when the package isn't installed
# at all (there's nothing to update), so we have to probe with rpm -q
# first to distinguish "absent" from "current". Other dnf failures
# collapse into "needs update" so the subsequent install step surfaces
# the real diagnostic rather than this check swallowing it.
needs_update() {
    local host="$1" pkg="$2"
    # Not installed → needs work.
    if ! ssh "${host}" "rpm -q ${pkg}" >/dev/null 2>&1; then
        return 0
    fi
    # Installed; ask dnf whether the repo has something newer.
    if ssh "${host}" sudo dnf check-update --refresh -q "${pkg}" >/dev/null 2>&1; then
        return 1
    else
        return 0
    fi
}

# ---------------------------------------------------------------------------
# cortex (gateway)
# ---------------------------------------------------------------------------

ensure_lair_repo "${cortex_host}"
cortex_nvr=$(installed_nvr "${cortex_host}" cortex)
if needs_update "${cortex_host}" cortex; then
    echo "[${cortex_host}] cortex update available (current: ${cortex_nvr})"
    # Stop the service only if the unit file exists — fresh installs
    # don't have it, and `systemctl stop` on a missing unit returns
    # non-zero, which would otherwise short-circuit the install branch
    # under set -e.
    if ssh "${cortex_host}" "[ ! -f /usr/lib/systemd/system/cortex.service ] || sudo systemctl stop cortex.service"; then
        echo "[${cortex_host}] stopped cortex service"
        if dnf_output=$(ssh "${cortex_host}" sudo dnf install --refresh --allowerasing -y cortex 2>&1); then
            cortex_nvr=$(installed_nvr "${cortex_host}" cortex)
            echo "[${cortex_host}] installed/upgraded cortex to ${cortex_nvr}"
        else
            echo "[${cortex_host}] failed to install/upgrade cortex:"
            echo "${dnf_output}" | sed "s/^/[${cortex_host}]   /"
        fi
    else
        echo "[${cortex_host}] failed to stop cortex service"
    fi
else
    echo "[${cortex_host}] cortex is up to date (${cortex_nvr})"
    ssh "${cortex_host}" sudo systemctl stop cortex.service || true
fi

# Sync cortex.toml whether the package was upgraded or not — the config
# can change without a package bump.
if rsync \
    --archive \
    --compress \
    --rsync-path 'sudo rsync' \
    --chown root:root \
    --chmod 644 \
    "${REPO_DIR}/cortex.toml" \
    "${cortex_host}:/etc/cortex/cortex.toml"; then
    echo "[${cortex_host}] sync'd cortex.toml"
else
    echo "[${cortex_host}] failed to sync cortex.toml"
fi

ssh "${cortex_host}" sudo systemctl daemon-reload
if ssh "${cortex_host}" systemctl is-active --quiet cortex.service; then
    echo "[${cortex_host}] cortex service is active"
elif ssh "${cortex_host}" sudo systemctl start cortex.service; then
    echo "[${cortex_host}] started cortex service"
else
    echo "[${cortex_host}] failed to start cortex service"
fi

# ---------------------------------------------------------------------------
# neuron (per-host, flavour from manifest)
# ---------------------------------------------------------------------------

for entry in "${neuron_entries[@]}"; do
    IFS=$'\t' read -r neuron_host neuron_flavour <<< "${entry}"
    package="helexa-neuron-${neuron_flavour}"

    ensure_lair_repo "${neuron_host}"
    neuron_nvr=$(installed_nvr "${neuron_host}" "${package}")
    if needs_update "${neuron_host}" "${package}"; then
        echo "[${neuron_host}] ${package} update available (current: ${neuron_nvr})"
        if ssh "${neuron_host}" "[ ! -f /usr/lib/systemd/system/neuron.service ] || sudo systemctl stop neuron.service"; then
            echo "[${neuron_host}] stopped neuron service"
            # --allowerasing lets dnf swap out a previously-installed
            # bare helexa-neuron or a different flavour without manual
            # intervention. The Conflicts: clauses in the spec ensure
            # only one flavour is ever resident.
            if dnf_output=$(ssh "${neuron_host}" sudo dnf install --refresh --allowerasing -y "${package}" 2>&1); then
                neuron_nvr=$(installed_nvr "${neuron_host}" "${package}")
                echo "[${neuron_host}] installed/upgraded ${package} to ${neuron_nvr}"
                # Ensure firewalld allows neuron port
                ssh "${neuron_host}" "sudo firewall-cmd --query-service=helexa-neuron --quiet 2>/dev/null || sudo firewall-cmd --add-service=helexa-neuron --permanent && sudo firewall-cmd --reload" 2>/dev/null || true
                if ssh "${neuron_host}" "sudo systemctl daemon-reload && sudo systemctl start neuron.service"; then
                    echo "[${neuron_host}] started neuron service"
                else
                    echo "[${neuron_host}] failed to start neuron service"
                fi
            else
                echo "[${neuron_host}] failed to install ${package}:"
                echo "${dnf_output}" | sed "s/^/[${neuron_host}]   /"
            fi
        else
            echo "[${neuron_host}] failed to stop neuron service"
        fi
    else
        echo "[${neuron_host}] ${package} is up to date (${neuron_nvr})"
        if ssh "${neuron_host}" systemctl is-active --quiet neuron.service; then
            echo "[${neuron_host}] neuron service is active"
        elif ssh "${neuron_host}" sudo systemctl start neuron.service; then
            echo "[${neuron_host}] started neuron service"
        else
            echo "[${neuron_host}] failed to start neuron service"
        fi
    fi
done
