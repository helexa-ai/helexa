#!/usr/bin/env bash
#
# One-time setup for the gitea_ci deploy-user on every host that the
# .gitea/workflows/deploy.yml workflow targets:
#   - create the gitea_ci system user (if missing)
#   - install the runner's pubkey into ~gitea_ci/.ssh/authorized_keys
#   - install the appropriate /etc/sudoers.d/helexa_gitea_ci sudoers
#     drop-in (cortex flavour on the gateway, neuron flavour on each
#     neuron host)
#
# Idempotent — safe to re-run after fleet changes. Continues past
# unreachable hosts so a single offline node doesn't block the rest.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_path="$(cd "${script_dir}/.." && pwd)"

cortex_host=hanzalova.internal
neuron_hosts=(
    beast.hanzalova.internal
    benjy.hanzalova.internal
    quadbrat.hanzalova.internal
)

pubkey="${HOME}/.ssh/id_gitea_ci.pub"
if [[ ! -f "${pubkey}" ]]; then
    echo "fatal: ${pubkey} not found" >&2
    echo "  generate with: ssh-keygen -t ed25519 -f ${pubkey%.pub} -C gitea_ci" >&2
    exit 1
fi

# Provision gitea_ci on every host (cortex + all neurons).
#
# Quoting matters here: "${cortex_host} ${neuron_hosts[@]}" inside a
# single pair of quotes collapses the scalar and the first array
# element into one space-joined word, which then word-splits when
# referenced unquoted in `ssh ${host}` — and ssh interprets the second
# hostname as the remote command. Separate quoting fixes it.
for host in "${cortex_host}" "${neuron_hosts[@]}"; do
    echo "==> ${host}"
    if ! ssh "${host}" '
        set -eu
        if id -u gitea_ci >/dev/null 2>&1; then
            echo "  gitea_ci user already present"
        else
            sudo useradd --system --create-home \
                --home-dir /var/lib/gitea_ci --shell /bin/bash gitea_ci
            echo "  gitea_ci user created"
        fi
        # `sudo install` runs as root (not as gitea_ci), which avoids
        # the "sudo: unknown user gitea_ci" failure seen immediately
        # after useradd — NSS caching lags briefly and `sudo -u` cant
        # resolve the just-created user, but `install -o` does its
        # own fresh lookup.
        sudo install -d -o gitea_ci -g gitea_ci -m 0700 \
            /var/lib/gitea_ci/.ssh
    '; then
        echo "  failed to provision gitea_ci — skipping ${host}"
        continue
    fi

    if rsync \
        --archive \
        --compress \
        --chown gitea_ci:gitea_ci \
        --chmod 0600 \
        --rsync-path 'sudo rsync' \
        "${pubkey}" \
        "${host}:/var/lib/gitea_ci/.ssh/authorized_keys"; then
        echo "  authorized_keys synced"
    else
        echo "  failed to sync authorized_keys"
    fi
done

# Install /etc/sudoers.d/helexa_gitea_ci on a host and verify the
# resulting file parses, so a typo cant lock root out.
install_sudoers() {
    local host="$1" template="$2"
    echo "==> ${host}: installing /etc/sudoers.d/helexa_gitea_ci"
    if ! rsync \
        --archive \
        --compress \
        --chown root:root \
        --chmod 0440 \
        --rsync-path 'sudo rsync' \
        "${template}" \
        "${host}:/etc/sudoers.d/helexa_gitea_ci"; then
        echo "  failed to sync ${template##*/}"
        return
    fi
    if ssh "${host}" 'sudo visudo -cf /etc/sudoers.d/helexa_gitea_ci' \
            >/dev/null; then
        echo "  installed and verified"
    else
        echo "  WARNING: visudo rejected the installed file — review on ${host}"
    fi
}

install_sudoers "${cortex_host}" \
    "${repo_path}/asset/sudoers.d/cortex-host.conf"

for neuron_host in "${neuron_hosts[@]}"; do
    install_sudoers "${neuron_host}" \
        "${repo_path}/asset/sudoers.d/neuron-host.conf"
done

# Push application config to the fleet. The deploy workflow is
# scoped to package install + service restart; config changes ride
# along with this script instead, since:
#   - cortex.toml and models.toml are gitignored (operator-owned, may
#     include secrets), so CI never sees them
#   - asset/neuron/<short>.toml is tracked but iterating locally is
#     faster than pushing a commit and waiting for build-prerelease
#     to roll over
# Missing source files are skipped silently — re-run after editing.
sync_config() {
    local host="$1" src="$2" dst="$3"
    if [[ ! -f "${src}" ]]; then
        echo "  ${src##*/} not present locally — skipping"
        return
    fi
    if rsync \
        --archive \
        --compress \
        --chown root:root \
        --chmod 0644 \
        --rsync-path 'sudo rsync' \
        "${src}" \
        "${host}:${dst}"; then
        echo "  ${src##*/} → ${host}:${dst}"
    else
        echo "  failed to sync ${src##*/} to ${host}"
    fi
}

echo "==> ${cortex_host}: syncing gateway configs"
sync_config "${cortex_host}" "${repo_path}/cortex.toml" /etc/cortex/cortex.toml
sync_config "${cortex_host}" "${repo_path}/models.toml" /etc/cortex/models.toml

for neuron_host in "${neuron_hosts[@]}"; do
    short="${neuron_host%%.*}"
    echo "==> ${neuron_host}: syncing per-host neuron config"
    sync_config "${neuron_host}" \
        "${repo_path}/asset/neuron/${short}.toml" \
        /etc/neuron/neuron.toml
done
