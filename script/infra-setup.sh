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
# Bench host: runs helexa-bench (outbound-only; polls the neuron fleet).
# Also runs Agent Zero — it's a client host, not a GPU node.
bench_host=bob.hanzalova.internal

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
for host in "${cortex_host}" "${neuron_hosts[@]}" "${bench_host}"; do
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
        # Grant journal read access so the deploy workflow can capture
        # `journalctl -u <unit> -I` after a service start without
        # needing a sudoers entry. Idempotent — usermod -aG on an
        # already-member is a no-op.
        sudo usermod -aG systemd-journal gitea_ci
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

install_sudoers "${bench_host}" \
    "${repo_path}/asset/sudoers.d/bench-host.conf"

# bob doesn't otherwise carry the lair-cafe RPM repo (it's a client
# host, not a cortex/neuron node), so helexa-bench's `dnf install` in
# deploy.yml would have nothing to install from. Enable the unstable
# repo here, and pre-create /etc/helexa-bench so the config sync below
# lands even before the first package install. Idempotent.
echo "==> ${bench_host}: ensuring lair-cafe-unstable repo + config dir"
if ! ssh "${bench_host}" '
    set -eu
    if dnf repolist --all 2>/dev/null | grep -q "^lair-cafe-unstable"; then
        echo "  lair-cafe-unstable already present"
    else
        sudo dnf config-manager addrepo --from-repofile=https://rpm.lair.cafe/lair-cafe-unstable.repo
        sudo dnf config-manager setopt lair-cafe-unstable.enabled=1
        echo "  lair-cafe-unstable enabled"
    fi
    sudo install -d -o root -g root -m 0755 /etc/helexa-bench
'; then
    echo "  failed to prepare ${bench_host} for helexa-bench"
fi

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

echo "==> ${bench_host}: syncing bench config"
sync_config "${bench_host}" \
    "${repo_path}/asset/helexa-bench/${bench_host%%.*}.toml" \
    /etc/helexa-bench/helexa-bench.toml

# ── bench UI: public nginx vhost on the gateway (bench.helexa.ai) ──────
# The built SPA is rsynced to the webroot by deploy.yml; nginx serves it
# and reverse-proxies /api to the bench API on bob (internal WireGuard),
# so the UI is same-origin and the API stays off the public internet.
# This block runs as the operator (own sudo); deploy.yml only needs the
# scoped rsync grant in asset/sudoers.d/cortex-host.conf. Idempotent.
bench_ui_domain="bench.helexa.ai"
bench_ui_webroot="/var/www/${bench_ui_domain}"
# certbot via Cloudflare DNS-01 (ECDSA) — the convention used for the
# other certs on this host. DNS-01 needs neither nginx running nor :80,
# so the cert can be provisioned independently of the vhost being served.
le_email="ops@zap.pics"
cf_creds="/root/.certbot-internal"

echo "==> ${cortex_host}: bench UI nginx vhost (${bench_ui_domain})"
ssh "${cortex_host}" "sudo install -d -o root -g root -m 0755 '${bench_ui_webroot}'"
ssh "${cortex_host}" "test -f '${bench_ui_webroot}/index.html' || \
    echo 'helexa bench UI — not yet deployed' | sudo tee '${bench_ui_webroot}/index.html' >/dev/null"
# SELinux (enforcing on Fedora): label the webroot httpd_sys_content_t so
# nginx can read it (else 403). New files rsynced in inherit this type.
ssh "${cortex_host}" "sudo restorecon -R '${bench_ui_webroot}'"

cert_present() { ssh "${cortex_host}" "test -d '/etc/letsencrypt/live/${bench_ui_domain}'"; }
nginx_active() { ssh "${cortex_host}" "systemctl is-active --quiet nginx"; }

# Obtain the cert (idempotent: --keep-until-expiring). Cloudflare DNS-01,
# so it works even while nginx is stopped.
if ! cert_present; then
    echo "  obtaining Let's Encrypt cert via Cloudflare DNS-01…"
    ssh "${cortex_host}" "sudo certbot certonly \
        -m '${le_email}' --agree-tos --no-eff-email --noninteractive \
        --cert-name '${bench_ui_domain}' --key-type ecdsa \
        --dns-cloudflare --dns-cloudflare-credentials '${cf_creds}' \
        --dns-cloudflare-propagation-seconds 60 \
        --keep-until-expiring -d '${bench_ui_domain}'" \
        || echo "  WARNING: certbot failed (Cloudflare creds for helexa.ai?) — review on ${cortex_host}"
fi

# Install the matching vhost: the full TLS config once the cert exists,
# otherwise the http-only bootstrap. This invariant keeps `nginx -t`
# passing — never reference a cert that isn't there yet.
if cert_present; then
    cfg="${bench_ui_domain}.conf"
else
    cfg="${bench_ui_domain}.bootstrap.conf"
fi
if rsync --archive --compress --chown root:root --chmod 0644 \
    --rsync-path 'sudo rsync' \
    "${repo_path}/asset/nginx/${cfg}" \
    "${cortex_host}:/etc/nginx/sites-available/${bench_ui_domain}.conf"; then
    ssh "${cortex_host}" "
        set -eu
        sudo ln -sf ../sites-available/${bench_ui_domain}.conf \
            /etc/nginx/sites-enabled/${bench_ui_domain}.conf
        sudo nginx -t
    " && echo "  vhost installed (${cfg})"
    if nginx_active; then
        ssh "${cortex_host}" "sudo systemctl reload nginx"
    else
        echo "  NOTE: nginx is inactive on ${cortex_host} — start it to serve ${bench_ui_domain}"
        echo "        (this also re-activates the other enabled vhosts on the host)."
    fi
else
    echo "  failed to install ${bench_ui_domain} vhost"
fi
