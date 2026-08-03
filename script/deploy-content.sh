#!/usr/bin/env bash
# Deploy investor-portal round content to gallumbits.
#
# Content is NOT deployed by CI, and deliberately so. It lives in the
# private helexa/angels-content repository because helexa/helexa is open
# source — a business plan committed there would be a business plan
# published — and it rides the operator's own ssh rather than passing
# through a runner that would need read access to it.
#
# The commit sha of the content tree is written to VERSION and recorded
# against every document view, so "which version of the plan did this
# investor see?" is answerable later, when the terms discussed in a
# meeting differ from the terms on the page.
#
# Usage:
#   script/deploy-content.sh [path-to-angels-content] [target-host]
set -euo pipefail

content_repo="${1:-${HOME}/git/helexa/angels-content}"
target_host="${2:-gallumbits.kosherinata.internal}"
target_dir="/var/lib/helexa-angels/content"

if [ ! -d "${content_repo}/.git" ]; then
    echo "error: ${content_repo} is not a git checkout" >&2
    echo "  clone it first: git clone gitea@git.internal:helexa/angels-content.git" >&2
    exit 1
fi

sha=$(git -C "${content_repo}" rev-parse --short HEAD)
if ! git -C "${content_repo}" diff --quiet || ! git -C "${content_repo}" diff --cached --quiet; then
    echo "warning: ${content_repo} has uncommitted changes"
    echo "  the deployed VERSION will read ${sha}-dirty, which makes the"
    echo "  access record ambiguous. Commit first unless this is a test."
    sha="${sha}-dirty"
fi

echo "==> deploying content @ ${sha} to ${target_host}:${target_dir}"

# Stage VERSION in a temp copy rather than writing into the working tree
# (it is gitignored there for exactly this reason).
staging=$(mktemp -d)
trap 'rm -rf "${staging}"' EXIT
# --exclude .git: the deployed tree is content, not a checkout. Shipping
# .git would put the full revision history — including anything ever
# committed and later removed — on the server.
rsync --archive --exclude '.git' --exclude '.gitignore' --exclude 'README.md' \
    "${content_repo}/" "${staging}/"
printf '%s\n' "${sha}" > "${staging}/VERSION"

# 0640/0750 owned by the service user: this is the confidential material
# itself, and nothing else on the host has any business reading it.
rsync --archive --delete --compress \
    --rsync-path 'sudo rsync' \
    --chown helexa-angels:helexa-angels \
    --chmod 'D0750,F0640' \
    "${staging}/" "${target_host}:${target_dir}/"

echo "==> restarting helexa-angels (round metadata is synced from disk at startup)"
ssh "${target_host}" 'sudo /usr/bin/systemctl restart helexa-angels.service'

sleep 3
ssh "${target_host}" 'systemctl is-active helexa-angels.service' \
    && echo "==> deployed ${sha}" \
    || { echo "helexa-angels did not come back — check journalctl" >&2; exit 1; }
