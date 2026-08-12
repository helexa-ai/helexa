#!/bin/env bash
#
# Watch for a model repo to become public, then fetch it immediately.
#
# Announced-but-unreleased weights appear without warning, often
# outside working hours. Polling for the repo and starting the download
# the moment it exists turns a multi-hour wait at the start of a working
# session into a wait that already happened overnight.
#
# Several candidate ids can be watched at once because the exact
# repo name is rarely known in advance (`-Instruct`, `-Thinking` and
# bare variants are all plausible). The first one to appear wins; the
# rest keep being watched, so a family that lands in stages is fetched
# in stages.
#
# Both HuggingFace and ModelScope are polled. ModelScope frequently
# publishes Qwen weights at or before the HF mirror, and a repo that is
# reachable on either host is worth fetching.
#
# Output is one line per event, so the script can be tailed as a
# progress stream.
#
# Usage:
#   script/await-model.sh <repo-id> [<repo-id> ...]
#
# Environment:
#   CACHE_DIR    HF-layout cache to download into.
#                Default: $HF_HUB_CACHE, else ~/.cache/huggingface/hub
#   LINK_DIR     If set, a symlink to each fetched model is created
#                here. For fleets that keep a cache on a slow bulk
#                volume but want weights served off a fast one, point
#                CACHE_DIR at the fast volume and LINK_DIR at the
#                directory the serving daemon actually reads.
#   POLL_SECS    Seconds between polls (default 60).
#   MAX_HOURS    Give up after this long (default 48).
#   MIN_FREE_GB  Refuse to start a download when the cache volume has
#                less than this much free (default 80).

set -uo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <repo-id> [<repo-id> ...]" >&2
    exit 64
fi

CACHE_DIR="${CACHE_DIR:-${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}}"
LINK_DIR="${LINK_DIR:-}"
POLL_SECS="${POLL_SECS:-60}"
MAX_HOURS="${MAX_HOURS:-48}"
MIN_FREE_GB="${MIN_FREE_GB:-80}"

# HuggingFace answers 401 for both "private" and "does not exist", so a
# 200 is the only signal that means "public and fetchable". Anything
# else is treated as not-yet-there rather than as an error, because a
# transient 5xx during a release is common and must not end the watch.
hf_public() {
    [ "$(curl -sf -o /dev/null -w '%{http_code}' \
        "https://huggingface.co/api/models/$1" 2>/dev/null)" = "200" ]
}

# ModelScope answers 200 with an empty payload for a repo id that has
# been *reserved* but not published — the countdown pages that appear
# ahead of a launch behave exactly this way. Asking for the file tree
# instead is the honest question: it succeeds only once there are files.
#
# Testing the model endpoint alone reports every announced-but-unreleased
# model as available, once per poll, which is worse than no signal at
# all — it trains you to ignore the one that means something.
ms_public() {
    curl -sf "https://modelscope.cn/api/v1/models/$1/repo/files?Revision=master" \
        2>/dev/null | grep -q '"Success": *true'
}

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# HF's on-disk cache name for a repo: `models--<org>--<name>`.
cache_name() { echo "models--${1//\//--}"; }

fetch() {
    local repo="$1" dest free_gb
    dest="$CACHE_DIR/$(cache_name "$repo")"

    # Unattended fetching is only safe with a floor under it. Several
    # candidate ids are usually watched at once because the exact name
    # is not known in advance, and a family that publishes every variant
    # at once would otherwise fill the volume overnight and take the
    # serving daemon's cache down with it.
    free_gb=$(df -BG --output=avail "$CACHE_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')
    if [ -n "$free_gb" ] && [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
        log "SKIPPED $repo (${free_gb}G free < ${MIN_FREE_GB}G floor)"
        return 1
    fi

    log "FETCHING $repo -> $dest (${free_gb}G free)"
    # `hf download` is resumable and verifies checksums, so an
    # interrupted run is safe to repeat. Weights only: the config and
    # tokenizer come along anyway, but consolidated/original-format
    # duplicates of the same tensors are excluded so a repo that ships
    # two layouts is not downloaded twice.
    #
    # Each pattern needs its own `--exclude`. Passing two patterns to a
    # single flag makes the CLI treat the second as an explicit
    # *filename to download*, which then 404s -- and it only says so in
    # a warning, so the failure looks like a missing file in the repo.
    if ! HF_HUB_CACHE="$CACHE_DIR" hf download "$repo" \
        --exclude "original/*" --exclude "consolidated*" >/dev/null; then
        log "FAILED $repo (will retry next poll)"
        return 1
    fi

    local size
    size="$(du -sh "$dest" 2>/dev/null | cut -f1)"
    log "FETCHED $repo ($size)"

    if [ -n "$LINK_DIR" ] && [ ! -e "$LINK_DIR/$(cache_name "$repo")" ]; then
        ln -s "$dest" "$LINK_DIR/$(cache_name "$repo")" \
            && log "LINKED $LINK_DIR/$(cache_name "$repo")"
    fi
    return 0
}

mkdir -p "$CACHE_DIR"
log "WATCHING $* (cache=$CACHE_DIR poll=${POLL_SECS}s)"

deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
remaining=("$@")
announced=()

while [ "${#remaining[@]}" -gt 0 ]; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        log "TIMEOUT after ${MAX_HOURS}h; still waiting on: ${remaining[*]}"
        exit 1
    fi

    still=()
    for repo in "${remaining[@]}"; do
        if hf_public "$repo"; then
            log "APPEARED $repo (huggingface)"
            fetch "$repo" || still+=("$repo")
        elif ms_public "$repo"; then
            # Published on ModelScope but not yet mirrored to HF. Useful
            # as a heads-up, since the mirror usually follows within
            # minutes, but `hf download` cannot fetch from ModelScope so
            # the watch continues.
            #
            # Announced once per repo, not once per poll: this state can
            # last a while, and a line a minute would bury whatever
            # follows it.
            case " ${announced[*]-} " in
                *" $repo "*) ;;
                *)
                    log "APPEARED $repo (modelscope; awaiting hf mirror)"
                    announced+=("$repo")
                    ;;
            esac
            still+=("$repo")
        else
            still+=("$repo")
        fi
    done
    remaining=("${still[@]}")

    [ "${#remaining[@]}" -gt 0 ] && sleep "$POLL_SECS"
done

log "DONE all requested models fetched"
