#!/bin/bash
# Submit an SRPM to COPR, watch the build, and dump per-chroot build logs
# to stdout so they are captured in CI output.
#
# Usage: copr-build.sh <project> <srpm> [srpm...]
# Example: copr-build.sh helexa/cortex ./cortex-0.1.2-1.fc43.src.rpm

set -o pipefail

PROJECT="$1"
shift

if [ -z "$PROJECT" ] || [ "$#" -eq 0 ]; then
  echo "usage: $0 <project> <srpm> [srpm...]" >&2
  exit 2
fi

# Submit without waiting; capture the build ID from stdout.
SUBMIT_OUT=$(copr-cli build --nowait "$PROJECT" "$@")
echo "$SUBMIT_OUT"
BUILD_ID=$(echo "$SUBMIT_OUT" | grep -oP 'Created builds: \K[0-9]+' | head -n1)

if [ -z "$BUILD_ID" ]; then
  echo "error: could not parse build ID from copr-cli output" >&2
  exit 1
fi

echo
echo "Build $BUILD_ID submitted to $PROJECT"
echo "Follow live: https://copr.fedorainfracloud.org/coprs/build/$BUILD_ID"
echo

# Watch the build; captures status transitions to stdout. Exit non-zero
# on build failure, but defer propagating that until after we've fetched
# logs so the CI output contains diagnostics either way.
if copr-cli watch-build "$BUILD_ID"; then
  STATUS=0
else
  STATUS=$?
fi

# Fetch per-chroot results (logs + rpms). Anonymous download — no auth needed.
mkdir -p copr-logs
copr-cli download-build --dest copr-logs "$BUILD_ID" || {
  echo "warning: failed to download build artifacts" >&2
}

# Dump each chroot's builder-live.log as a collapsible group.
for chroot_dir in copr-logs/*/; do
  [ -d "$chroot_dir" ] || continue
  chroot=$(basename "$chroot_dir")
  log="${chroot_dir}builder-live.log"
  if [ -f "$log" ]; then
    echo
    echo "::group::${chroot} builder-live.log"
    cat "$log"
    echo "::endgroup::"
  fi
done

exit "$STATUS"
