#!/usr/bin/env bash
# Fail when a binary needs a newer glibc than the release it is packaged for.
#
# CI builds directly on the runner host, whose Fedora release can be newer
# than the one the RPMs target. A binary linked there can pick up a symbol
# version (e.g. log10f@GLIBC_2.43) that the target's glibc lacks; dnf then
# refuses the upgrade and the host keeps running the old build.
#
# Usage: script/check-glibc-floor.sh <max-glibc> <binary>...
set -euo pipefail
max="$1"; shift
status=0
for bin in "$@"; do
  need=$(objdump -T "$bin" | grep -oE 'GLIBC_[0-9]+(\.[0-9]+)+' | sed 's/GLIBC_//' | sort -uV | tail -1)
  if [ "$(printf '%s\n%s\n' "$need" "$max" | sort -V | tail -1)" != "$max" ]; then
    echo "::error::$bin needs glibc $need; the target ships $max. Symbols:"
    objdump -T "$bin" | awk -v m="$max" '/GLIBC_/ { v=$0; sub(/.*GLIBC_/, "", v); sub(/[) ].*/, "", v); print v, $NF }' \
      | sort -uV | awk -v m="$max" '{ cmd="printf \"%s\\n%s\\n\" " $1 " " m " | sort -V | tail -1"; cmd | getline top; close(cmd); if (top != m) print "  " $2 "@GLIBC_" $1 }'
    status=1
  else
    echo "$bin: needs glibc $need (target $max) ok"
  fi
done
exit $status
