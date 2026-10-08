#!/usr/bin/env bash
# version.sh - prints the version of the commit checked out: yy.mmdd.### - the commit's date (UTC), then its number
# among that day's commits on the first-parent history (the day's first commit is .001). Uncommitted changes to
# tracked files add "-dirty"; outside a git checkout it prints "dev". build.sh hands it to cargo, the workflow names
# the release after it, so there is no version file to bump.
set -euo pipefail
cd "$(dirname "$0")"
git rev-parse --git-dir >/dev/null 2>&1 || { echo dev; exit 0; }
day() { TZ=UTC git log "$@" --format=%cd --date=format-local:%y.%m%d; }
d=$(day -1)
# awk reads to the end: an early `exit` closed the pipe on git, whose SIGPIPE failed the script under pipefail (the
# CI's Version step, with the whole history checked out)
n=$(day --first-parent | awk -v d="$d" 'stop { next } $0 == d { n++; next } { stop = 1 } END { print n + 0 }')
dirty=; git diff --quiet HEAD -- 2>/dev/null || dirty=-dirty
printf '%s.%03d%s\n' "$d" "$n" "$dirty"
