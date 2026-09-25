#!/usr/bin/env bash

# Runs merge with -matrix and -index, then replay on the history file
# that merge wrote, and checks that all output files are identical.
# usage: test_merge_direct.sh [-dl] [-skip <n>] -purged <file> [merge args...]

: ${wdir:?missing}
build_tree="$PROJECT_BINARY_DIR"

suffix=""
purged=""
skip=()
merge_args=()
while [ $# -gt 0 ] ; do
    case "$1" in
        -b) build_tree="$2"; shift 2;;
        -dl) suffix="-dl"; shift;;
        -skip) skip=(-skip "$2"); shift 2;;
        -purged) purged="$2"; shift 2;;
        *) merge_args+=("$1"); shift;;
    esac
done
: ${purged:?missing}

MERGE="$build_tree/filter/merge$suffix"
REPLAY="$build_tree/filter/replay$suffix"
ideals=()
[ -n "$suffix" ] && ideals=(-ideals)

set -e
for out in mat.sparse.bin mat.sparse.txt ; do
    md="$wdir/merge_${out//./_}"
    rd="$wdir/replay_${out//./_}"
    mkdir -p "$md" "$rd"
    "$MERGE" -mat "$purged" -out "$wdir/history.gz" -t 2 "${skip[@]}" \
        "${merge_args[@]}" -matrix "$md/$out" -index "$md/index.gz" \
        ${ideals:+-ideals "$md/ideals"}
    "$REPLAY" -purged "$purged" -his "$wdir/history.gz" -t 2 "${skip[@]}" \
        -out "$rd/$out" -index "$rd/index.gz" ${ideals:+-ideals "$rd/ideals"}
    for f in $(ls "$md") ; do
        [ -f "$rd/$f" ] || { echo "$0: replay did not write $f" >&2; exit 1; }
        if [[ $f == *.gz ]] ; then
            cmp <(zcat "$md/$f") <(zcat "$rd/$f")
        else
            cmp "$md/$f" "$rd/$f"
        fi
    done
    [ "$(ls "$md" | wc -l)" = "$(ls "$rd" | wc -l)" ]
done
