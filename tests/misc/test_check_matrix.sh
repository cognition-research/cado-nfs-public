#!/usr/bin/env bash

# Runs merge and replay on a purged file and checks their matrix and
# index outputs with check_matrix; then checks that check_matrix
# rejects a truncated matrix and a corrupted index, accepts a good
# kernel dependency, and rejects a bad one.
# usage: test_check_matrix.sh -purged <file> [merge args...]

: ${wdir:?missing}
build_tree="$PROJECT_BINARY_DIR"

purged=""
merge_args=()
while [ $# -gt 0 ] ; do
    case "$1" in
        -b) build_tree="$2"; shift 2;;
        -purged) purged="$2"; shift 2;;
        *) merge_args+=("$1"); shift;;
    esac
done
: ${purged:?missing}

MERGE="$build_tree/filter/merge"
REPLAY="$build_tree/filter/replay"
CHECK="$build_tree/misc/check_matrix"

set -e

md="$wdir/merge"
rd="$wdir/replay"
mkdir -p "$md" "$rd"
"$MERGE" -mat "$purged" -out "$wdir/history.gz" -t 2 -skip 8 \
    "${merge_args[@]}" -matrix "$md/mat.sparse.bin" -index "$md/index.gz"
"$REPLAY" -purged "$purged" -his "$wdir/history.gz" -t 2 -skip 8 \
    -out "$rd/mat.sparse.bin" -index "$rd/index.gz"
"$CHECK" -purged "$purged" -index "$md/index.gz" -matrix "$md/mat.sparse.bin"
"$CHECK" -purged "$purged" -index "$rd/index.gz" -matrix "$rd/mat.sparse.bin"

# a truncated matrix file must be rejected
for f in sparse.rw.bin sparse.cw.bin dense.bin dense.rw.bin dense.cw.bin ; do
    cp "$md/mat.$f" "$md/bad.$f"
done
size=$(wc -c < "$md/mat.sparse.bin")
head -c "$((size - 4))" "$md/mat.sparse.bin" > "$md/bad.sparse.bin"
if "$CHECK" -purged "$purged" -index "$md/index.gz" \
    -matrix "$md/bad.sparse.bin" ; then
    echo "$0: a truncated matrix passed check_matrix" >&2
    exit 1
fi

# an index with two swapped rows must be rejected
zcat "$md/index.gz" | awk 'NR==2{s=$0;next} NR==3{print;print s;next} {print}' \
    > "$md/bad-index"
if "$CHECK" -purged "$purged" -index "$md/bad-index" \
    -matrix "$md/mat.sparse.bin" ; then
    echo "$0: a corrupted index passed check_matrix" >&2
    exit 1
fi

# hand-made kernel check: rows 0+2 and 1+3 of the purged file cancel,
# so the dependency {row 0, row 1} of the index is good, and the
# dependency {row 0} is not
kd="$wdir/kernel"
mkdir -p "$kd"
cat > "$kd/purged" <<EOF
# 4 3 3
1,1:0,1
2,1:0,1
3,1:1,2
4,1:1,2
EOF
printf '2\n2 0 2\n2 1 3\n' > "$kd/index"
printf '\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00' \
    > "$kd/ker.good"
printf '\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00' \
    > "$kd/ker.bad"
"$CHECK" -purged "$kd/purged" -index "$kd/index" -ker "$kd/ker.good"
if "$CHECK" -purged "$kd/purged" -index "$kd/index" -ker "$kd/ker.bad" ; then
    echo "$0: a bad dependency passed check_matrix" >&2
    exit 1
fi
