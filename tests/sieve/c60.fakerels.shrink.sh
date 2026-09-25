#!/usr/bin/env bash

# The renumber table interleaves the sides of one prime, so -shrink-factor
# can map an index of each side to the same column. A relation of the
# output must hold every column at most once (merge requires it).

: ${CADO_NFS_BINARY_DIR:?missing}
: ${CADO_NFS_SOURCE_DIR:?missing}

set -e

FREEREL=${CADO_NFS_BINARY_DIR}/sieve/freerel
FAKERELS=${CADO_NFS_BINARY_DIR}/sieve/fake_rels
POLY=${CADO_NFS_SOURCE_DIR}/parameters/polynomials/c60.poly
SAMPLE=${CADO_NFS_SOURCE_DIR}/tests/sieve/c60.sample
WORKDIR=${wdir:?missing}

RENUMBER="${WORKDIR}/cadotest.fakerel.renumber"
$FREEREL -poly $POLY -renumber $RENUMBER -lpb0 18 -lpb1 19 -pmax 1

$FAKERELS -poly $POLY -lpb0 18 -lpb1 19 -q0 61961 -q1 62961 -sqside 1 \
    -shrink-factor 2 -renumber $RENUMBER -sample $SAMPLE -t 2 \
    > $WORKDIR/shrink2

nrels=$(grep -vc '^#' $WORKDIR/shrink2)
[ "$nrels" -gt 0 ]

# relations with a repeated column
nrep=$(grep -v '^#' $WORKDIR/shrink2 | cut -d: -f2 \
    | awk -F, '{ delete seen; for (i = 1; i <= NF; i++) { if ($i in seen) { n++; next } seen[$i] = 1 } } END { print n + 0 }')
if [ "$nrep" -ne 0 ]; then
    echo "$nrep of $nrels shrunk relations hold a repeated column"
    exit 1
fi
