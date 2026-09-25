#!/usr/bin/env bash

# Worker tnum of a chunk draws from seed+tnum. Two chunks of the same
# sample whose seeds are less than -t apart share a random sequence and
# output twin relations that differ only in the special-q (purge keeps
# both, merge cancels them). fake_rels refuses such chunk lists; the
# optional fifth field gives each chunk its seed. Chunks of different
# samples may share a seed.

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

# expand <threads> <seed1> <seed2> <tag> [sample2]: chunks 61961-62961
# of $SAMPLE and 62961-63961 of sample2 (default $SAMPLE), in one chunk
# list, to chunk1.<tag> and chunk2.<tag>
expand() {
    printf '61961 62961 %s %s %s\n' $SAMPLE $WORKDIR/chunk1.$4 "$2" > $WORKDIR/chunklist
    printf '62961 63961 %s %s %s\n' ${5:-$SAMPLE} $WORKDIR/chunk2.$4 "$3" >> $WORKDIR/chunklist
    $FAKERELS -poly $POLY -lpb0 18 -lpb1 19 -sqside 1 -renumber $RENUMBER \
        -chunk-list $WORKDIR/chunklist -t $1
}

# same_ab <chunk1 file> <chunk2 file>: the a,b of the relations of
# chunk 1, truncated to the length of chunk 2, equal those of chunk 2
same_ab() {
    grep -v '^#' $2 | cut -d: -f1 > $WORKDIR/ab2
    grep -v '^#' $1 | cut -d: -f1 | head -n $(wc -l < $WORKDIR/ab2) > $WORKDIR/ab1
    cmp -s $WORKDIR/ab1 $WORKDIR/ab2
}

# the same seed in two chunks is refused (here the default -seed)
if expand 1 "" "" equal ; then
    echo "chunk list with equal seeds was accepted"
    exit 1
fi

# seeds 0 and 1 collide between threads with -t 2 (worker 1 of chunk 1
# and worker 0 of chunk 2 both draw from seed 1)
if expand 2 0 1 close ; then
    echo "chunk list with seeds less than -t apart was accepted"
    exit 1
fi

# two samples (here a copy) may share the default seed
cp $SAMPLE $WORKDIR/sample.copy
expand 1 "" "" twosamples $WORKDIR/sample.copy

# chunks with the same seed are twins: this is what a list with equal
# seeds would output
expand 1 7 1000000 a
expand 1 1000000 7 b
if ! same_ab $WORKDIR/chunk1.a $WORKDIR/chunk2.b ; then
    echo "chunks with the same seed are expected to output the same a,b"
    exit 1
fi

# with seeds -t apart, the chunks differ
expand 2 0 1000000 apart
if same_ab $WORKDIR/chunk1.apart $WORKDIR/chunk2.apart ; then
    echo "chunks with distinct seeds output the same a,b"
    exit 1
fi

exit 0
