#!/usr/bin/env bash

# With no argument fake_rels reports the missing -poly and prints its
# usage (print_usage used to read argv[0] from a null pointer when no
# argument had been parsed).

: ${CADO_NFS_BINARY_DIR:?missing}

out=$(${CADO_NFS_BINARY_DIR}/sieve/fake_rels 2>&1)
[ $? -eq 1 ] || exit 1
grep -q 'available parameters' <<< "$out"
