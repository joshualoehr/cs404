#!/bin/bash
# Downloads a batch of N tar.gz files from PubMed.
# Requires the 0-indexed batch number as an argument.

if [[ $# -lt 1 ]]; then
    echo 'Error: missing argument: batch number'
    exit 0
fi

N=10000
csvcut -c File oa_file_list.csv | tail -n +$((2 + $1 * $N)) | head -n$N | xargs -n1 -P8 -I {} ./download.sh {}
