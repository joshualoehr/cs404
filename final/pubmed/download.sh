#!/bin/bash
# Download a tar.gz file from PubMed OpenAccess.
# Must be given the name of the tar.gz file, which is found in the first column of 
# oa_file_list.csv.

if [[ $# -lt 1 ]]; then
    echo 'Error: missing argument: name of tar.gz file to download'
    exit 0
fi

TARGZ=$(echo $1 | sed "s/.*\///g")
curl -sL ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/$1 > data/$TARGZ
tar -C data -xzf data/$TARGZ
rm data/$TARGZ
