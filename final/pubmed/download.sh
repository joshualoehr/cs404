#!/bin/bash
# Download a tar.gz file from PubMed OpenAccess.
# Must be given the name of the tar.gz file, which is found in the first column of 
# oa_file_list.csv.

if [[ $# -lt 1 ]]; then
    echo 'Error: missing argument: name of tar.gz file to download'
    exit 0
fi

DATADIR=$(dirname ${BASH_SOURCE[0]})/data
if ! [[ -d $DATADIR ]]; then
    mkdir $DATADIR
fi

TARGZ=$(echo $1 | sed "s/.*\///g")
curl -sL ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/$1 > $DATADIR/$TARGZ
tar -C $DATADIR -xzf $DATADIR/$TARGZ
rm $DATADIR/$TARGZ
