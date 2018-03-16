#!/bin/bash
# Downloads oa_file_list.csv from nih.gov. This file is too large to be included in git.

curl -sL ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv > $(dirname ${BASH_SOURCE[0]})/oa_file_list.csv
