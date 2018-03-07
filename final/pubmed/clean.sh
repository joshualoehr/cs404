#!/bin/bash
# Delete any downloaded data files other than .nxml and .txt files.

find data/* -type f -not -name *.nxml -not -name *.txt -exec rm {} +
