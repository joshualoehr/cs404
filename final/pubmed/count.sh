#!/bin/bash
# Count how many files have been downloaded so far, and how much space they take up.

echo Documents downloaded: $(find data/* -type f -name *.nxml | wc -l) \($(du -sh data)\)
