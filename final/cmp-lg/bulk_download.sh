#!/bin/bash
# Downloads the TIPSTER Text Summarization Evaluation Conference (SUMMAC) cmp-lg corpus.
# See: http://www-nlpir.nist.gov/related_projects/tipster_summac/cmp_lg.html

curl -sL http://www-nlpir.nist.gov/related_projects/tipster_summac/cmplg-xml.tar.gz > cmplg-xml.tar.gz
tar -xzf cmplg-xml.tar.gz
rm cmplg-xml.tar.gz
mv cmplg-xml data
