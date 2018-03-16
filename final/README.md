
First, set up an appropriate python3 virtual environment:
    virtualenv .venv --python=python3.5
    source .venv/bin/activate
    pip install -r requirements.txt

All following commands will assume this virtualenv is activate,
and that the current directory is the project's top level directory,
i.e., "final/".

Steps to download and prepare cmp-lg dataset:
    ./cmp-lg/bulk_download.sh
    python cmp-lg/scrape_cmp_lg.py "cmp-lg/data/*.xml"
    python cmp-lg/preprocess_cmp_lg.py "cmp-lg/data/*"
    python tfidf_vectorize.py "cmp-lg/data/*"
    python vector_compare.py "cmp-lg/data/*"

Steps to download and prepare medline dataset:
    ./pubmed/get_file_list.sh
    # B = number of batches to download, where batch size is 1000 documents
    for i in {0..B}; do ./pubmed/batch_download.sh $i; done
    python pubmed/scrape_medline.py "pubmed/data/*.nxml"
    python pubmed/preprocess_medline.py "pubmed/data/*"
    python tfidf_vectorize.py "pubmed/data/*"
    python vector_compare.py "pubmed/data/*"

Note: for parallelized batch downloading of medline documents over C cores, use:
    seq 0 B | xargs -n1 -P C ./pubmed/batch_download.sh
