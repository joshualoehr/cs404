#!/bin/python

import argparse
import glob
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate tf-idf vectors for each sentence in a document.")
    parser.add_argument("documents", type=glob.iglob,
            help="Glob pattern of documents (directories) comprising the corpus to vectorize.")
    args = parser.parse_args()

    doc_sents = {}
    for doc_dir in filter(os.path.isdir, args.documents):
        for doc_part in ['article', 'abstract']:
            doc_tokens = '{}/{}.tokens'.format(doc_dir, doc_part)
            with open(doc_tokens, 'r') as f:
                sents = f.read().split('\n')
            doc_sents['{}/{}'.format(doc_dir, doc_part)] = sents

    tfidf = TfidfVectorizer()
    tfidf.fit_transform((sent for doc in doc_sents.values() for sent in doc))

    feature_names = tfidf.get_feature_names()
    for doc,sents in doc_sents.items():
        scores = tfidf.transform(sents)

        doc_tfidf = '{}.tfidf'.format(doc)
        with open(doc_tfidf, 'w+') as f:
            lines = []
            for row in scores:
                dense_row = row.todense()
                lines.append(','.join([str(dense_row[0, col]) for col in range(scores.shape[1])]))
            f.write('\n'.join(lines))

