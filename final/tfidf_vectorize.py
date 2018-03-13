#!/bin/python

import argparse
import glob
import re
from sklearn.feature_extraction.text import TfidfVectorizer

header = re.compile('^{{h.*}}$')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate tf-idf vectors for each sentence in a document.")
    parser.add_argument("documents", type=glob.iglob,
            help="Glob pattern of .tokens files representing the corpus to vectorize.")
    args = parser.parse_args()

    doc_sents = {}
    for doc in args.documents:
        with open(doc, 'r') as f:
            sents = f.read().split('\n')
        sents = [sent for sent in sents if not header.match(sent)]
        doc_sents[doc] = sents

    tfidf = TfidfVectorizer()
    tfidf.fit_transform((sent for doc in doc_sents.values() for sent in doc))

    feature_names = tfidf.get_feature_names()
    for doc,sents in doc_sents.items():
        scores = tfidf.transform(sents)

        basename = '.'.join(doc.split('.')[:-1])
        filename = '{}.scores'.format(basename)
        with open(filename, 'w+') as f:
            for row in scores:
                dense_row = row.todense()
                line = ','.join([str(dense_row[0, col]) for col in range(scores.shape[1])])
                f.write('{}\n'.format(line))


