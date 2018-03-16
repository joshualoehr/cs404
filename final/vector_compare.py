#!/bin/python

import argparse
import glob
import os
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity


def load_tfidf_vecs(document):
    with open(document, 'r') as d:
        data = d.read().split('\n')
        data = [line.split(',') for line in data]

    M, N = 1, len(data[0])
    vecs = []

    for sent_scores in data:
        vec = lil_matrix((M, N))
        for i,val in enumerate(sent_scores):
            if val != '0.0' and val != '':
                vec[0,i] = float(val)
        vecs.append(csr_matrix(vec))

    return vecs

def calc_similarity(article_vec, abstract_vecs):
    similarities = [cosine_similarity(article_vec, vec) for vec in abstract_vecs]
    return max(similarities)

def get_rankings(similarities):
    similarities = list(enumerate(similarities))
    similarities = sorted(similarities, key=lambda s: s[1], reverse=True)
    similarities = [(idx, rank) for rank, (idx, score) in enumerate(similarities)]
    similarities = sorted(similarities, key=lambda s: s[0])
    similarities = [rank for idx, rank in similarities]
    return similarities

def write_labels(document, rankings):
    filename = '{}/article.labels'.format(document)
    with open(filename, 'w+') as f:
        f.write('\n'.join([str(rank) for rank in rankings]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Compares sentences in an article body to those in the abstract, creating a similarity score per sentence.")
    parser.add_argument('documents', type=glob.iglob,
            help="The documents for which to generate scores.")
    args = parser.parse_args()

    
    for doc in filter(os.path.isdir, args.documents):
        abstract = '{}/abstract.tfidf'.format(doc)
        abstract_vecs = load_tfidf_vecs(abstract)

        article  = '{}/article.tfidf'.format(doc)
        article_vecs  = load_tfidf_vecs(article)
        
        similarities = [calc_similarity(vec, abstract_vecs) for vec in article_vecs]
        rankings = get_rankings(similarities)

        write_labels(doc, rankings)
       
