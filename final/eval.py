import argparse
import glob
import numpy as np
import os
from scipy.stats import entropy
from numpy.linalg import norm
import nltk


def JSD(P, Q):
    P_norm = P / norm(P, ord=1)
    Q_norm = Q / norm(Q, ord=1)
    M = 0.5 * (P_norm + Q_norm)
    return 0.5 * (entropy(P_norm, M) + entropy(Q_norm, M))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate results of auto summarization.')
    parser.add_argument('documents', type=glob.iglob,
            help="The documents to evalutate.")
    parser.add_argument('--method', type=str, choices=['JSD','ROUGE'], default='ROUGE')
    args = parser.parse_args()

    for doc in filter(os.path.isdir, args.documents):
        doc_abstract = '{}/abstract.tokens'.format(doc)
        with open(doc_abstract, 'r') as f:
            abstract = f.read().split('\n')
            abstract = [sent.split(' ') for sent in abstract]
            abstract = [token for sent in abstract for token in sent] 
        ref_vocab = nltk.FreqDist(abstract)

        doc_summary = '{}/summary.tokens'.format(doc)
        with open(doc_summary, 'r') as f:
            summary = f.read().split('\n')
            summary = [sent.split(' ') for sent in summary]
            summary = [token for sent in summary for token in sent]
            summary = summary[:len(abstract)]
        sys_vocab = nltk.FreqDist(summary)

        ref_total = sum(ref_vocab.values())
        P_ref = { word: count/ref_total for word,count in ref_vocab.items() }

        sys_total = sum(sys_vocab.values())
        P_sys = { word: count/sys_total for word,count in sys_vocab.items() }

        P, Q = [], []
        for word in set(P_ref.keys()) | set(P_sys.keys()):
            P.append(P_ref.get(word, 0.0))
            Q.append(P_sys.get(word, 0.0))

        print(doc, JSD(P, Q))

        
