import argparse
import glob
import os
import string
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


punctuation = list(string.punctuation)
punctuation.remove('{')
punctuation.remove('}')
punctuation.append('...')
punctuation.append('--')
punctuation.append('``')
punctuation.append("''")

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def remove_stopwords(sent):
    return (token for token in sent if token not in stop_words)

def remove_braces(sent):
    for token in sent:
        if re.match('^({{.{3}}})$', token):
            yield token
        else: 
            yield token.replace('}','').replace('{','')

def replace_numerics(sent):
    for token in sent:
        if re.match('^([0-9]+|[0-9\.]*[iv]+|[0-9]+[\.a-z]*)$', token):
            yield '{{NUM}}'
        else:
            yield token

def replace_symbols(sent):
    for i,token in enumerate(sent):
        if i == 0 and token == 'A':
            yield token
        elif len(token) == 1:
            yield '{{SYM}}'
        elif re.match('^[A-Z]+$', token):
            yield '{{ACR}}'
        elif re.match('^[A-Z0-9]+$', token):
            yield '{{SYM}}'
        elif re.match('^[A-Za-z][0-9ijk]+$', token):
            yield '{{SYM}}'
        else:
            yield token

def stem(sent):
    return (stemmer.stem(token) for token in sent)


def tokenize(sents):
    sents = ((sent.split(' ')) for sent in sents)
    sents = (remove_stopwords(sent) for sent in sents)
    sents = (remove_braces(sent) for sent in sents)
    sents = (replace_numerics(sent) for sent in sents)
    sents = (replace_symbols(sent) for sent in sents)
    sents = (stem(sent) for sent in sents)
    sents = (' '.join(sent) for sent in sents)
    return sents


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Map sentence rankings back to the original sentences.')
    parser.add_argument('documents', type=glob.iglob,
            help='The documents (directories) for which to map rankings and sentences.')
    parser.add_argument('--ranking-file', type=str, choices=['labels', 'baseline'], default='labels',
            help='Extension of the file(s) containing the rankings to map against.')
    args = parser.parse_args()

    for doc_dir in filter(os.path.isdir, args.documents):
        doc_sents = '{}/article.sentences'.format(doc_dir)
        with open(doc_sents, 'r') as f:
            sents = f.read().split('\n')

        doc_ranks = '{}/article.{}'.format(doc_dir, args.ranking_file)
        with open(doc_ranks, 'r') as f:
            ranks = f.read().split('\n')

        sent_rank_to_idx = { int(rank): idx for idx, rank in enumerate(ranks) }
        ranked_sents = []

        for rank in range(len(sents)):
            sent_idx = sent_rank_to_idx[rank]
            sent = sents[sent_idx]
            ranked_sents.append(sent)

        doc_summary = '{}/summary.sentences'.format(doc_dir)
        with open(doc_summary, 'w+') as f:
            f.write('\n'.join(ranked_sents))

        tokens = list(tokenize(ranked_sents))
        doc_summary_tokens = '{}/summary.tokens'.format(doc_dir)
        with open(doc_summary_tokens, 'w+') as f:
            f.write('\n'.join(tokens))

