#!/bin/env python

###############################################################################
# CSCI 404 - Assignment 3 - Josh Loehr & Robin Cosbey - naive_bayes.py        #
###############################################################################

import argparse
from collections import Counter
from itertools import product
import nltk
from pathlib import Path

def get_filepaths(data_path):
    train_path = data_path.joinpath('train.txt').absolute().as_posix()
    test_path  = data_path.joinpath('test.txt').absolute().as_posix()
    return train_path, test_path

def load_data(data_dir):
    train_path, test_path = get_filepaths(data_dir)
    with open(train_path, 'r') as f:
        train = [l.replace('\n', '') for l in f.readlines()]
    with open(test_path, 'r') as f:
        test = [l.replace('\n', '') for l in f.readlines()]
    return train, test

def create_ngram_models(sentences, lambda):
    sentences = ['<s> {} </s>'.format(s) for s in sentences]
    tokens = ' '.join(sentences).split(' ') 
    
    vocab = nltk.Text(tokens).vocab()
    unigrams = [word if vocab[word] > 1 else '<UNK>' for token in tokens]
    bigrams  = [' '.join(unigrams[i:i+2]) for i in range(0, len(unigrams), 2)]
    trigrams = [' '.join(unigrams[i:i+3]) for i in range(0, len(unigrams), 3)]  

    tokens   = nltk.Text(tokens)
    unigrams = nltk.Text(unigrams)
    bigrams  = nltk.Text(bigrams)
    trigrams = nltk.Text(trigrams)

    return smooth(tokens, unigrams, bigrams, trigrams)

def smooth(tokens, unigrams, bigrams, trigrams):
    unique_tokens = tokens.vocab().keys()
    unique_tokens += ['<UNK>']

    all_bigrams = [''.join(pair) for pair in product(unique_tokens, unique_tokens)]
    all_trigrams = [''.join(pair) for pair in product(unique_tokens, all_bigrams)]

    bigram_counts = {key: 0 for key in all_bigrams}
    trigram_counts = {key: 0 for key in all_trigrams}

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Naive Bayes Text Classifier")
    parser.add_argument('--data', type=str, required=True,
            help='Location of the data directory containing train.txt and test.txt')
    parser.add_argument('--lambda', type=float, default=0.01,
            help='Lambda parameter for Laplace smoothing (use 1 for add-1 smoothing)')
    args = parser.parse_args()

    data_path = Path(args.data)

    # Load and prepare train/test data
    train, test  = load_data(data_path)
    train_models = create_ngram_models(train)
    test_models  = create_ngram_models(test)

    print("Size of training vocab: {}".format("TODO")



