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

def add_sentence_tokens(sentences, n):
    sos = "<s> " * (n-1)
    eos = "</s>"
    return ['{}{} {}'.format(sos, s, eos) for s in sentences]

def replace_singletons(tokens):
    fdist = nltk.FreqDist(tokens)
    return [token if fdist[token] > 1 else '<UNK>' for token in tokens]

def preprocess(sentences, n):
    sentences = add_sentence_tokens(sentences, n)
    tokens = ' '.join(sentences).split(' ')
    tokens = replace_singletons(tokens)
    return nltk.Text(tokens)

def smooth(tokens, ngrams, vocab_size):
    vocab = nltk.FreqDist(tokens)
    fdist = nltk.FreqDist(ngrams)

    laplace = lambda ngram, count: (count+1) / (vocab[ngram[:-1]] + vocab_size)
    return { ngram: laplace(ngram, count) for ngram, count in fdist.items() }

def create_model(tokens, n):
    if n == 1:
        return tokens
    if n == 2:
        return smooth(tokens, nltk.bigrams(tokens), len(tokens.vocab()))
    if n == 3:
        return smooth(nltk.bigrams(tokens), nltk.trigrams(tokens), len(tokens.vocab()))

def most_probable_tokens(model, prev, num_candidates=1, blacklist=[]):
    blacklist.append("<UNK>")
    filter = lambda ngram: ngram[-1] not in blacklist

    candidates = [(ngram[-1], prob) for ngram, prob in model.items() if ngram[:-1] == prev and filter(ngram)]
    candidates = sorted(candidates, key=lambda pair: pair[1], reverse=True)[:num_candidates]
    return candidates 

    
def generate(model, n, len_range, num=10):
    min_length, max_length = len_range

    sentences = ["<s>"] * num
    probabilities = [1] * num

    sos_grams = most_probable_tokens(model, ("<s>",), num_candidates=num)
    sentences = list(zip(sentences, [ngram for ngram, _ in sos_grams]))
    sentences = [list(sentence) for sentence in sentences]
    probabilities = [probabilities[i] * prob for i, (_, prob) in enumerate(sos_grams)]

    for i in range(num):
        while sentences[i][-1] != "</s>" and len(sentences[i]) < max_length:
            prev = tuple(sentences[i][-(n-1):])
            blacklist = ["</s>"] if len(sentences[i]) < min_length else []
            next_token, next_prob = most_probable_tokens(model, prev, blacklist=blacklist)[0]
            sentences[i].append(next_token)
            probabilities[i] *= next_prob
    
    return zip(sentences, probabilities)

def display_generated(model, n, len_range=(0,25)):
    generated = generate(model, n, len_range=len_range)
    for sentence, prob in generated:
        print("{} ({})".format(sentence, prob))



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Naive Bayes Text Classifier")
    parser.add_argument('--data', type=str, required=True,
            help='Location of the data directory containing train.txt and test.txt')
    parser.add_argument('--lambda', type=float, default=0.01,
            help='Lambda parameter for Laplace smoothing (use 1 for add-1 smoothing)')
    parser.add_argument('--unigrams', action="store_true")
    parser.add_argument('--bigrams', action="store_true")
    parser.add_argument('--trigrams', action="store_true")
    args = parser.parse_args()

    # Load and prepare train/test data
    data_path = Path(args.data)
    train, test = load_data(data_path)

    
    if args.unigrams:
        tokens = preprocess(train, n=1)
        print("Size of training vocab: {}".format(len(tokens.vocab())))
        # avg_sentence_length = int(len(tokens) / tokens.vocab()['<s>'])
        len_range = (0, 24)

        print("Generating sentences...")
        model = create_model(train, n=1)
        display_generated(model, 1, len_range)

    if args.bigrams:
        tokens = preprocess(train, n=2)
        print("Size of training vocab: {}".format(len(tokens.vocab())))
        avg_sentence_length = int(len(tokens) / tokens.vocab()['<s>'])
        len_range = (0, avg_sentence_length)

        print("Generating sentences...")
        model = create_model(train, n=2)
        display_generated(model, 2, len_range)

    if args.trigrams:
        tokens = preprocess(train, n=3)
        print("Size of training vocab: {}".format(len(tokens.vocab())))
        avg_sentence_length = int(len(tokens) / (tokens.vocab()['<s>'] / 2))
        len_range = (0, avg_sentence_length)

        print("Generating sentences...")
        model = create_model(train, n=3)
        display_generated(model, 3, len_range)




        
