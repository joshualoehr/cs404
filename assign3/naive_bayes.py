#!/bin/env python

###############################################################################
# CSCI 404 - Assignment 3 - Josh Loehr & Robin Cosbey - naive_bayes.py        #
###############################################################################

import argparse
from collections import Counter
from itertools import product
import math
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
    sos = "<s> " * max(n-1, 1) 
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
        vocab = tokens.vocab()
        total_tokens = len(tokens)
        return { (unigram,): count/total_tokens for unigram, count in vocab.items() }
    if n == 2:
        return smooth([(token,) for token in tokens], nltk.bigrams(tokens), len(tokens.vocab()))
    if n == 3:
        return smooth(nltk.bigrams(tokens), nltk.trigrams(tokens), len(tokens.vocab()))

def most_probable_tokens(model, prev, num_candidates=1, blacklist=[]):
    blacklist += ["<UNK>", "<s>"]
    filter = lambda ngram: ngram[-1] not in blacklist
    
    candidates = [(ngram[-1], prob) for ngram, prob in model.items() if ngram[:-1] == prev and filter(ngram)]
    candidates = sorted(candidates, key=lambda pair: pair[1], reverse=True)[:num_candidates]
    return candidates 
    
def generate(model, n, len_range, num=10):
    min_length, max_length = len_range

    sentences = ["<s>"] * num
    probabilities = [1] * num

    prev = tuple(["<s>"]*(n-1))
    sos_grams = most_probable_tokens(model, prev, num_candidates=num, blacklist=["</s>"])
    sentences = list(zip(sentences, [ngram for ngram, _ in sos_grams]))
    sentences = [list(sentence) for sentence in sentences]
    probabilities = [probabilities[i] * prob for i, (_, prob) in enumerate(sos_grams)]

    for i in range(num):
        while sentences[i][-1] != "</s>" and len(sentences[i]) < max_length:
            prev = () if n==1 else tuple(sentences[i][-(n-1):]) 
            blacklist = []
            if n == 1:
                blacklist += sentences[i]
            if len(sentences[i]) < min_length + 2:
                blacklist.append("</s>")
            next_token, next_prob = most_probable_tokens(model, prev, blacklist=blacklist)[0]
            sentences[i].append(next_token)
            probabilities[i] *= next_prob
    
    return zip(sentences, probabilities)

def display_generated(model, n, len_range=(0,25)):
    generated = generate(model, n, len_range=len_range)
    for sentence, prob in generated:
        print("{} ({})".format(sentence, math.log(prob)))

def compute_perplexity(model, tokens, n):
    flatten = lambda s: (s,) if type(s) is str else s
    all_masks = lambda n: reversed(list(product((0,1), repeat=n)))
    mask = lambda ngram, bitmask: tuple((token if bit == 1 else "<UNK>" for token, bit in zip(ngram, bitmask)))
    
    tokens = nltk.Text(tokens)
    bitmasks = list(all_masks(n))

    probabilities = []
    for token in tokens:
        for ngram in [mask(flatten(token), bitmask) for bitmask in bitmasks]:
            if ngram in model:
                # print(ngram, model[ngram])
                probabilities.append(model[ngram])
                break
    N = len(tokens)
    print("num probs", len(probabilities), " | num tokens:", N)
    return math.exp((-1/N)*sum(map(math.log, probabilities)))


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
        avg_sentence_length = int(len(tokens) / tokens.vocab()['<s>'])
        len_range = (1, avg_sentence_length)

        print("Generating unigram sentences...")
        model = create_model(tokens, n=1)
        # display_generated(model, 1, len_range)

        test_tokens = preprocess(test, n=1)
        perplexity = compute_perplexity(model, test_tokens, 1)
        print("Unigram perplexity: {}".format(perplexity))

    if args.bigrams:
        tokens = preprocess(train, n=2)
        print("Size of training vocab: {}".format(len(tokens.vocab())))
        avg_sentence_length = int(len(tokens) / tokens.vocab()['<s>'])
        len_range = (1, avg_sentence_length)

        print("Generating bigram sentences...")
        model = create_model(tokens, n=2)
        # display_generated(model, 2, len_range)
        
        test_tokens = preprocess(test, n=2)
        perplexity = compute_perplexity(model, nltk.bigrams(test_tokens), 2)
        print("Bigram perplexity: {}".format(perplexity))

    if args.trigrams:
        tokens = preprocess(train, n=3)
        print("Size of training vocab: {}".format(len(tokens.vocab())))
        avg_sentence_length = int(len(tokens) / (tokens.vocab()['<s>'] / 2))
        len_range = (1, avg_sentence_length)

        print("Generating trigram sentences...")
        model = create_model(tokens, n=3)
        # display_generated(model, 3, len_range)

        test_tokens = preprocess(test, n=3)
        perplexity = compute_perplexity(model, nltk.trigrams(test_tokens), 3)
        print("Trigram perplexity: {}".format(perplexity))

        


        
