#!/bin/env python

###############################################################################
# CSCI 404 - Assignment 3 - Josh Loehr & Robin Cosbey - naive_bayes.py        #
###############################################################################

import argparse
from itertools import product
import math
import nltk
from pathlib import Path

from preprocess import preprocess

def load_data(data_dir):
    train_path = data_dir.joinpath('train.txt').absolute().as_posix()
    test_path  = data_dir.joinpath('test.txt').absolute().as_posix()

    with open(train_path, 'r') as f:
        train = [l.replace('\n', '') for l in f.readlines()]
    with open(test_path, 'r') as f:
        test = [l.replace('\n', '') for l in f.readlines()]
    return train, test


class LanguageModel:

    def __init__(self, train_data, n, laplace=1, min_sent_length=12, max_sent_length=24):
        self.n = n
        self.laplace = laplace
        self.tokens = preprocess(train_data, n)
        self.model  = self.create_model()
        self.masks  = list(reversed(list(product((0,1), repeat=n))))
        self.min_sent_length = min_sent_length 
        self.max_sent_length = max_sent_length

    def smooth(self):
        vocab_size = len(self.tokens.vocab())

        n_grams = nltk.ngrams(self.tokens, self.n)
        n_vocab = nltk.FreqDist(n_grams)

        m_grams = nltk.ngrams(self.tokens, self.n-1)
        m_vocab = nltk.FreqDist(m_grams)

        def smoothed_count(n_gram, n_count):
            m_gram = n_gram[:-1]
            m_count = m_vocab[m_gram]
            return (n_count+1) / (m_count + self.laplace * vocab_size)

        return { n_gram: smoothed_count(n_gram, count) for n_gram, count in n_vocab.items() }

    def create_model(self):
        if self.n == 1:
            vocab = self.tokens.vocab()
            num_tokens = len(self.tokens)
            return { (unigram,): count / num_tokens for unigram, count in vocab.items() }
        else:
            return self.smooth()

    def convert_oov(self, ngram):
        mask = lambda ngram, bitmask: tuple((token if flag == 1 else "<UNK>" for token,flag in zip(ngram, bitmask)))

        ngram = (ngram,) if type(ngram) is str else ngram
        for possible_known in [mask(ngram, bitmask) for bitmask in self.masks]:
            if possible_known in self.model:
                return possible_known

    def perplexity(self, test_data):
        test_tokens = preprocess(test_data, self.n)
        test_ngrams = nltk.ngrams(test_tokens, self.n)
        N = len(test_tokens)

        known_ngrams  = (self.convert_oov(ngram) for ngram in test_ngrams)
        probabilities = [self.model[ngram] for ngram in known_ngrams]

        return math.exp((-1/N) * sum(map(math.log, probabilities)))

    def generate_sentences(self, num):
        for i in range(num):
            sent, prob = ["<s>"] * max(1, self.n-1), 1
            while sent[-1] != "</s>":
                prev = () if self.n == 1 else tuple(sent[-(self.n-1):])
                blacklist = sent + (["</s>"] if len(sent) < self.min_sent_length else [])
                next_token, next_prob = self.best_candidate(prev, i, without=blacklist)
                sent.append(next_token)
                prob *= next_prob
                
                if len(sent) >= self.max_sent_length:
                    sent.append("</s>")

            yield ' '.join(sent), -1/math.log(prob)

    def best_candidate(self, prev, i, without=[]):
        blacklist  = ["<UNK>"] + without
        candidates = ((ngram[-1],prob) for ngram,prob in self.model.items() if ngram[:-1]==prev)
        candidates = filter(lambda candidate: candidate[0] not in blacklist, candidates)
        candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
        if len(candidates) == 0:
            return ("</s>", 1)
        else:
            return candidates[0 if prev != () and prev[-1] != "<s>" else i]
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Naive Bayes Text Classifier")
    parser.add_argument('--data', type=str, required=True,
            help='Location of the data directory containing train.txt and test.txt')
    parser.add_argument('--laplace', type=float, default=0.01,
            help='Lambda parameter for Laplace smoothing (use 1 for add-1 smoothing)')
    args = parser.parse_args()

    # Load and prepare train/test data
    data_path = Path(args.data)
    train, test = load_data(data_path)

    for n, model_type in zip([3], ["Trigram"]):
        print("Loading {} model...".format(model_type))
        lm = LanguageModel(train, n, laplace=args.laplace)
        print("Generating {} sentences...".format(model_type))
        for sentence, prob in lm.generate_sentences(10):
            print("{} ({:.5f})".format(sentence, prob))
        perplexity = lm.perplexity(test)
        print("{} model perplexity: {:.3f}".format(model_type, perplexity))
        print("")

