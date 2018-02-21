#!/bin/env python

import argparse
import nltk

def txtfile(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            if lines[0] == '###/###':
                lines = lines[1:]
            pairs = [tuple(line.split('/')) for line in lines]
            return pairs
    except Exception as e:
        raise argparse.ArgumentError("Error reading input file: {}: {}".format(filepath, e))

class Viterbi(object):

    def __init__(self, train):
        self.train = train
        self.word_counts = nltk.FreqDist([word for word,_ in train]) 
        self.tag_counts  = nltk.FreqDist([tag for _,tag in train])
        self.word_tag_counts = nltk.FreqDist(train)
        self.tt_probs, self.tw_probs = self.calculate_probabilities()
        print(self.tt_probs)

    def calculate_probabilities(self):
        tt_probs = {}
        tw_probs = {}

        tag_bigrams = nltk.bigrams([tag for _,tag in self.train])
        tag_bigram_counts = nltk.FreqDist(tag_bigrams)

        for prev_gram, next_gram in nltk.bigrams(self.train):
            _, prev_tag = prev_gram
            next_word, next_tag = next_gram

            tag_bigram = (prev_tag, next_tag)
            if not tt_probs.get(tag_bigram):
                tt_probs[tag_bigram] = tag_bigram_counts[tag_bigram] / self.tag_counts[prev_tag]

            if not tw_probs.get(next_gram):
                tw_probs[next_gram] = self.word_tag_counts[next_gram] / self.tag_counts[next_tag]

        return tt_probs, tw_probs

    def decode(self, sequence):
        tags = self.tag_counts.keys()

        T = len(sequence)
        N = len(tags)
        viterbi_probs = []
        backpointer = []

        viterbi_probs.append([1] * N)
        backpointer.append(['###'] * N)

        for t in range(1,T):
            viterbi_probs.append([0]*N)
            backpointer.append(['']*N)
            for n, tag in enumerate(tags):
                for prev_tag in tags:
                    p = self.tt_probs.get((prev_tag, tag), 0) * self.tw_probs.get((sequence[t], tag), 0)
                    v = viterbi_probs[t-1][n] * p
                    print("{}_{} -> {}_{} = {} * {}".format(prev_tag, t-1, tag, t, viterbi_probs[t-1][n], p))

                    if v >= viterbi_probs[t][n]:
                        viterbi_probs[t][n] = v
                        backpointer[t][n] = prev_tag

        tag_lookup = { tag: idx for idx,tag in enumerate(tags)}
        sequence_tags = ['###']
        for i in reversed(range(1, T)):
            tag_idx = tag_lookup[sequence_tags[-1]]
            sequence_tags.append(backpointer[i][tag_idx])

        return list(reversed(sequence_tags))

        




                
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Viterbi POS Decoder")
    parser.add_argument("train", type=txtfile, 
            help="The tagged training data, with a single word/tag pair per line.")
    parser.add_argument("test", type=txtfile,
            help="The tagged testing data, with a single word/tag pair per line.")
    args = parser.parse_args()

    viterbi = Viterbi(args.train)
    print(viterbi.decode([word for word,_ in args.test]))

