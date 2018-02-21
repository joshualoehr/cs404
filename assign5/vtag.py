#!/bin/env python

import argparse
import nltk
import math
from operator import mul
from functools import reduce

def txtfile(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            if lines[-1] == '###/###':
                lines = lines[:-1]
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
        # self.tt_probs = { ('S','S'): 0.8, ('S','V'): 0.2, ('V','S'): 0.4, ('V','V'): 0.6, ('###','S'): 0.5, ('###','V'): 0.5 }
        # self.tw_probs = { ('G','S'): 0.5, ('F','S'): 0.5, ('G','V'): 0.8, ('F','V'): 0.2, ('###','###'): 1.0 }

        print(self.tt_probs)
        print(self.tw_probs)
        log = lambda x: math.log(x)
        tags = [tag for tag in self.tag_counts.keys() if tag != '###']
        # tags = ['###','S','V']

        T = len(sequence)
        N = len(tags)
        viterbi_probs = []
        backpointer = []

        initial_tt_probs = [self.tt_probs.get(('###', tag), 0.0000001) for tag in tags]
        initial_tw_probs = [self.tw_probs.get((sequence[1], tag), 0.0000001) for tag in tags]
        print(list(zip(initial_tt_probs, initial_tw_probs)))
        initial_probs = [reduce(mul, probs) for probs in zip(initial_tt_probs, initial_tw_probs)]
        initial_probs = map(log, initial_probs)
        initial_probs = list(zip(initial_probs, tags))

        viterbi_probs.append(initial_probs)
        print(initial_probs)
        print()

        for j, word in list(enumerate(sequence[1:]))[1:]:
            viterbi_probs.append([(-float("inf"), '')]*N)

            for i, tag in enumerate(tags):

                for prev_tag in tags:
                    tt_prob = log(self.tt_probs.get((prev_tag, tag), 0.0000001)) # 0.0000001 is wrong, store log probabilities to begin with
                    tw_prob = log(self.tw_probs.get((word, tag), 0.0000001))
                    v = viterbi_probs[j-1][i][0] + tt_prob + tw_prob
                    print("{}_{} -> {}_{} = {:.5f} + {} + {} = {:.5f}".format(prev_tag, j-1, tag, j, viterbi_probs[j-1][i][0], tt_prob, tw_prob, v))

                    if v >= viterbi_probs[j][i][0]:
                        viterbi_probs[j][i] = (v, prev_tag)
            print("word:{}, v's:{}".format(word, viterbi_probs[j]))
            print()

        for probs in viterbi_probs:
            print(' '.join(['{:.4f}'.format(prob) for prob,_ in probs]))
        print()

        tag_lookup = { tag: idx for idx,tag in enumerate(tags)}
        sequence_tags = []
        sequence_tags.append(max(viterbi_probs[T-2], key=lambda pair: pair[0])[1])
        for j in reversed(range(0, T-2)):
            i = tag_lookup[sequence_tags[-1]]
            sequence_tags.append(viterbi_probs[j][i][1])
        sequence_tags.append('###')

        return list(reversed(sequence_tags))

        
    def evaluate(self, test_seq, tag_seq):
        sequence = [(word, test_tag, pred_tag) for (word, test_tag), pred_tag in zip(test_seq, tag_seq) ]
        overall_count, known_count, novel_count = 0, 0, 0
        known_total, novel_total = 0, 0
        for word, test_tag, pred_tag in sequence:
            correct = test_tag == pred_tag
            if word in self.word_counts:
                known_total += 1
                known_count += 1 if correct else 0 
            else:
                novel_total += 1
                novel_count += 1 if correct else 0
            overall_count += 1 if correct else 0
        overall = (overall_count / len(sequence)) * 100
        known = (known_count / known_total if known_total > 0 else 1) * 100
        novel = (novel_count / novel_total if novel_total > 0 else 1) * 100

        print("Tagging accuracy (Viterbi decoding): {:.2f}% (known: {:.2f}% novel: {:.2f}%)".format(overall, known, novel))

                
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Viterbi POS Decoder")
    parser.add_argument("train", type=txtfile, 
            help="The tagged training data, with a single word/tag pair per line.")
    parser.add_argument("test", type=txtfile,
            help="The tagged testing data, with a single word/tag pair per line.")
    args = parser.parse_args()

    viterbi = Viterbi(args.train)
    tag_seq = viterbi.decode([word for word,_ in args.test])
    print(tag_seq)
    viterbi.evaluate(args.test, tag_seq)
    
