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
                tt_probs[tag_bigram] = (tag_bigram_counts[tag_bigram]+1) / (self.tag_counts[prev_tag]+len(self.tag_counts))

            if not tw_probs.get(next_gram):
                if next_gram == ('###','###'):
                    tw_probs[next_gram] = self.word_tag_counts[next_gram] / self.tag_counts[next_tag]
                else:
                    tw_probs[next_gram] = (self.word_tag_counts[next_gram]+1) / (self.tag_counts[next_tag]+len(self.tag_counts))

        return tt_probs, tw_probs

    def possible_tags(self, w):
        tags = [tag for (word,tag),count in self.word_tag_counts.items() if word == w]
        if not tags:
            return [tag for tag in self.tag_counts if tag != '###']
        else:
            return tags

    def unknown_prob(self, tag):
        return 1 / (self.tag_counts[tag] + len(self.tag_counts))

    def decode(self, sequence):
        sequence = sequence + ['###']

        log = lambda x: math.log(x)
        tags = [tag for tag in self.tag_counts.keys()]

        T = len(sequence)
        N = len(tags)
        viterbi_probs = []

        viterbi_probs.append({'###': (1, '')})
        for j in range(1, T):
            viterbi_probs.append({ tag: (-float("inf"), '') for tag in tags })

            word = sequence[j]
            tag_dict = self.possible_tags(word)

            for tag in tag_dict:

                prev_word = sequence[j-1]
                prev_tag_dict = self.possible_tags(prev_word)

                for prev_tag in prev_tag_dict:

                    tt_prob = self.tt_probs.get((prev_tag, tag), self.unknown_prob(tag))
                    tw_prob = self.tw_probs.get((word, tag), self.unknown_prob(tag))
                    p = log(tt_prob * tw_prob)
                    v = viterbi_probs[j-1][prev_tag][0] + p
                    if v > viterbi_probs[j][tag][0]:
                        viterbi_probs[j][tag] = (v, prev_tag)
                    

        sequence_tags = ['###']
        tag = '###'
        for j in reversed(range(1, T)):
            _,prev_tag = viterbi_probs[j][tag]
            tag = prev_tag
            sequence_tags.append(tag)

        return list(reversed(sequence_tags[1:]))

        
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
    
