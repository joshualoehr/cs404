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

    def __init__(self, train, raw, smoothing="onecount"):
        self.train = train
        self.raw_vocab = nltk.FreqDist([word for word,_ in train] + raw)
        self.sing_tt_counts = {}
        self.sing_tw_counts = {}
        self.word_counts = nltk.FreqDist([word for word,_ in train]) 
        self.tag_counts  = nltk.FreqDist([tag for _,tag in train])
        self.word_tag_counts = nltk.FreqDist(train)
        
        self.smoothing = smoothing
        if not smoothing:
            self.tt_smooth, self.tw_smooth = self.no_tt_smooth, self.no_tw_smooth
        elif smoothing == 'onecount':
            self.tt_smooth, self.tw_smooth = self.onecount_tt_smooth, self.onecount_tw_smooth
        elif smoothing == 'addone':
            self.tt_smooth, self.tw_smooth = self.addone_tt_smooth, self.addone_tw_smooth

        self.tt_probs, self.tw_probs = self.calculate_probabilities()
        
    def sing_tt(self, prev_tag):
        if prev_tag not in self.sing_tt_counts:
            singletons = [tag for tag,count in self.tag_counts.items() if tag == prev_tag and count == 1]
            self.sing_tt_counts[prev_tag] = len(singletons)
        return self.sing_tt_counts[prev_tag]

    def sing_tw(self, tag):
        if tag not in self.sing_tw_counts:
            singletons = [t for (word,t),count in self.word_tag_counts.items() if t == tag and count == 1]
            self.sing_tw_counts[tag] = len(singletons)
        return self.sing_tw_counts[tag]

    def backoff_tt_prob(self, tag):
        return self.tag_counts[tag] / len(self.train)

    def backoff_tw_prob(self, word):
        return (self.word_counts[word] + 1) / (len(self.train) + len(self.raw_vocab) + 1)

    def calculate_probabilities(self):
        tt_probs, tw_probs = {}, {}

        tag_bigrams = nltk.bigrams([tag for _,tag in self.train])
        self.tag_bigram_counts = nltk.FreqDist(tag_bigrams)

        for prev_gram, next_gram in nltk.bigrams(self.train):
            _, prev_tag = prev_gram
            next_word, next_tag = next_gram
            tag_bigram = (prev_tag, next_tag)
            
            if not tt_probs.get(tag_bigram):
                tt_probs[tag_bigram] = self.tt_smooth(tag_bigram)
            if not tw_probs.get(next_gram):
                tw_probs[next_gram] = self.tw_smooth(next_gram)

        return tt_probs, tw_probs

    def no_tt_smooth(self, tag_bigram):
        prev_tag, _ = tag_bigram
        return self.tag_bigram_counts[tag_bigram] / self.tag_counts[prev_tag]

    def no_tw_smooth(self, next_gram):
        _, next_tag = next_gram
        return self.word_tag_counts[next_gram] / self.tag_counts[next_tag]

    def addone_tt_smooth(self, tag_bigram):
        prev_tag, _ = tag_bigram
        vocab_size = len(self.tag_counts)
        return (self.tag_bigram_counts[tag_bigram] + 1) / (self.tag_counts[prev_tag] + vocab_size)

    def addone_tw_smooth(self, next_gram):
        _, next_tag = next_gram
        vocab_size = len(self.tag_counts)
        if next_gram == ('###','###'):
            return self.no_tw_smooth(next_gram)
        else:
            return (self.word_tag_counts[next_gram] + 1) / (self.tag_counts[next_tag] + vocab_size)

    def onecount_tt_smooth(self, tag_bigram):
        prev_tag, next_tag = tag_bigram
        lam = 1 + self.sing_tt(prev_tag)
        backoff = lam * self.backoff_tt_prob(next_tag)
        return (self.tag_bigram_counts[tag_bigram] + backoff) / (self.tag_counts[prev_tag] + lam)

    def onecount_tw_smooth(self, next_gram):
        next_word, next_tag = next_gram
        if next_gram == ('###','###'):
            return self.no_tw_smooth(next_gram)
        else:
            lam = 1 + self.sing_tw(next_tag)
            backoff = lam * self.backoff_tw_prob(next_word)
            return (self.word_tag_counts[next_gram] + backoff) / (self.tag_counts[next_tag] + lam)

    def possible_tags(self, w):
        tags = [tag for (word,tag),count in self.word_tag_counts.items() if word == w]
        return tags if tags else [tag for tag in self.tag_counts if tag != '###']

    def unknown_tt_prob(self, prev_tag, next_tag):
        if not self.smoothing:
            return -1
        if self.smoothing == 'addone':
            return 1 / (self.tag_counts[next_tag] + len(self.tag_counts))
        elif self.smoothing == 'onecount':
            lam = 1 + self.sing_tt(prev_tag)
            backoff = lam * self.backoff_tt_prob(next_tag)
            return backoff / (self.tag_counts[prev_tag] + lam)

    def unknown_tw_prob(self, word, tag):
        if not self.smoothing:
            return -1
        if self.smoothing == 'addone':
            return 1 / (self.tag_counts[tag] + len(self.tag_counts))
        elif self.smoothing == 'onecount':
            lam = 1 + self.sing_tw(tag)
            backoff = lam * self.backoff_tw_prob(word)
            return backoff / (self.tag_counts[tag] + lam)

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

                    tt_prob = self.tt_probs.get((prev_tag, tag), self.unknown_tt_prob(prev_tag, tag))
                    tw_prob = self.tw_probs.get((word, tag), self.unknown_tw_prob(word, tag))
                    p = log(tt_prob * tw_prob)
                    v = viterbi_probs[j-1][prev_tag][0] + p
                    if v >= viterbi_probs[j][tag][0]:
                        viterbi_probs[j][tag] = (v, prev_tag)

        sequence_tags = ['###']
        tag = '###'
        for j in reversed(range(1, T)):
            _,prev_tag = viterbi_probs[j][tag]
            tag = prev_tag
            sequence_tags.append(tag)

        return list(reversed(sequence_tags[1:]))

        
    def evaluate(self, test_seq, tag_seq):
        sequence = [(word, test_tag, pred_tag) for (word, test_tag), pred_tag in zip(test_seq, tag_seq) if word != '###']

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
    parser.add_argument("raw", type=txtfile,
            help="The raw untagged data, with a single word per line.")
    parser.add_argument("--smoothing", type=str, choices=["addone", "onecount"], required=False)
    args = parser.parse_args()

    viterbi = Viterbi(args.train, args.raw, smoothing=args.smoothing)
    tag_seq = viterbi.decode([word for word,_ in args.test])
    viterbi.evaluate(args.test, tag_seq)
    
