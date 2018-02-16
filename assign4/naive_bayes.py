#!/bin/env python

import argparse
import glob
import nltk
import itertools
from math import log
import pickle
from functools import reduce
from operator import mul

def load_data(datapath, dataset):
    spam=[] 
    nonspam=[]
    for filename in glob.glob('{}/{}-{}/*.txt'.format(datapath, "spam", dataset)):
        with open(filename, 'r') as f:
            spam.append(f.read().strip().split(' '))
    for filename in glob.glob('{}/{}-{}/*.txt'.format(datapath, "nonspam", dataset)):
        with open(filename, 'r') as f:
            nonspam.append(f.read().strip().split(' '))
    return (spam, nonspam)

class NaiveBayesClassifier(object):

    def __init__(self, train, test, num_features=2500):
        self.train_documents = dict(spam=train[0], nonspam=train[1])
        self.train_vocab = self.build_vocab(*train)
        test_vocab = self.build_vocab(*test)
        
        self.vocabulary = nltk.FreqDist()
        for vocab in itertools.chain(self.train_vocab.values(), test_vocab.values()):
            for word,count in vocab.items():
                self.vocabulary[word] += count

        self.features = [word for word,_ in self.vocabulary.most_common(num_features)]
        self.total_counts = {}
        self.prior_probs  = {}

    def build_vocab(self, spam, nonspam):
        vocabulary = nltk.ConditionalFreqDist()
        for label, documents in zip(['spam', 'nonspam'], [spam, nonspam]):
            for doc in documents:
                for word in doc:
                    vocabulary[label][word] += 1
        return vocabulary 

    def document_features(self, document):
        fdist = nltk.FreqDist(document)
        doc_features = { word: 0 for word in self.features }
        for word in doc_features:
            if word in fdist:
                doc_features[word] = fdist[word]
        return doc_features

    def count(self, word, cls):
        return self.train_vocab[cls][word] + 1

    def total_count(self, cls):
        if not self.total_counts.get(cls):
            self.total_counts[cls] = sum([self.count(word, cls) for word in self.features])
        return self.total_counts[cls] + len(self.features)

    def prior_prob(self, cls):
        if not self.prior_probs.get(cls):
            class_total = len(self.train_documents[cls])
            doc_total = sum(map(len, self.train_documents.values()))
            self.prior_probs[cls] = class_total / doc_total
        return log(self.prior_probs[cls])
    
    def word_prob(self, word, cls):
        return log(self.count(word, cls) / self.total_count(cls))

    def document_prob(self, cls, document):
        return sum([self.word_prob(word, cls) for word,count in self.document_features(document).items() if count > 0])

    def classify(self, document):
        return max(['spam', 'nonspam'], key=lambda c: self.prior_prob(c) + self.document_prob(c, document))

    def save(self, filepath):
        with open(filepath, 'wb+') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Naive Bayes Classifier')
    parser.add_argument('--data', type=str, required=True,
            help='Path to the directory containing training and testing data subdirectories.')
    parser.add_argument('--model-name', type=str, default='classifier.pkl',
            help='Name to save model as.')
    args = parser.parse_args()

    train = load_data(args.data, "train")
    test = load_data(args.data, "test")

    classifier = NaiveBayesClassifier(train, test)
    classifier.save(args.model_name)

