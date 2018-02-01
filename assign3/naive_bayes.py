#!/bin/env python

###############################################################################
# CSCI 404 - Assignment 3 - Josh Loehr & Robin Cosbey - naive_bayes.py        #
###############################################################################

import argparse
import nltk
from pathlib import Path


def get_filepaths(data_path):
    train_path = data_path.joinpath('train.txt').absolute().as_posix()
    test_path  = data_path.joinpath('test.txt').absolute().as_posix()
    return train_path, test_path

def load_data(data_dir):
    train_path, test_path = get_filepaths(data_dir)
    with open(train_path, 'r') as f:
        train = f.readlines()
    with open(test_path, 'r') as f:
        test = f.readlines()
    return train, test

def preprocess(sentences):
    sos = '<s>'
    eos = '</s>'
    return ['{} {} {}'.format(sos, sentence, eos) for sentence in sentences]

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Naive Bayes Text Classifier")
    parser.add_argument('--data', type=str, required=True,
            help='Location of the data directory containing train.txt and test.txt')
    args = parser.parse_args()

    data_path = Path(args.data)

    # Load and prepare train/test data
    train, test = load_data(data_path)
    train = preprocess(train)
    test  = preprocess(test)



