#!/bin/env python

################################################################################
# CSCI 404 - Assignment 4 - Josh Loehr & Robin Cosbey - eval.py                #
################################################################################

import argparse
import glob
import os
import pickle

from naive_bayes import NaiveBayesClassifier

def valid_model(filename):
    """Determine if a given model file is valid.

    File must:
        - exist
        - be readable
        - be a pickle file which can be successfully loaded
    
    Args:
        filename (str): Path to the model file to validate.
    Returns:
        (NaiveBayesClassifier) The model, loaded from its pickle file.
    """
    err_msg = None
    if not os.path.exists(filename) or not os.path.isfile(filename):
        err_msg = 'not a file'
    elif not os.access(filename, os.R_OK):
        err_msg = 'not a readable file'
    if not err_msg:
        try:
            return load_model(filename)
        except:
            err_msg = 'not a valid pickle file'
    raise argparse.ArgumentTypeError('{} - {}'.format(filename, err_msg))


def valid_dir(dirname):
    """Determine if a given directory path is valid.
    
    Args:
        dirname (str): The path of the directory to validate.
        required (str): Name of subdirectory required for this directory
            to be considered valid.
    """
    err_msg = None
    if not os.path.exists(dirname) or not os.path.isdir(dirname):
        err_msg = 'not a directory'
    elif not os.access(dirname, os.R_OK):
        err_msg = 'not a readable directory'
    if err_msg:
        raise argparse.ArgumentTypeError('{} - {}'.format(dirname, err_msg))
    return dirname


def load_model(filepath):
    """Load the saved model pickle file.
    
    Args:
        filepath (str): Path to the pickle file to load.
    Returns:
        The contents of the pickle file.
    """
    with open(filepath, 'rb') as m:
        return pickle.load(m)


def load_docs(dirpath):
    """Load all documents in a directory.
    
    Args:
        dirpath (str): Path to the directory containing the documents.
    Returns:
        (list of list of str) The documents split by words.
    """
    docs = []
    for filepath in glob.glob('{}/*.txt'.format(dirpath)):
        with open(filepath, 'r') as f:
            docs.append(f.read().strip().split(' '))
    return docs


def output(true_pos, true_neg, false_pos, false_neg):
    """Ouput the TP, TN, FP, and FN results in a 2x2 table."""
    print()
    print('Evaluation of Naive Bayes Classifer:')
    print()
    print('      {:^6s}   {:^6s}'.format('True', 'False'))
    print('    +-{}-+-{}-+'.format('-'*6, '-'*6))
    print('Pos | {:^6d} | {:^6d} |'.format(true_pos, false_pos))
    print('    +-{}-+-{}-+'.format('-'*6, '-'*6))
    print('Neg | {:^6d} | {:^6d} |'.format(true_neg, false_neg))
    print('    +-{}-+-{}-+'.format('-'*6, '-'*6))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Naive Bayes Classifier - Evaluation Script')
    parser.add_argument('--model', type=valid_model, required=True,
            help='The model to evaluate, saved in a pickle file.')
    parser.add_argument('--test-positives', '-p', type=valid_dir, required=True,
            help='Directory containing the positive test documents (i.e. spam).')
    parser.add_argument('--test-negatives', '-n', type=valid_dir, required=True,
            help='Directory containing the negative test documents (i.e. nonspam).')
    args = parser.parse_args()

    # Load the model, and spam/nonspam documents
    model = args.model
    spam_docs = load_docs(args.test_positives)
    nonspam_docs = load_docs(args.test_negatives)

    # Classify the documents
    spam_results = map(model.classify, spam_docs)
    nonspam_results = map(model.classify, nonspam_docs)

    # Count results from true-spam documents
    true_pos = len([result for result in spam_results if result == 'spam'])
    false_neg = len(spam_docs) - true_pos

    # Count results from true-nonspam documents
    true_neg = len([result for result in spam_results if result == 'nonspam'])
    false_pos = len(nonspam_docs) - true_neg
    
    # Print the results in a table
    output(true_pos, true_neg, false_pos, false_neg)
