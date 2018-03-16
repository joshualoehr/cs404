#!/bin/env python

import argparse
import glob
import os
import string
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

punctuation = list(string.punctuation)
punctuation.remove('{')
punctuation.remove('}')
punctuation.append('...')
punctuation.append('--')
punctuation.append('``')
punctuation.append("''")

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

roman_numeral_regex = re.compile('[0-9\.]*[ivx]+')
ordering_regex = re.compile('[0-9]+[\.a-z]*')
special_token_regex = re.compile('^{{.*}}$')
header_regex = re.compile('^{{[hH].*}}$')

def write_output(tokens, output_path):
    with open(output_path, 'w+') as f:
        lines = '\n'.join([' '.join(sent) for sent in tokens])
        f.write(lines)

def merge_special_tokens(tokens):
    new_tokens = []
    special_token = ''
    in_special = False
    for token in tokens:
        if not in_special and token == '{':
            in_special = True
        if not in_special:
            new_tokens.append(token)

        if in_special:
            special_token += token
            if special_token[-2:] == '}}':
                new_tokens.append(special_token)
                in_special = False
                special_token = ''
    return new_tokens
            
def process_punctuation(tokens):
    return [token for token in tokens if token not in punctuation]

def remove_stopwords(tokens):
    return [token for token in tokens if token not in stop_words]

def stem(tokens):
    return [stemmer.stem(token) for token in tokens]

def is_numeric(token):
    if token.isdigit():
        return True
    if roman_numeral_regex.match(token):
        return True
    if ordering_regex.match(token):
        return True
    return False

def replace_nums(tokens):
    return ['{{num}}' if is_numeric(token) else token for token in tokens]

def misc_processing(token):
    if special_token_regex.match(token):
        return token
    if token == "'s":
        return None
    token = token.replace('/', ' ')
    token = token.replace('-', ' ')
    token = token.replace('_', ' ')
    token = token.replace('.', '')
    token = token.replace('`', '')
    tokens = token.split(' ')
    tokens = [token.strip() for token in tokens]
    tokens = ['{{sym}}' if len(token) == 1 else token for token in tokens]
    return ' '.join(tokens)

def preprocess(tokens):
    tokens = [process_punctuation(sent) for sent in tokens]
    tokens = [remove_stopwords(sent) for sent in tokens]
    tokens = [stem(sent) for sent in tokens]
    tokens = [replace_nums(sent) for sent in tokens]
    tokens = [[misc_processing(token) for token in sent] for sent in tokens]
    tokens = [[token for token in sent if token != None] for sent in tokens]
    return tokens

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocess cmp-lg abstract and article texts.')
    parser.add_argument('documents', type=glob.glob,
            help='Glob pattern of documents (directories) to preprocess.')
    args = parser.parse_args()
    
    for doc_dir in args.documents:
        for doc_part in ['article', 'abstract']:
            doc_txt = '{}/{}.txt'.format(doc_dir, doc_part)
            with open(doc_txt, 'r') as f:
                text = f.read()

            tokens = [word_tokenize(sent) for sent in sent_tokenize(text)]
            tokens = [merge_special_tokens(sent) for sent in tokens]
            tokens = [sent for sent in tokens if len(sent) and sent[0] != '.']
            tokens = [sent for sent in tokens if not header_regex.match(sent[0])]

            sents_file = '{}/{}.sentences'.format(doc_dir, doc_part)
            write_output(tokens, sents_file)

            tokens = preprocess(tokens)
            tokens_file = '{}/{}.tokens'.format(doc_dir, doc_part)
            write_output(tokens, tokens_file)


