###############################################################################
# CSCI 404 - Assignment 1 - Josh Loehr & Robin Cosbey                         #
###############################################################################

import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.book import FreqDist
import re

# ntlk.download()

### Part 1 ###

with open('sample.txt', 'r') as f:
    lines = f.read().replace('\n', ' ')

sentences = sent_tokenize(lines)
print("All sentences:")
for sentence in sentences:
    print(sentence)
print("Number of sentences: {}".format(len(sentences)))

### Part 2 ###

def print_top_freq_percents(tokens, n=10):
    total = len(tokens)
    fdist = FreqDist(tokens)
    top = fdist.most_common(n)
    top = [(token, count, 100*count/total) for token, count in top]
    for entry in top:
        print("{:4s} :    {:5d} ({:1.2f}%)".format(*entry))

with open('nc.txt', 'r') as f:
    nc_lines = f.read().replace('\n', ' ').lower()
with open('stopwords.txt', 'r') as f:
    stopwords = [line.replace('\n', '') for line in f.readlines()]

words = word_tokenize(nc_lines)
total = len(words)
print("2.a) ", total)

print('2.b)')
print("10 most common words:")
print_top_freq_percents(words)

print('2.c)')
print("10 most common words (excl. stopwords and punctuations):")
no_stopwords = [word for word in words if word not in stopwords]
r = re.compile('[a-z]+')
no_punct = [word for word in no_stopwords if bool(r.match(word))]
print_top_freq_percents(no_punct)

print('2.d)')
print("10 most frequent token pairs (excl. stopwords and punctuations):")
pairs = [" ".join(pair) for pair in nltk.bigrams(no_punct)]
print_top_freq_percents(pairs)

print('2.e)')
freqs = FreqDist(words)
print("{} total types".format(len(freqs)))

print('2.f)')
print("{:1.3} type/token ratio".format(len(freqs)/total))

print('2.g)')
print('10 most frequent tokens with digits:')
r = re.compile('[0-9]+')
with_digits = [word for word in words if bool(r.match(word))]
print_top_freq_percents(with_digits)

print('2.h)')
print('Number and percentage of singletons:')
singletons = [(word,count) for word,count in freqs.items() if count == 1]
print('{}, {:1.5f}%'.format(len(singletons), len(singletons)/total))

