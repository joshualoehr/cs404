import argparse
import glob
import os
import string
import re
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

def write_output(sents, output_path, headers=True):
    if not headers:
        sents = filter(lambda sent: sent != '{{HED}}', sents)
    sents = (sent.strip() for sent in sents)
    with open(output_path, 'w+') as f:
        lines = '\n'.join(sents)
        f.write(lines)

def remove_stopwords(sent):
    return (token for token in sent if token not in stop_words)

def remove_braces(sent):
    for token in sent:
        if re.match('^({{.{3}}})$', token):
            yield token
        else: 
            yield token.replace('}','').replace('{','')

def replace_numerics(sent):
    for token in sent:
        if re.match('^([0-9]+|[0-9\.]*[iv]+|[0-9]+[\.a-z]*)$', token):
            yield '{{NUM}}'
        else:
            yield token

def replace_symbols(sent):
    for i,token in enumerate(sent):
        if i == 0 and token == 'A':
            yield token
        elif len(token) == 1:
            yield '{{SYM}}'
        elif re.match('^[A-Z]+$', token):
            yield '{{ACR}}'
        elif re.match('^[A-Z0-9]+$', token):
            yield '{{SYM}}'
        elif re.match('^[A-Za-z][0-9ijk]+$', token):
            yield '{{SYM}}'
        else:
            yield token

def stem(sent):
    return (stemmer.stem(token) for token in sent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocess cmp-lg abstract and article texts.')
    parser.add_argument('documents', type=glob.iglob,
            help='Glob pattern of documents (directories) to preprocess.')
    args = parser.parse_args()
    
    for doc_dir in filter(os.path.isdir, args.documents):
        for doc_part in ['article', 'abstract']:
            doc_txt = '{}/{}.txt'.format(doc_dir, doc_part)
            with open(doc_txt, 'r') as f:
                text = f.read()

            for c in set(punctuation) - {'.'}:
                text = text.replace(c, ' ')
            text = text.replace('e.g.', 'eg')
            text = text.replace('i.e.', 'ie')
            text = re.sub(r'[a-z]+\{.*?\}', ' ', text)
            text = re.sub(r'\{\{EQN\}\} (?=[A-Z][a-z]|{{HED}})', '{{EQN}}. ', text)
            text = re.sub(r'(\s+(?!{{HED}}).*?)\s+({{HED}})', '\g<1>. {{HED}}', text)
            sents = re.split('({{EQN}}|\.)\s+({{HED}}|[A-Z][A-Za-z\s{}0-9]+)', text)
            sents = [re.sub('\s+', ' ', sent) for sent in sents if len(sent) and sent[0] != '.']
            
            sents_file = '{}/{}.sentences'.format(doc_dir, doc_part)
            write_output(sents, sents_file, headers=False)

            sents = ((sent.split(' ')) for sent in sents)
            sents = (remove_stopwords(sent) for sent in sents)
            sents = (remove_braces(sent) for sent in sents)
            sents = (replace_numerics(sent) for sent in sents)
            sents = (replace_symbols(sent) for sent in sents)
            sents = (stem(sent) for sent in sents)
            sents = (' '.join(sent) for sent in sents)
            
            tokens_file = '{}/{}.tokens'.format(doc_dir, doc_part)
            write_output(sents, tokens_file)


