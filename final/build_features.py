
import argparse
from collections import Counter
import glob
import os


class ID(dict):
    def __call__(self, token):
        if token not in self:
            self[token] = [len(self) + 1, 0]
        self[token][1] += 1
        return self[token]


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Builds feature vectors from preprocessed documents.")
    parser.add_argument('documents', type=glob.glob, 
            help='The documents from which to build feature vectors.')
    parser.add_argument('--max-vocab', type=int, default=None,
            help='Maximum number of words to allow in the vocabulary. Default is unlimited.')
    args = parser.parse_args()

    token_map = ID()
    features = {}
    first_sents = {}

    for doc in filter(os.path.isdir, args.documents):
        features[doc] = []
        first_sents[doc] = []
        header_id = -1
        first_sent = False

        doc_sentences = '{}/article.sentences'.format(doc)
        with open(doc_sentences, 'r') as f:
            sentences = f.read().split('\n')
            sentences = [sent.split(' ') for sent in sentences]
            sent_lens = map(len, sentences)

        doc_tokens_file = '{}/article.tokens'.format(doc)
        with open(doc_tokens_file, 'r') as f:
            sents = f.read().split('\n')
    
            for idx,(sent,sent_len) in enumerate(zip(sents, sent_lens)):
                if sent == '{{hed}}':
                    header_id += 1
                    first_sent = True
                    continue
                if first_sent:
                    idx = idx - (header_id + 1) # adjust for number of headers seen
                    first_sents[doc].append(idx)
                    first_sent = False
                tokens = sent.split(' ')
                for token in tokens:
                    token_map(token)

                features[doc].append((header_id, sent_len, tokens))


    for doc in filter(os.path.isdir, args.documents):
        doc_features_file = '{}/article.features'.format(doc)
        with open(doc_features_file, 'w+') as f:
            doc_features = []
            for header_id, sent_len, tokens in features[doc]:
                token_ids = (token_map[token] for token in tokens)
                token_ids = [0 if count == 1 else idx for idx,count in token_ids]
                feature = [header_id, sent_len] + token_ids
                feature = ' '.join(map(str, feature))
                doc_features.append(feature)
            f.write('\n'.join(doc_features))

        doc_first_sents_file = '{}/article.baseline'.format(doc)
        with open(doc_first_sents_file, 'w+') as f:
            f.write('\n'.join(map(str, first_sents[doc])))

