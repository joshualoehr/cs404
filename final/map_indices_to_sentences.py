import argparse
import glob
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Map sentence indices back to the original sentences.')
    parser.add_argument('documents', type=glob.iglob,
            help='The documents (directories) for which to map indices and sentences.')
    parser.add_argument('--index-file', type=str, choices=['baseline'], default='baseline',
            help='Extension of the file(s) containing the indices to map against.')
    args = parser.parse_args()

    for doc_dir in filter(os.path.isdir, args.documents):
        doc_sents = '{}/article.sentences'.format(doc_dir)
        with open(doc_sents, 'r') as f:
            sents = f.read().split('\n')

        doc_indices = '{}/article.{}'.format(doc_dir, args.index_file)
        with open(doc_indices, 'r') as f:
            indices = f.read().split('\n')
            
        idxs = set([int(idx) for idx in indices])
        sents = [sent for idx,sent in enumerate(sents) if idx in idxs]

        doc_summary = '{}/article.baseline_summary'.format(doc_dir)
        with open(doc_summary, 'w+') as f:
            f.write('\n'.join(sents))

