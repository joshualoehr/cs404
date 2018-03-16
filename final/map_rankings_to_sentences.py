import argparse
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Map sentence rankings back to the original sentences.')
    parser.add_argument('documents', type=glob.iglob,
            help='The documents (directories) for which to map rankings and sentences.')
    parser.add_argument('--ranking-file', type=str, choices=['labels'], default='labels',
            help='Extension of the file(s) containing the rankings to map against.')
    args = parser.parse_args()

    for doc_dir in args.documents:
        doc_sents = '{}/article.sentences'.format(doc_dir)
        with open(doc_sents, 'r') as f:
            sents = f.read().split('\n')

        doc_ranks = '{}/article.{}'.format(doc_dir, args.ranking_file)
        with open(doc_ranks, 'r') as f:
            ranks = f.read().split('\n')

        sent_rank_to_idx = { int(rank): idx for idx, rank in enumerate(ranks) }
        ranked_sents = []

        for rank in range(len(sents)):
            sent_idx = sent_rank_to_idx[rank]
            sent = sents[sent_idx]
            ranked_sents.append(sent)

        doc_summary = '{}/article.summary'.format(doc_dir)
        with open(doc_summary, 'w+') as f:
            f.write('\n'.join(ranked_sents))

