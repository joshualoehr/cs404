"""
Load all label and feature files. Store labels from all labels and corresponding features
from all documents in .npy array files. Additionally store abstracts and document identifiers
if specified by the user.

python load_data.py --data "data/*_train/*" --set_name train --out_dir npy_data/train --sent_len 5 
python load_data.py --data "data/*_test/*" --set_name test --out_dir npy_data/test/ --sent_len 5 -save_abs

"""

import argparse
import glob
import os
import numpy as np

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=glob.glob, 
        help="List of paths to all files for a given set.")
    parser.add_argument("--set_name", required=True, type=str,
        help="Name of dataset to load (e.g. train).")
    parser.add_argument("--out_dir",  required=True, type=str, 
        help="Directory to write binary files to.")
    parser.add_argument("--sent_len", required=True, type=int)
    parser.add_argument("-save_abs", action='store_true')
 
    return parser.parse_args()

def getFiles(data):
    label_files = sorted([l for l in data if ".labels" in l])
    feat_files = sorted([f for f in data if ".feats" in f])
    
    document_names = [label_name.split("/")[1] for label_name in label_files]
    return label_files, feat_files, document_names

def getAbstractFiles(data): 
    abs_files = sorted([a for a in data if ".abs" in a])
    doc_files = sorted([d for d in data if ".sentences" in d])
    return abs_files, doc_files

def loadFeatures(feat_files, max_len):
    all_feats = []
    for f in feat_files:
        with open(f, 'r') as feats:
            all_lines = []
            for line in feats:
                line = line.strip().split()
                curr_len = len(line)
                line += ["0"]*(max_len-curr_len)
                all_lines.append(line)
        all_feats.append(np.array(all_lines))
    return all_feats

def loadLabels(l_files):
    """
    For each label and feature pair, load as np array and store the length of the files.
    Once max length is found, pad remaining files and return np arrays as well as lengths.
    """
    all_labels, all_lens= [], []
    max_len = 0
    for label in l_files:
        curr_l = np.loadtxt(label)
        all_labels.append(curr_l)
        all_lens.append(curr_l.shape[0])
    return all_labels, all_lens

def padFiles(all_labels, all_feats, max_len):
    """
    Given a maximum length, pad label and feature files with 0s.
    Return a label and feature array. 
    labels = number of documents x max length
    feats = number of documents x max length x number of features
    """
    for i in range(len(all_labels)):
        pad = max_len - len(all_labels[i])
        all_labels[i] = np.concatenate((all_labels[i], np.zeros((pad))))
        all_feats[i] = np.concatenate((all_feats[i], np.zeros((pad, all_feats[i].shape[1]))))
    
    labels = np.stack(all_labels)
    feats = np.stack(all_feats)
    return labels, feats


def makeBinaries(outfile, labels, feats, all_lens):
    """
    Saves the label, feature, and length arrays provided as .npy files.
    """
    label_out = outfile + "_labels.npy"
    feat_out = outfile + "_feats.npy"    
    len_out = outfile + "_lens.npy"
    np.save(label_out, labels)
    np.save(feat_out, feats)
    np.save(len_out, all_lens)    

def loadAbstracts(a_files, d_files):
    """
    Store the contents of each abstract file as an element of a list.
    """
    all_abs = []
    all_docs = []
    for abstract in a_files:
        with open(abstract, 'r') as f:
            all_abs.append(len(f.readlines()))
    for document in d_files:
        with open(document, 'r') as f:
            all_docs += [[sentence.strip() for sentence in f]]
    return all_abs, all_docs

def makeAbstractBinary(outfile, abstracts, documents, names):
    """
    Save the abstract provided and corresponding document identifier as .npy files.
    """
    abs_out = outfile + "_abs.npy"
    doc_out = outfile + "_docs.npy"
    nam_out = outfile + "_names.npy"
    np.save(abs_out, abstracts)
    np.save(doc_out, documents)
    np.save(nam_out, names)

def main():
    args = parseArgs()
    out_dir = args.out_dir + "/" if args.out_dir[-1] != "/" else args.out_dir
    outfile = out_dir + args.set_name 

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
 
    label_files, feat_files, names = getFiles(args.data)
    if args.save_abs:
        abs_files, doc_files = getAbstractFiles(args.data)
        abstracts, documents = loadAbstracts(abs_files, doc_files)
        makeAbstractBinary(outfile, abstracts, documents, names)
    all_labels, all_lens = loadLabels(label_files)
    
    all_feats = loadFeatures(feat_files, args.sent_len)
    labels, feats = padFiles(all_labels, all_feats, max(all_lens))
    makeBinaries(outfile, labels, feats, all_lens)
    
if __name__ == "__main__":
    main()
