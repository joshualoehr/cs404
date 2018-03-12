"""
Load all label and feature files. Store labels from all labels and corresponding features
from all documents in .npy array files. Additionally store abstracts and document identifiers
if specified by the user.

"""

import argparse
import glob
import os
import os.path as osp
import numpy as np

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_files", required=True, type=glob.glob, help="List of paths to all label files for a given set.")
    parser.add_argument("--feat_files", required=True, type=glob.glob, help="List of paths to all feature files for a given set.")
    parser.add_argument("--abs_files", default=False, type=glob.glob, help="List of paths to all tokenized abstract files for a given set.")
    parser.add_argument("--set_name", required=True, type=str, help="Name of dataset to load (e.g. train).")
    parser.add_argument("--out_dir",  required=True, type=str, help="Directory to write binary files to.")
    return parser.parse_args()

def padFiles(all_labels, all_feats,  max_len):
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

def loadFiles(l_files, f_files):
    """
    For each label and feature pair, load as np array and store the length of the files.
    Once max length is found, pad remaining files and return np arrays as well as lengths.
    """
    all_labels,all_feats,all_lens= [], [], []
    max_len = 0
    for label, feat in zip(l_files,f_files):
        curr_l = np.loadtxt(label)
        curr_f = np.loadtxt(feat)
        all_labels.append(curr_l)
        all_feats.append(curr_f)
        all_lens.append(curr_l.shape[0])
        max_len = curr_l.shape[0] if curr_l.shape[0] > max_len else max_len
    labels, feats = padFiles(all_labels, all_feats, max_len)
    return labels, feats, all_lens

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

def loadAbstracts(a_files):
    """
    Store the contents of each abstract file as an element of a list.
    """
    all_abs = []
    for abstract in a_files:
        with open(abstract, 'r') as f:
            all_abs += [[sentence.strip() for sentence in f]]
    return all_abs

def makeAbstractBinary(outfile, abstracts, doc_names):
    """
    Save the abstract provided and corresponding document identifier as .npy files.
    """
    abs_out = outfile + "_abs.npy"
    doc_out = outfile + "_docs.npy"
    np.save(abs_out, abstracts)
    np.save(doc_out, doc_names)

def main():
    args = parseArgs()
    out_dir = args.out_dir + "/" if args.out_dir[-1] != "/" else args.out_dir
    outfile = out_dir + args.set_name 

    # create outfile if need be

    label_files = sorted(args.label_files)
    document_names = [label_name.split("/")[1] for label_name in label_files]
    feat_files = sorted(args.feat_files)
    if args.abs_files:
        abs_files = sorted(args.abs_files)    
        abstracts = loadAbstracts(abs_files)
        makeAbstractBinary(outfile, abstracts, document_names)
    labels, feats, all_lens = loadFiles(label_files, feat_files)
    makeBinaries(outfile, labels, feats, all_lens)
    
if __name__ == "__main__":
    main()
