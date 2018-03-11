"""
Load all label and feature files. Store labels from all documents and corresponding features
from all documents in .npy array files.

"""

import argparse
import os
import os.path as osp
import numpy as np

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str, help="Directory containing label and feature files.")
    parser.add_argument("--set_name", required=True, type=str, help="Name of dataset to load (e.g. train).")
    parser.add_argument("--out_dir",  required=True, type=str, help="Directory to write binary files to.")
    return parser.parse_args()

def getFiles(data_dir):
    """
    Return a list of label files and a list of the corresponding feature files.
    """
    label_files = sorted([osp.join(data_dir, f) for f in os.listdir(data_dir) if osp.isfile(osp.join(data_dir, f)) and "label" in f])
    feat_files = sorted([osp.join(data_dir, f) for f in os.listdir(data_dir) if osp.isfile(osp.join(data_dir, f)) and "feat" in f])
    return label_files, feat_files

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

def main():
    args = parseArgs()
    data_dir = args.data_dir + "/" if args.data_dir[-1] != "/" else args.data_dir
    out_dir = args.out_dir + "/" if args.out_dir[-1] != "/" else args.out_dir
    outfile = out_dir + args.set_name 

    label_files, feat_files = getFiles(data_dir)
    labels, feats, all_lens = loadFiles(label_files, feat_files)
    makeBinaries(outfile, labels, feats, all_lens)
    
if __name__ == "__main__":
    main()
