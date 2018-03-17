"""
Load and save label files, feature files, document length values for each document as 
.npy array files. Additionally store abstract length values, the raw document files, and
the document directory names if the -save_abs argument is provided.

Output: 
* features: load each article.feats and save as <setname>_feats.npy
* labels: load each article.labels and save as <setname>_labels.npy
* lengths: with the information from each label file, store the length of each document, 
    and save as <setname>_lens.npy

If -save_abs:
* abstracts: load each abstract.sentences, store the length of each abstract, 
    and save as <setname>_abs.npy
* documents: load each article.sentences, store a list of the sentences in each document,  
    and save as <setname>_docs.npy
* names: save the unique document folder names as <setname>_names.npy    

python load_data.py --data "data/*_train/*" --set train --out npy_data/train --len 5 
python load_data.py --data "data/*_test/*" --set test --out npy_data/test/ --len 5 -save_abs

"""

import argparse
import glob
import os
import numpy as np

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=glob.glob, 
        help="List of paths to all files for a given set.")
    parser.add_argument("--set", required=True, type=str,
        help="Name of dataset to load [train, dev, or test].")
    parser.add_argument("--out",  required=True, type=str, 
        help="Directory to write binary files to.")
    parser.add_argument("--len", required=True, type=int,
        help="Maximum sentence length for the given set.")
    parser.add_argument("-save_abs", action='store_true',
        help="Store the abstracts, documents, and names.")
    return parser.parse_args()

def getFiles(data, file_name):
    """
    Extract files with the provided file_name from the data directory and return as a list.
    """
    return [f for f in data if file_name in f]

def loadAbstracts(sorted_data):
    """
    Store the number of sentences (length) of each abstract file as an element of a list.
    Store the extracted sentences from each document as an element of a list.
    Return the abstract lengths, document sentences, and document names.
    """
    all_abs = []
    all_docs = []
    
    abs_files = getFiles(sorted_data, "abstract.sentences")
    doc_files = getFiles(sorted_data, "article.sentences")
    document_names = [doc_name.split("/")[1] for doc_name in doc_files]
    
    for abstract in abs_files:
        with open(abstract, 'r') as f:
            all_abs.append(len(f.readlines()))
    for document in doc_files:
        with open(document, 'r') as f:
            all_docs += [[sentence.strip() for sentence in f]]
    return all_abs, all_docs, document_names

def makeAbstractBinary(outfile, abstracts, documents, names):
    """
    Save the abstract, document, and name lists as .npy binary files.
    """
    abs_out = outfile + "_abs.npy"
    doc_out = outfile + "_docs.npy"
    nam_out = outfile + "_names.npy"
    np.save(abs_out, abstracts)
    np.save(doc_out, documents)
    np.save(nam_out, names)

def loadFeatures(sorted_data, max_feat_len):
    """
    For each line in each feature file, pad line to max length provided and append to a list. 
    Return a list of lists (num documents x num sentences x max number of features)
    """
    feat_files = getFiles(sorted_data, "article.feats")
    all_feats = []  # all documents
    for f in feat_files:
        with open(f, 'r') as feats:
            all_lines = []  # lines for a given document
            for line in feats:
                line = line.strip().split()
                curr_len = len(line)
                line += ["0"]*(max_feat_len-curr_len)
                all_lines.append(line)
        all_feats.append(np.array(all_lines))
    return all_feats

def loadLabels(sorted_data):
    """
    For each label file, store as a numpy array and append the number of sentences to a list.
    Return a list of numpy arrays (num documents x variable num sentences)
    Return a list containing the number of sentences in each document (num documents)
    """
    label_files = getFiles(sorted_data, "article.labels")
    all_labels, all_lens= [], []
    max_len = 0
    for label in label_files:
        curr_l = np.loadtxt(label)
        all_labels.append(curr_l)
        all_lens.append(curr_l.shape[0])
    return all_labels, all_lens

def padFiles(all_feats, all_labels, max_sentence_len):
    """
    Given a maximum sentence length, pad label and feature files with 0s.
    Return a label and feature array. 
    labels = num documents x max num sentences
    feats = num documents x max num sentences x max num features
    """
    for i in range(len(all_labels)):
        pad = max_sentence_len - len(all_labels[i])
        all_feats[i] = np.concatenate((all_feats[i], np.zeros((pad, all_feats[i].shape[1]))))
        all_labels[i] = np.concatenate((all_labels[i], np.zeros((pad))))
    
    feats = np.stack(all_feats)
    labels = np.stack(all_labels)
    return feats, labels

def makeBinaries(outfile, feats, labels, all_lens):
    """
    Save the label, feature, and document length arrays provided as .npy binary files.
    """
    feat_out = outfile + "_feats.npy"    
    label_out = outfile + "_labels.npy"
    len_out = outfile + "_lens.npy"
    np.save(feat_out, feats)
    np.save(label_out, labels)
    np.save(len_out, all_lens)    

def main():
    args = parseArgs()
    out_dir = args.out + "/" + args.set if args.out[-1] != "/" else args.out + args.set
    outfile = out_dir + "/" + args.set  

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    sorted_data = sorted(args.data)

    if args.save_abs:
        abstracts, documents, names = loadAbstracts(sorted_data)
        makeAbstractBinary(outfile, abstracts, documents, names)
    
    all_feats = loadFeatures(sorted_data, args.len)
    all_labels, all_lens = loadLabels(sorted_data)
    feats, labels = padFiles(all_feats, all_labels, max(all_lens))
    makeBinaries(outfile, feats, labels, all_lens)
    
if __name__ == "__main__":
    main()
