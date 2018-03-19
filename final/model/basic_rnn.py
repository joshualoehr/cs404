"""
How to run (update with custom epoch number and hidden dimension L):
python basic_rnn.py --data model_data --out system -ep 10 -L 10

features == number of documents x maximum length x number of features
labels   == number of documents x maximum length (scores for each sentence) 
lens     == number of sentences contained in each document

Adapted from code written by Brian Hutchinson.
"""

import argparse
import numpy as np
import tensorflow as tf
import os

def parseArgs():
    parser = argparse.ArgumentParser()
    
    # data
    parser.add_argument("--data", required=True, type=str,
        help="Directory containing train, dev, and test subdirectories.")
    parser.add_argument("--out",   required=True, type=str,
        help="Directory to store produced summaries to.")
    
    # hyperparameter values
    parser.add_argument("-lr", type=float, default=0.001, help="The learning rate.")
    parser.add_argument("-L",  type=int,   default=100,   help="The hidden layer dimension.")
    parser.add_argument("-ep", type=int,   default=20,    help="The number of epochs to train for.")

    return parser.parse_args()

def loadData(directory, set_name):
    dir_file = directory + set_name + "/" + set_name
    labels = np.load(dir_file + "_labels.npy")
    features = np.load(dir_file + "_feats.npy")
    all_lens = list(np.load(dir_file + "_lens.npy"))   # the number of sentences in each document
    
    abstracts = list(np.load(dir_file + "_abs.npy")) if set_name == "test" else ""
    documents = np.load(dir_file + "_docs.npy") if set_name == "test" else ""
    doc_names = np.load(dir_file + "_names.npy") if set_name == "test" else ""

    return labels, features, all_lens, abstracts, documents, doc_names
 
def buildGraph(args, vocab_size, num_feats):
    """
    """
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(args.L)

    # placeholders
    h0     = tf.placeholder(dtype=tf.float32, shape=(None, args.L), name="h0")
    lens   = tf.placeholder(dtype=tf.int64, shape=(None,), name="lens")
    x = tf.placeholder(dtype=tf.float32, shape=(None, None, num_feats), name="x")
    y_true = tf.placeholder(dtype=tf.int64, shape=(None, None), name="y_true")
    
    # outputs (mb, num_steps, args.L) - per time outputs for each sentence
    # state   (mb, args.L)            - last hidden state per sentence
    outputs, state = tf.nn.dynamic_rnn(cell=rnn_cell, 
                                       inputs=x,
                                       sequence_length=lens,
                                       initial_state=h0,
                                       dtype=tf.float32)
    # hidden to output layer weights (hidden layer dimension by vocabulary size)
    W = tf.get_variable(dtype=tf.float32,
                                shape=(args.L, vocab_size),
                                initializer=tf.glorot_uniform_initializer(),
                                name="W")
    b = tf.get_variable(dtype=tf.float32,
                                shape=(vocab_size),
                                initializer=tf.glorot_uniform_initializer(),
                                name="b")

    # map the args.L dimension of outputs with the args.L dimension of W 
    logits = tf.add(tf.tensordot(outputs, W, [[2],[0]]), b, name="logits")

    # mask containing only valid y_true
    mask = tf.sequence_mask(lens, maxlen=tf.shape(x)[1], dtype=tf.float32, name="mask")

    # compute loss over all num_steps and all sequences in the minibatch
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
    unnormalized_objective = tf.reduce_sum(losses * mask, name="perplexity")
    denominator = tf.reduce_sum(mask, name="denom")

    # minimize average loss
    normalized_objective = unnormalized_objective / denominator
    train_step = tf.train.AdamOptimizer(args.lr).minimize(normalized_objective, name="train_step")

    # perplexity and accuracy
    #ppl = tf.exp(normalized_objective, name="perplexity")
    max_logits = tf.argmax(logits, axis=2, name="ls")
    acc = tf.reduce_sum(tf.cast(tf.equal(y_true, max_logits), tf.float32), name="accuracy")

    init = tf.global_variables_initializer()

    return init

def evaluateDev(args, features, labels, lens, sess):
    ppl = 0.0
    acc = 0.0
    denominator = 0.0
    hidden_states = np.zeros(shape=(features.shape[0], args.L), dtype=np.float32)
    for row in range(features.shape[0]):    # row corresponds to document
        x = np.reshape(features[row,:,:], (1, features.shape[1], features.shape[2]))
        y_true = np.reshape(labels[row,:], (1, labels.shape[1]))
        h = np.reshape(hidden_states[row,:], (1, hidden_states.shape[1]))
        p, a, denom = sess.run(fetches=["perplexity:0","accuracy:0", "denom:0"],
                            feed_dict={"x:0":x,
                                       "y_true:0":y_true,
                                       "h0:0":h,
                                       "lens:0":([lens[row]])})
        ppl += p
        acc += a
        denominator += denom
    print("Dev ppl = %.5f acc = %.5f" % (ppl/denominator, acc/denominator))
 
def generateSummary(logits, abstract_len, document, doc_name, out_dir):
    out_dir = out_dir + "/" if out_dir[-1] != "/" else out_dir
    
    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)

    out_file = out_dir + doc_name + ".txt"
    logits = logits[0][:len(document)]
    
    indexed_logits = list(enumerate(logits))
    indexed_logits.sort(key=lambda logit: logit[1])
    sentence_indices = sorted([index for index, score in indexed_logits[:abstract_len]])
    print(sentence_indices)
    with open(out_file, 'w') as f:
        f.write("\n".join([document[i] for i in sentence_indices]))

def evaluateTest(args, features, labels, lens, abstracts, documents, document_names, sess):
    hidden_states = np.zeros(shape=(features.shape[0], args.L), dtype=np.float32)
    for row in range(features.shape[0]):    # row corresponds to document
        x = np.reshape(features[row,:,:], (1, features.shape[1], features.shape[2]))
        y_true = np.reshape(labels[row,:], (1, labels.shape[1]))
        h = np.reshape(hidden_states[row,:], (1, hidden_states.shape[1]))
        p, a, denom, logits = sess.run(fetches=["perplexity:0","accuracy:0", "denom:0", "ls:0"],
                            feed_dict={"x:0":x,
                                       "y_true:0":y_true,
                                       "h0:0":h,
                                       "lens:0":([lens[row]])})
        print("%s: ppl = %.5f acc = %.5f" % (document_names[row].split("_")[0], p/denom, a/denom), end=" ")
        generateSummary(logits, abstracts[row], documents[row], document_names[row], args.out)

def main():
    """
    Load datasets, train on minibatches for each epoch, and report scores on dev set.
    """
    args = parseArgs()
    
    data = args.data + "/" if args.data[-1] != "/" else args.data

    # load data
    train_labels, train_feats, train_lens, _, _, _ = loadData(data, "train")
    dev_labels, dev_feats, dev_lens, _, _, _ = loadData(data, "dev")
    test_labels, test_feats, test_lens, test_abs, test_docs, test_names = loadData(data, "test")

    N = train_feats.shape[0]            # number of documents (number of training sequences)
    num_steps = train_feats.shape[1]    # number of sentences, vocabulary size (length of the unrolled graph)
    num_feats = train_feats.shape[2]    # number of features
    
    init = buildGraph(args, num_steps, num_feats)
 
    # TRAIN
    with tf.Session() as sess:
        sess.run(fetches=[init])

        print("Total minibatches per epoch: %d" % N)

        for epoch in range(args.ep):
            print("epoch %d update... " % (epoch), end='', flush=True)
            for mb_row in range(N):
                mb_x = np.reshape(train_feats[mb_row,:,:], (1, num_steps, num_feats))
                mb_y = np.reshape(train_labels[mb_row,:], (1, num_steps))
                mb_lens = [train_lens[mb_row]]
                mb_h0 = np.zeros(shape=(1, args.L), dtype=np.float32)

                sess.run(fetches=["train_step"],
                         feed_dict={"x:0":mb_x,
                                    "y_true:0":mb_y,
                                    "h0:0":mb_h0,
                                    "lens:0":(mb_lens)})

            print("training loop finished.", end=" ")
    
            if(epoch % 1 == 0):
                evaluateDev(args, dev_feats, dev_labels, dev_lens, sess)
            else:
                print()

        print()
        evaluateTest(args, test_feats, test_labels, test_lens, test_abs, test_docs, test_names, sess)          

if __name__ == "__main__":
    main()
