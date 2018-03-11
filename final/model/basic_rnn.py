"""

Adapted from code written by Brian Hutchinson.

features == number of documents x maximum length x number of features
labels   == number of documents x maximum length (scores for each sentence) 
lens     == number of sentences contained in each document

"""

import argparse
import numpy as np

def parseArgs():
    parser = argparse.ArgumentParser()
    
    # data
    parser.add_argument("--train_dir", required=True, help="Directory containing train .npy binary files.")
    parser.add_argument("--dev_dir",   required=True, help="Directory containing dev .npy binary files.")
    parser.add_argument("--test_dir"   required=True, help="Directory containing test .npy binary files.")
    # hyperparameter values
    parser.add_argument("-lr", type=float, default=0.001, help="The learning rate.")
    parser.add_argument("-L",  type=int,   default=100,   help="The hidden layer dimension.")
    parser.add_argument("-ep", type=int,   default=20,    help="The number of epochs to train for.")
    parser.add_argument("-mb", type=int,   default=64,    help="The minibatch size.")

    return parser.parse_args()

def buildGraph(args, vocab_size):
    """
    """
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(args.L)

    # placeholders
    h0     = tf.placeholder(dtype=tf.float32, shape=(None, args.L), name="h0")
    lens   = tf.placeholder(dtype=tf.int64, shape=(None,), name="lens")
    x = tf.placeholder(dtype=tf.int64, shape=(None, None, None), name="x")
    y_true = tf.placeholder(dtype=tf.int64, shape=(None, None), name="y_true")
    
    # outputs (mb, num_steps, args.L) - per time outputs for each sentence
    # state   (mb, args.L)            - last hidden state per sentence
    outputs, state = tf.nn.dynamic_rnn(cell=rnn_cell, 
                                       inputs=x,
                                       sequence_lengths=lens,
                                       initial_state=h0,
                                       dtype=tf.float32)
    
    # hidden to output layer weights
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
    mask = tf.sequence_mask(lens, maxlen=tf.shape(x_ints)[1], dtype=tf.float32, name="mask")

    # compute loss over all num_steps and all sequences in the minibatch
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
    unnormalized_objective = tf.reduce_sum(losses * mask)
    denominator = tf.reduce_sum(mask)

    # minimize average loss
    normalized_objective = unnormalized_objective / denominator
    train_step = tf.train.AdamOptimizer(args.lr).minimize(normalized_objective, name="train_step")

    # perplexity and accuracy
    ppl = tf.exp(normalized_objective, name="perplexity")
    acc = tf.reduce_mean(tf.cast(tf.equal(y_true, tf.argmax(logits, axis=2)), tf.float32), name="accuracy")
    

    init = tf.global_variables_initializer()

    return init

def loadData(directory):
    directory = args.directory + "/" if args.directory[-1] != "/" else args.directory
    labels = np.load(glob.glob(directory + "*_labels.npy"))
    features = np.load(glob.glob(directory + "*_feats.npy"))
    all_lens = list(np.load(glob.glob(directory + "*_lens.npy")))
    abs_lens = list(np.load(glob.glob(directory + "*_abs.npy"))
    return labels, features, all_lens, abs_lens    
    
def evaluateDev(args, featurs, labels, lens, sess, name):
    hidden_states = np.zeros(shape=(features.shape[0], args.L), dtype=np.float32)
    
    ppl, acc = sess.run(fetches=["perplexity:0","accuracy:0"],
                        feed_dict={"x_ints:0":sentences[:,:,:]
                                   "y_true:0":labels[:,:]
                                   "h0:0":hidden_states,
                                   "lens:0":(lens)})
    print(name + " ppl = %.5f acc = %.5f" % (ppl, acc))

#def generateSummary(logits, n)

def evaluateTest(args, featurs, labels, lens, abs_lens, sess, name):
    hidden_states = np.zeros(shape=(features.shape[0], args.L), dtype=np.float32)
    for row in range(features.shape[0]):
        ppl, acc, logits = sess.run(fetches=["perplexity:0","accuracy:0", "logits:0"],
                            feed_dict={"x_ints:0":sentences[row,:,:]
                                       "y_true:0":labels[row,:]
                                       "h0:0":hidden_states[row,:],
                                       "lens:0":(lens[row])})
        print(name + " ppl = %.5f acc = %.5f" % (ppl, acc))
        generateSummary(logits, abs_lens[row])

def main():
    """
    Load datasets, train on minibatches for each epoch, and report scores on dev set.
    """
    args = parseArgs()

    # load data
    train_labels, train_feats, train_lens, _ = loadData(args.train_dir)
    dev_labels, dev_feats, dev_lens, _ = loadData(args.dev_dir)
    test_labels, test_feats, test_lens, test_abs = loadData(args.test_dir)
    
    num_steps = train_feats.shape[1]    # the length of our unrolled graph 
    N = train_feats.shape[0]            # number of training sequences
    #NOTE V = ?                             # vocab size

    init = buildGraph(args, V)

    # TRAIN
    with tf.Session() as sess:
        sess.run(fetches=[init])

        num_batches = int(np.ceil(N/args.mb))
        print("Total minibatches per epoch: %d" % num_batches)

        for epoch in range(args.epochs):
            print("epoch %d update... " % (epoch), end='', flush=True)
            
            for mb_row in range(num_batches):
                row_start = mb_row*args.mb
                row_end   = np.min([(mb_row+1)*args.mb, N]) # last minibatch may be partial

                mb_x = train_feats[row_start:row_end, :, :]
                mb_y = train_labels[row_start:row_end, :]
                mb_lens = train_lens[row_start:row_end]
                mb_h0 = np.zeros(shape=((), args.L), dtype=np.float32)

                sess.run(fetches=["train_step"],
                         feed_dict={"x_ints:0":mb_x,
                                    "y_trues:0":mb_y,
                                    "h0:0":mb_h0,
                                    "lens:0":mb_lens})

                if(mb_row % 100 = 0):
                    print("%d... " % (mb_row),end='',flush=True)
            print("training loop finished.")
    
            evaluate(args, dev_feats, dev_labels, dev_lens, "dev")
        evaluate(args, test_feats, test_labels, test_lens, abs_lens, "test")          

if __name__ == "__main__":
    main()
