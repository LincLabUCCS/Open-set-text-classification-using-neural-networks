#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import yaml

# Hyperparameters and config

dataset = "20newsgroup"
embedding_dimension = 300 # word2vec

four_classes = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
twenty_classes = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc', 'talk.religion.misc']

training_classes = ['comp.graphics', 'alt.atheism', 'comp.sys.mac.hardware', 'misc.forsale', 'rec.autos']


# load data
print("Loading data...")
if dataset == "20newsgroup":
    datasets = data_helpers.get_datasets_20newsgroup(subset='train', categories=training_classes, remove=()) # TODO: use the remove parameter
    x_text, y_train = data_helpers.load_data_labels_remove_SW(datasets)
else:
    dataset = data_helpers.get_datasets_localdata("./data/20newsgroup", categories=None) # TODO: tweak parameters in the future
    x_text, y_train = data_helpers.load_data_labels(dataset) # text is stored in x_test; # labels are stored in y

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text]) # TODO: should be hardcoded to save time
print("Max document length: {}".format(max_document_length))
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x_train = np.array(list(vocab_processor.fit_transform(x_text)))


# Randomly shuffle data
# np.random.seed(10)
# shuffle_indices = np.random.permutation(np.arange(len(y)))
# x_shuffled = x[shuffle_indices]
# y_shuffled = y[shuffle_indices]

print(x_train.shape)
print(y_train.shape)

# __TRAINING__

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False, gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        vocabulary = vocab_processor.vocabulary_
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocabulary),
            embedding_size=embedding_dimension,
            filter_sizes=[3, 4, 5],
            num_filters=100, # using 100 filters each total 300 filters
            l2_reg_lambda=0.0) #  l2 regularization


    # training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(0.003)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # __SUMMARIES AND CHECKPOINTS__
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # loss and accuracy
    loss_summary = tf.summary.scalar("loss", cnn.loss)
    acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    vocab_processor.save(os.path.join(out_dir, "vocab"))


    sess.run(tf.global_variables_initializer())
    # word embeddings
    print("loading word2vec file...\n")
    initW = data_helpers.load_embedding_vectors_word2vec(vocabulary, "./data/GoogleNews-vectors-negative300.bin", True)
    print("word2vec has been loaded\n")
    sess.run(cnn.W.assign(initW))


    def train_step(x_batch, y_batch):
        """
            A single training step
        """
        feed_dict = {
          cnn.input_x: x_batch,
          cnn.input_y: y_batch,
          cnn.dropout_keep_prob: 0.5 # dropout prob
        }
        _, step, summaries, loss, accuracy = sess.run(
            [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)

    # Generate batches
    batches = data_helpers.batch_iter(
        list(zip(x_train, y_train)), 5, 25) # 25 = batch_size, 25 = no. epochs
    # Training loop. For each batch...
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)

        # if current_step % FLAGS.evaluate_every == 0:
        #     print("\nEvaluation:")
        #     dev_step(x_dev, y_dev, writer=dev_summary_writer)
        #     print("")
        if current_step % 100 == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))
    print("Done.")
