# NCC: Neural Code Comprehension
# https://github.com/spcl/ncc
# Copyright 2018 ETH Zurich
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================
"""inst2vec embedding training"""


from inst2vec import inst2vec_evaluate as i2v_eval
from inst2vec import inst2vec_appflags
from inst2vec import inst2vec_utils as i2v_utils
import numpy as np
import pickle
import os
import subprocess
import datetime
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.client import timeline
from datetime import datetime
import random
import sys
import math
from absl import flags

FLAGS = flags.FLAGS


########################################################################################################################
# Reading, writing and dumping files
########################################################################################################################
def get_data_pair_files(folders, context_width):
    """
    Given a data set composed of several raw folders, return a list of the files containing the binary records
    :param folders: list of sub-folders containing pre-processed LLVM IR code
    :param context_width
    :return:
    """
    assert len(folders) > 1, "Expected combineable dataset"
    data_pairs_strings_filenames = list()
    for folder in folders:
        folder_dataset = folder + '_dataset' + '_cw_' + str(context_width)
        file = os.path.join(folder_dataset, 'data_pairs' + '_cw_' + str(context_width) + '.rec')
        assert os.path.exists(file), 'File ' + file + ' does not exist'
        data_pairs_strings_filenames.append(file)

    # Return
    return data_pairs_strings_filenames


def record_parser(record):
    """
    Read the bytes of a string as a vector of numbers
    :return pair of integers (target-context indices)
    """
    return tf.decode_raw(record, tf.int32)


########################################################################################################################
# Helper functions for training and evaluation
########################################################################################################################
def print_neighbors(op, examples, top_k, reverse_dictionary):
    """
    Print the nearest neighbours of certain statements
    :param op: "nearest-neighbour" tensorflow operation
    :param examples: list of statements indices to evaluate on
    :param top_k: number of nearest neighbours to print
    :param reverse_dictionary: [keys=statement index, values=statement]
    """
    # compute cosine similarity
    sim = op.eval()

    # search for nearest neighbor and print
    for i, ex in enumerate(examples):
        valid_word = reverse_dictionary[ex]  # get dictionary index
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:\n    ' % valid_word
        for k in range(top_k):
            close_word = reverse_dictionary[nearest[k]]
            log_str = '%s %s\n    ' % (log_str, close_word)
        print(log_str)


########################################################################################################################
# Training embeddings
########################################################################################################################
def train_skip_gram(V, data_folder, data_folders, dataset_size, reverse_dictionary,
                    param, valid_examples, log_dir, vocab_metada_file, embeddings_pickle,
                    ckpt_saver_file, ckpt_saver_file_init, ckpt_saver_file_final,
                    restore_variables):
    """
    Train embeddings (Skip-Gram model)
    :param V: vocabulary size
    :param data_folder: string containing the path to the parent directory of raw data sub-folders
    :param data_folders: list of sub-folders containing pre-processed LLVM IR code
    :param dataset_size: number of data pairs in total in the training data set
    :param reverse_dictionary: [keys=statement index, values=statement]
    :param param: parameters of the inst2vec training
    :param valid_examples: statements to be used as validation examples (list of indices)
    :param log_dir: logging directory for Tensorboard output
    :param vocab_metada_file: vocabulary metadata file for Tensorboard
    :param embeddings_pickle: file in which to pickle embeddings
    :param ckpt_saver_file: checkpoint saver file (intermediate states of training)
    :param ckpt_saver_file_init: checkpoint saver file (initial state of training)
    :param ckpt_saver_file_final: checkpoint saver file (final state of training)
    :param restore_variables: boolean: whether to restore variables from a previous training
    :return: embeddings matrix
    """
    ####################################################################################################################
    # Extract parameters from dictionary "param"
    N = param['embedding_size']
    mini_batch_size = param['mini_batch_size']
    num_sampled = param['num_sampled']
    num_epochs = param['num_epochs']
    learning_rate = param['learning_rate']
    l2_reg_scale = param['beta']
    freq_print_loss = param['freq_print_loss']
    step_print_neighbors = param['step_print_neighbors']
    context_width = param['context_width']

    ####################################################################################################################
    # Set up for analogies
    analogies, analogy_types, n_questions_total, n_questions_relevant = i2v_eval.load_analogies(data_folder)
    folder_evaluation = embeddings_pickle.replace('.p', '') + 'eval'
    if not os.path.exists(folder_evaluation):
        os.makedirs(folder_evaluation)
    analogy_evaluation_file = os.path.join(folder_evaluation, "analogy_results")

    config = None
    options = None
    metadata = None
    if FLAGS.profile:
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        metadata = tf.RunMetadata()
    if FLAGS.xla:
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    ####################################################################################################################
    # Read data using Tensorflow's data API
    data_files = get_data_pair_files(data_folders, context_width)
    print('\ttraining with data from files:', data_files)
    with tf.name_scope("Reader") as scope:

        random.shuffle(data_files)
        dataset_raw = tf.data.FixedLengthRecordDataset(filenames=data_files,
                                                       record_bytes=8)  # <TFRecordDataset shapes: (), types: tf.string>
        dataset = dataset_raw.map(record_parser)
        dataset = dataset.shuffle(int(1e5))
        dataset_batched = dataset.apply(tf.contrib.data.batch_and_drop_remainder(mini_batch_size))
        dataset_batched = dataset_batched.prefetch(int(100000000))
        iterator = dataset_batched.make_initializable_iterator()
        saveable_iterator = tf.contrib.data.make_saveable_from_iterator(iterator)
        next_batch = iterator.get_next()  # Tensor("Shape:0", shape=(2,), dtype=int32)

    ####################################################################################################################
    # Tensorflow computational graph
    # Placeholders for inputs
    with tf.name_scope("Input_Data") as scope:
        train_inputs = next_batch[:, 0]
        train_labels = tf.reshape(next_batch[:, 1], shape=[mini_batch_size, 1], name="training_labels")

    # (input) Embedding matrix
    with tf.name_scope("Input_Layer") as scope:
        W_in = tf.Variable(tf.random_uniform([V, N], -1.0, 1.0), name="input-embeddings")

        # Look up the vector representing each source word in the batch (fetches rows of the embedding matrix)
        h = tf.nn.embedding_lookup(W_in, train_inputs, name="input_embedding_vectors")

    # Normalized embedding matrix
    with tf.name_scope("Embeddings_Normalized") as scope:
        normalized_embeddings = tf.nn.l2_normalize(W_in, name="embeddings_normalized")

    # (output) Embedding matrix ("output weights")
    with tf.name_scope("Output_Layer") as scope:
        if FLAGS.softmax:
            W_out = tf.Variable(tf.truncated_normal([N, V], stddev=1.0 / math.sqrt(N)), name="output_embeddings")
        else:
            W_out = tf.Variable(tf.truncated_normal([V, N], stddev=1.0 / math.sqrt(N)), name="output_embeddings")

        # Biases between hidden layer and output layer
        b_out = tf.Variable(tf.zeros([V]), name="nce_bias")

    # Optimization
    with tf.name_scope("Optimization_Block") as scope:
        # Loss function
        if FLAGS.softmax:
            logits = tf.layers.dense(inputs=h, units=V)
            onehot = tf.one_hot(train_labels, V)
            loss_tensor = tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot, logits=logits)
        else:
            loss_tensor = tf.nn.nce_loss(weights=W_out,
                                         biases=b_out,
                                         labels=train_labels,
                                         inputs=h,
                                         num_sampled=num_sampled,
                                         num_classes=V)
        train_loss = tf.reduce_mean(loss_tensor, name="nce_loss")

        # Regularization (optional)
        if l2_reg_scale > 0:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W_in)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, W_out)
            regularizer = tf.contrib.layers.l2_regularizer(l2_reg_scale)
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
            loss = train_loss + reg_term
        else:
            loss = train_loss

        # Optimizer
        if FLAGS.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        elif FLAGS.optimizer == 'nadam':
            optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=learning_rate).minimize(loss)
        elif FLAGS.optimizer == 'momentum':
            global_train_step = tf.Variable(0, trainable=False, dtype=tf.int32, name="global_step")
            # Passing global_step to minimize() will increment it at each step.
            optimizer = (
                tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_train_step)
            )
        else:
            raise ValueError('Unrecognized optimizer ' + FLAGS.optimizer)

    if FLAGS.optimizer != 'momentum':
        global_train_step = tf.Variable(0, trainable=False, dtype=tf.int32, name="global_step")

    ####################################################################################################################
    # Validation block
    with tf.name_scope("Validation_Block") as scope:
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32, name="validation_data_size")
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        cosine_similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    ####################################################################################################################
    # Summaries
    with tf.name_scope("Summaries") as scope:
        tf.summary.histogram("input_embeddings", W_in)
        tf.summary.histogram("input_embeddings_normalized", normalized_embeddings)
        tf.summary.histogram("output_embeddings", W_out)
        tf.summary.scalar("nce_loss", loss)

        analogy_score_tensor = tf.Variable(0, trainable=False, dtype=tf.int32, name="analogy_score")
        tf.summary.scalar("analogy_score", analogy_score_tensor)

    ####################################################################################################################
    # Misc.
    restore_completed = False
    init = tf.global_variables_initializer()        # variables initializer
    summary_op = tf.summary.merge_all()             # merge summaries into one operation

    ####################################################################################################################
    # Training
    with tf.Session(config=config) as sess:

        # Add TensorBoard components
        writer = tf.summary.FileWriter(log_dir)  # create summary writer
        writer.add_graph(sess.graph)
        gvars = [gvar for gvar in tf.global_variables() if 'analogy_score' not in gvar.name]
        saver = tf.train.Saver(gvars, max_to_keep=5)  # create checkpoint saver
        config = projector.ProjectorConfig()  # create projector config
        embedding = config.embeddings.add()  # add embeddings visualizer
        embedding.tensor_name = W_in.name
        embedding.metadata_path = vocab_metada_file  # link metadata
        projector.visualize_embeddings(writer, config)  # add writer and config to projector

        # Set up variables
        if restore_variables:   # restore variables from disk
            restore_file = tf.train.latest_checkpoint(log_dir)
            assert restore_file is not None, "No restore file found in folder " + log_dir
            assert os.path.exists(restore_file + ".index"), \
                "Trying to restore Tensorflow session from non-existing file: " + restore_file + ".index"
            init.run()
            saver.restore(sess, restore_file)
            print("\tVariables restored from file", ckpt_saver_file, "in TensorFlow ")

        else:  # save the computational graph to file and initialize variables

            graph_saver = tf.train.Saver(allow_empty=True)
            init.run()
            graph_saver.save(sess, ckpt_saver_file_init, global_step=0, write_meta_graph=True)
            tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable_iterator)
            print("\tVariables initialized in TensorFlow")

        # Compute the necessary number of steps for this epoch as well as how often to print the avg loss
        num_steps = int(math.ceil(dataset_size / mini_batch_size))
        step_print_loss = int(math.ceil(num_steps / freq_print_loss))
        print('\tPrinting loss every ', step_print_loss, 'steps, i.e.', freq_print_loss, 'times per epoch')

        ################################################################################################################
        # Epoch loop
        epoch = 0
        global_step = 0
        while epoch < int(num_epochs):
            print('\n\tStarting epoch ', epoch)
            sess.run(iterator.initializer)      # initialize iterator

            # If restoring a previous training session, set the right training epoch
            if restore_variables and not restore_completed:
                epoch = int(math.floor(global_train_step.eval() / (dataset_size / mini_batch_size)))
                global_step = global_train_step.eval()
                print('Starting from epoch', epoch)

            ############################################################################################################
            # Loop over steps (mini batches) inside of epoch
            step = 0
            avg_loss = 0
            while True:

                try:

                    # Print average loss every x steps
                    if step_print_loss > 0 and step % int(step_print_loss) == 0:    # update step with logging

                        # If restoring a previous training session, set the right training epoch
                        if restore_variables and not restore_completed:
                            restore_completed = True

                        # Write global step
                        if FLAGS.optimizer != 'momentum':
                            global_train_step.assign(global_step).eval()

                        # Perform an update
                        # print('\tStarting local step {:>6}'.format(step))  # un-comment for debugging
                        [_, loss_val, train_loss_val, global_step] = sess.run(
                            [optimizer, loss, train_loss, global_train_step], options=options,
                            run_metadata=metadata)
                        assert not np.isnan(loss_val), "Loss at step " + str(step) + " is nan"
                        assert not np.isinf(loss_val), "Loss at step " + str(step) + " is inf"
                        avg_loss += loss_val

                        if step > 0:
                            avg_loss /= step_print_loss

                        analogy_score = i2v_eval.evaluate_analogies(W_in.eval(), reverse_dictionary, analogies,
                                                                    analogy_types, analogy_evaluation_file,
                                                                    session=sess, print=i2v_eval.nop)
                        total_analogy_score = sum([a[0] for a in analogy_score])
                        analogy_score_tensor.assign(total_analogy_score).eval()  # for tf.summary

                        [summary, W_in_val] = sess.run([summary_op, W_in])

                        if FLAGS.savebest is not None:
                            filelist = [f for f in os.listdir(FLAGS.savebest)]
                            scorelist = [int(s.split('-')[1]) for s in filelist]
                            if len(scorelist) == 0 or total_analogy_score > sorted(scorelist)[-1]:
                                i2v_utils.safe_pickle(W_in_val, FLAGS.savebest + '/' + 'score-' +
                                                      str(total_analogy_score) + '-w.p')

                        # Display average loss
                        print('{} Avg. loss at epoch {:>6,d}, step {:>12,d} of {:>12,d}, global step {:>15} : {:>12.3f}, analogies: {})'.format(
                            str(datetime.now()), epoch, step, num_steps, global_step, avg_loss, str(analogy_score)))
                        avg_loss = 0

                        # Pickle intermediate embeddings
                        i2v_utils.safe_pickle(W_in_val, embeddings_pickle)

                        # Write to TensorBoard
                        saver.save(sess, ckpt_saver_file, global_step=global_step, write_meta_graph=False)
                        writer.add_summary(summary, global_step=global_step)

                        if FLAGS.profile:
                            fetched_timeline = timeline.Timeline(metadata.step_stats)
                            chrome_trace = fetched_timeline.generate_chrome_trace_format()
                            with open('timeline_step_%d.json' % step, 'w') as f:
                                f.write(chrome_trace)

                        if step > 0 and FLAGS.extreme:
                            sys.exit(22)

                    else:   # ordinary update step
                        [_, loss_val] = sess.run([optimizer, loss])
                        avg_loss += loss_val

                    # Compute and print nearest neighbors every x steps
                    if step_print_neighbors > 0 and step % int(step_print_neighbors) == 0:
                        print_neighbors(op=cosine_similarity, examples=valid_examples, top_k=6,
                                        reverse_dictionary=reverse_dictionary)

                    # Update loop index (steps in epoch)
                    step += 1
                    global_step += 1

                except tf.errors.OutOfRangeError:

                    # We reached the end of the epoch
                    print('\n\t Writing embeddings to file ', embeddings_pickle)
                    i2v_utils.safe_pickle([W_in.eval()], embeddings_pickle)                   # WEIRD!
                    epoch += 1      # update loop index (epochs)
                    break           # from this inner loop

        ################################################################################################################
        # End of training:
        # Print the nearest neighbors at the end of the run
        if step_print_neighbors == -1:
            print_neighbors(op=cosine_similarity, examples=valid_examples, top_k=6,
                            reverse_dictionary=reverse_dictionary)

        # Save state of training and close the TensorBoard summary writer
        save_path = saver.save(sess, ckpt_saver_file_final, global_step)
        writer.add_summary(summary, global_step)
        writer.close()

        return W_in.eval()


########################################################################################################################
# Main function for embedding training workflow
########################################################################################################################
def train_embeddings(data_folder, data_folders):
    """
    Main function for embedding training workflow
    :param data_folder: string containing the path to the parent directory of raw data sub-folders
    :param data_folders: list of sub-folders containing pre-processed LLVM IR code
    :return embedding matrix

    Folders produced:
        data_folder/FLAGS.embeddings_folder/emb_cw_X_embeddings
        data_folder/FLAGS.embeddings_folder/emb_cw_X_train
    """

    # Get flag values
    restore_tf_variables_from_ckpt = FLAGS.restore
    context_width = FLAGS.context_width
    outfolder = FLAGS.embeddings_folder
    param = {k: FLAGS[k].value for k in FLAGS}

    # Set file signature
    file_signature = i2v_utils.set_file_signature(param, data_folder)
    
    # Print model parameters
    out_ = '\n--- Data files: '
    print(out_)
    out = out_ + '\n'
    num_data_pairs = 0
    data_pair_files = get_data_pair_files(data_folders, context_width)
    for data_pair_file in data_pair_files:
        filesize_bytes = os.path.getsize(data_pair_file)  # num pairs = filesize_bytes / 2 (pairs) / 4 (32-bit integers)
        file_pairs = int(filesize_bytes / 8)
        num_data_pairs += file_pairs
        out_ = '\t{:<60}: {:>12,d} pairs'.format(data_pair_file, file_pairs)
        print(out_)
        out += out_ + '\n'
    
    out_ = '\t{:<60}: {:>12,d} pairs'.format('total', num_data_pairs)
    print(out_)
    out += out_ + '\n'
    
    # Get dictionary and vocabulary
    print('\n\tGetting dictionary ...')
    folder_vocabulary = os.path.join(data_folder, 'vocabulary')
    dictionary_pickle = os.path.join(folder_vocabulary, 'dic_pickle')
    with open(dictionary_pickle, 'rb') as f:
        dictionary = pickle.load(f)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    del dictionary
    vocabulary_size = len(reverse_dictionary.keys())

    # Print Skip-Gram model parameters
    out_ = '\n--- Skip Gram model parameters'
    print(out_)
    out += out_ + '\n'
    out_ = '\tData folder             : {:<}'.format(data_folder)
    print(out_)
    out += out_ + '\n'
    out_ = '\tNumber of data pairs    : {:>15,d}'.format(num_data_pairs)
    print(out_)
    out += out_ + '\n'
    out_ = '\tVocabulary size         : {:>15,d}'.format(vocabulary_size)
    print(out_)
    out += out_ + '\n'
    out_ = '\tEmbedding size          : {:>15,d}'.format(param['embedding_size'])
    print(out_)
    out += out_ + '\n'
    out_ = '\tContext width           : {:>15,d}'.format(param['context_width'])
    print(out_)
    out += out_ + '\n'
    out_ = '\tMini-batch size         : {:>15,d}'.format(param['mini_batch_size'])
    print(out_)
    out += out_ + '\n'
    out_ = '\tNegative samples in NCE : {:>15,d}'.format(param['num_sampled'])
    print(out_)
    out += out_ + '\n'
    out_ = '\tL2 regularization scale : {:>15,e}'.format(param['beta'])
    print(out_)
    out += out_ + '\n'
    out_ = '\tNumber of epochs        : {:>15,d}'.format(param['num_epochs'])
    print(out_)
    out += out_ + '\n'
    out_ = '\tRestoring a prev. train : {}'.format(restore_tf_variables_from_ckpt)
    print(out_)
    out += out_ + '\n'
    
    # Print training information to file
    log_dir_ = os.path.join(outfolder, 'emb_cw_' + str(context_width) + '_train/')
    log_dir = os.path.join(log_dir_, file_signature[1:])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    train_info_file = os.path.join(log_dir, 'training_info.txt')
    with open(train_info_file, 'w') as f:
        f.write(out)
    
    # Validation set used to sample nearest neighbors
    # Limit to the words that have a low numeric ID,
    # which by construction are also the most frequent.
    valid_size = 30    # Random set of words to evaluate similarity on.
    valid_window = 50  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    
    # Copy metadata file into TensorBoard folder
    vocab_metada_file_ = os.path.join(folder_vocabulary, 'vocabulary_metadata_for_tboard')
    v_metadata_file_name = 'vocab_metada_' + file_signature
    vocab_metada_file = os.path.join(log_dir, v_metadata_file_name)
    ckpt_saver_file = os.path.join(log_dir, "inst2vec.ckpt")
    ckpt_saver_file_init = os.path.join(log_dir, "inst2vec-init.ckpt")
    ckpt_saver_file_final = os.path.join(log_dir, "inst2vec-final.ckpt")
    os.makedirs(os.path.dirname(vocab_metada_file), exist_ok=True)
    subprocess.call('cp ' + vocab_metada_file_ + ' ' + vocab_metada_file, shell=True)
    
    # Train the embeddings (Skip-Gram model)
    print('\n--- Setup completed, starting to train the embeddings')
    folder_embeddings = os.path.join(outfolder, 'emb_cw_' + str(context_width) + '_embeddings')
    if not os.path.exists(folder_embeddings):
        os.makedirs(folder_embeddings)
    embeddings_pickle = os.path.join(folder_embeddings, "emb_" + file_signature + ".p")
    embeddings = train_skip_gram(vocabulary_size, data_folder, data_folders, num_data_pairs, reverse_dictionary,
                                 param, valid_examples, log_dir, v_metadata_file_name, embeddings_pickle,
                                 ckpt_saver_file, ckpt_saver_file_init, ckpt_saver_file_final,
                                 restore_tf_variables_from_ckpt)

    # Save the embeddings and dictionaries in an external file to be reused later
    print('\n\tWriting embeddings to file', embeddings_pickle)
    i2v_utils.safe_pickle(embeddings, embeddings_pickle)

    # Write the embeddings to CSV file
    embeddings_csv = os.path.join(folder_embeddings, "emb_" + file_signature + ".csv")
    print('\t Writing embeddings to file ', embeddings_csv)
    np.savetxt(embeddings_csv, embeddings, delimiter=',',
               header='Embeddings matrix, rows correspond to the embedding vector of statements')

    return embeddings, embeddings_pickle
