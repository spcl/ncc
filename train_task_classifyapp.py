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
"""Training workflow for app classification"""
from labm8 import fs
import task_utils
import rgx_utils as rgx
import pickle
from sklearn.utils import resample
import os
import numpy as np
import tensorflow as tf
import math
import struct
from keras import utils
from keras.callbacks import Callback
from absl import app
from absl import flags

# Parameters of classifyapp
flags.DEFINE_string('input_data', 'task/classifyapp', 'Path to input data')
flags.DEFINE_string('out', 'task/classifyapp', 'Path to folder in which to write saved Keras models and predictions')
flags.DEFINE_integer('num_epochs', 50, 'number of training epochs')
flags.DEFINE_integer('batch_size', 64, 'training batch size')
flags.DEFINE_integer('dense_layer', 32, 'dense layer size')
flags.DEFINE_integer('train_samples', 1500, 'Number of training samples per class')
flags.DEFINE_integer('vsamples', 0, 'Sampling on validation set')
flags.DEFINE_integer('save_every', 100, 'Save checkpoint every N batches')
flags.DEFINE_integer('ring_size', 5, 'Checkpoint ring buffer length')
flags.DEFINE_bool('print_summary', False, 'Print summary of Keras model')
flags.DEFINE_integer('buckets', 0, 'Number of buckets to use for training (or zero for no buckets)')
flags.DEFINE_string('bucketstr', None, 'Bucket adaptive histogram method (see numpy.histogram)')

FLAGS = flags.FLAGS


########################################################################################################################
# Utils
########################################################################################################################
def get_onehot(y, num_classes):
    """
    y is a vector of numbers (1, number of classes)
    """
    hot = np.zeros((len(y), num_classes), dtype=np.int32)
    for i, c in enumerate(y):
        # i: data sample index
        # c: class number in range [1, num_classes]
        hot[i][int(c) - 1] = 1

    return hot


def encode_srcs(input_files, dataset_name, unk_index):
    """
    encode and pad source code for learning
    data_folder: folder from which to read input files
    input_files: list of strings of file names
    """

    # Get list of source file names
    num_files = len(input_files)
    num_unks = 0
    seq_lengths = list()

    print('\n--- Preparing to read', num_files, 'input files for', dataset_name, 'data set')
    seqs = list()
    for i, file in enumerate(input_files):
        if i % 10000 == 0:
            print('\tRead', i, 'files')
        file = file.replace('.ll', '_seq.rec')
        assert os.path.exists(file), 'input file not found: ' + file
        with open(file, 'rb') as f:
            full_seq = f.read()
        seq = list()
        for j in range(0, len(full_seq), 4):    # read 4 bytes at a time
            seq.append(struct.unpack('I', full_seq[j:j + 4])[0])
        assert len(seq) > 0, 'Found empty file: ' + file
        num_unks += seq.count(unk_index)
        seq_lengths.append(len(seq))
        seqs.append([int(s) for s in seq])

    print('\tShortest sequence    : {:>5}'.format(min(seq_lengths)))
    maxlen = max(seq_lengths)
    print('\tLongest sequence     : {:>5}'.format(maxlen))
    print('\tMean sequence length : {:>5} (rounded down)'.format(math.floor(np.mean(seq_lengths))))
    print('\tNumber of \'UNK\'      : {:>5}'.format(num_unks))
    print('\tPercentage of \'UNK\'  : {:>8.4} (% among all stmts)'.format((num_unks*100)/sum(seq_lengths)))
    print('\t\'UNK\' index          : {:>5}'.format(unk_index))

    return seqs, maxlen, seq_lengths


def pad_src(seqs, maxlen, unk_index):
    from keras.preprocessing.sequence import pad_sequences

    encoded = np.array(pad_sequences(seqs, maxlen=maxlen, value=unk_index))
    return np.vstack([np.expand_dims(x, axis=0) for x in encoded])


class EmbeddingSequence(utils.Sequence):
    def __init__(self, batch_size, x_seq, y_1hot, embedding_mat):
        self.batch_size = batch_size
        self.num_samples = np.shape(x_seq)[0]
        self.dataset_len = int(self.num_samples // self.batch_size)
        self.x_seq = x_seq
        self.y_1hot = y_1hot
        self.emb = embedding_mat
        # Make tf block less gpu memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.xplace = tf.placeholder(x_seq.dtype, shape=(None, None))
        self.lookup = tf.nn.embedding_lookup(self.emb, self.xplace)
        self._set_index_array()

    def _set_index_array(self):
        self.index_array = np.random.permutation(self.num_samples)
        x_seq2 = self.x_seq[self.index_array]
        y_1hot2 = self.y_1hot[self.index_array]
        self.x_seq = x_seq2
        self.y_1hot = y_1hot2

    def on_epoch_end(self):
        self._set_index_array()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        idx_begin, idx_end = self.batch_size * idx, self.batch_size * (idx + 1)

        x = self.x_seq[idx_begin:idx_end]
        emb_x = self.sess.run(self.lookup, feed_dict={self.xplace: x})
        return emb_x, self.y_1hot[idx_begin:idx_end]


class EmbeddingPredictionSequence(utils.Sequence):
    def __init__(self, batch_size, x_seq, embedding_mat, seq_len):
        self.batch_size = 1  # batch_size
        self.x_seq = x_seq
        self.seqlen = seq_len
        self.dataset_len = int(math.ceil(np.shape(x_seq)[0] / self.batch_size))
        self.emb = embedding_mat
        # Make tf block less gpu memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.xplace = tf.placeholder(x_seq.dtype, shape=(None,None))
        self.lookup = tf.nn.embedding_lookup(self.emb, self.xplace)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        sl = self.seqlen[idx]
        x = self.x_seq[idx:idx+1,-sl:]
        emb_x = self.sess.run(self.lookup, feed_dict={self.xplace: x})
        
        return emb_x
                        


# Based on https://github.com/tbennun/keras-bucketed-sequence
class BucketedEmbeddingSequence(utils.Sequence):
    def __init__(self, num_buckets, batch_size, seq_lengths, x_seq, y_1hot, embedding_mat):
        self.batch_size = batch_size
        # Count bucket sizes
        bucket_sizes, bucket_ranges = np.histogram(seq_lengths, bins=num_buckets)

        actual_buckets = [bucket_ranges[i+1] for i,bs in enumerate(bucket_sizes) if bs > 0]
        actual_bucketsizes = [bs for bs in bucket_sizes if bs > 0]
        bucket_seqlen = [int(math.ceil(bs)) for bs in actual_buckets]
        num_actual = len(actual_buckets)
        print('Training with %d non-empty buckets' % num_actual)
        print(bucket_seqlen)
        print(actual_bucketsizes)
        self.bins = [(np.ndarray([bs, bsl], dtype=x_seq.dtype),
                      np.ndarray([bs, y_1hot.shape[1]], dtype=x_seq.dtype)) for bsl,bs in zip(bucket_seqlen, actual_bucketsizes)]
        assert len(self.bins) == num_actual
        bctr = [0]*num_actual
       
        for i,sl in enumerate(seq_lengths):
            for j in range(num_actual):
                bsl = bucket_seqlen[j]
                if sl < bsl or j == num_actual - 1:
                    self.bins[j][0][bctr[j],:bsl] = x_seq[i,-bsl:]
                    self.bins[j][1][bctr[j],:] = y_1hot[i]
                    bctr[j] += 1
                    break
        print(bctr)

        self.num_samples = x_seq.shape[0]
        self.dataset_len = int(sum([math.ceil(bs / self.batch_size) for bs in actual_bucketsizes]))
        self.emb = embedding_mat
        # Make tf block less gpu memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.xplace = tf.placeholder(x_seq.dtype, shape=(None, None))
        self.lookup = tf.nn.embedding_lookup(self.emb, self.xplace)
        self._set_index_array()            

    def _set_index_array(self):
        # Shuffle bins
        random.shuffle(self.bins)

        # Shuffle bin contents
        for i, (xbin, ybin) in enumerate(self.bins):
            index_array = np.random.permutation(xbin.shape[0])
            self.bins[i] = (xbin[index_array], ybin[index_array])

    def on_epoch_end(self):
        self._set_index_array()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        idx_begin, idx_end = self.batch_size*idx, self.batch_size*(idx+1)

        # Obtain bin index
        for i,(xbin,ybin) in enumerate(self.bins):
            #print('iter', i, idx_begin, idx_end, xbin.shape[0])
            rounded_bin = _roundto(xbin.shape[0], self.batch_size)
            if idx_begin >= rounded_bin:
                idx_begin -= rounded_bin
                idx_end -= rounded_bin
                continue
            # Found bin
            #print('found', i)
            idx_end = min(xbin.shape[0], idx_end) # Clamp to end of bin

            x = xbin[idx_begin:idx_end]
            emb_x = self.sess.run(self.lookup, feed_dict={self.xplace: x})
            return emb_x, ybin[idx_begin:idx_end]

            
        raise ValueError('out of bounds')
            


class WeightsSaver(Callback):
    def __init__(self, model, save_every, ring_size):
        self.model = model
        self.save_every = save_every
        self.ring_size = ring_size
        self.batch = 0
        self.ring = 0

    def on_epoch_end(self, epoch, logs={}):
        name = FLAGS.out + '/epoch-%d.h5' % epoch
        self.model.save_weights(name)

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.save_every == 0:
            name = FLAGS.out + '/weights%d.h5' % self.ring
            self.model.save_weights(name)
            self.ring = (self.ring + 1) % self.ring_size
        self.batch += 1


########################################################################################################################
# Model
########################################################################################################################
class NCC_classifyapp(object):
    __name__ = "NCC_classifyapp"
    __basename__ = "ncc_classifyapp"

    def init(self, seed: int, maxlen: int, embedding_dim: int, num_classes: int, dense_layer_size: int):
        from keras.layers import Input, LSTM, Dense
        from keras.layers.normalization import BatchNormalization
        from keras.models import Model

        np.random.seed(seed)

        # Keras model
        inp = Input(shape=(None, embedding_dim,), dtype="float32", name="code_in")
        x = LSTM(embedding_dim, implementation=1, return_sequences=True, name="lstm_1")(inp)
        x = LSTM(embedding_dim, implementation=1, name="lstm_2")(x)

        # Heuristic model: outputs 1-of-num_classes prediction
        x = BatchNormalization()(x)
        x = Dense(dense_layer_size, activation="relu")(x)
        outputs = Dense(num_classes, activation="sigmoid")(x)

        self.model = Model(inputs=inp, outputs=outputs)
        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=['accuracy'])
        print('\tbuilt Keras model')

    def save(self, outpath: str):
        self.model.save(outpath)

    def restore(self, inpath: str):
        from keras.models import load_model
        self.model = load_model(inpath)

    def train(self, sequences: np.array, y_1hot: np.array, sequences_val: np.array, 
              y_1hot_val: np.array, verbose: bool, epochs: int, batch_size: int) -> None:
        self.model.fit(x=sequences, y=y_1hot, epochs=epochs, batch_size=batch_size, 
                       verbose=verbose, shuffle=True,
                       validation_data=(sequences_val, y_1hot_val))

    def train_gen(self, train_generator: EmbeddingSequence,
                  validation_generator: EmbeddingSequence,
                  verbose: bool, epochs: int) -> None:
        if FLAGS.restore is not None:
            self.model.load_weights(FLAGS.restore)

        # checkpoint
        filepath = os.path.join(FLAGS.out, "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")
        checkpoint_epoch = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, 
                                           save_best_only=True, mode='max')
        checkpoint = WeightsSaver(self.model, FLAGS.save_every, FLAGS.ring_size)

        try:
            self.model.fit_generator(train_generator, epochs=epochs, verbose=verbose, 
                                     validation_data=validation_generator, shuffle=True, 
                                     callbacks=[checkpoint, checkpoint_epoch])
        except KeyboardInterrupt:
            print('Ctrl-C detected, saving weights to file')
            self.model.save_weights(os.path.join(FLAGS.out, 'weights-kill.h5'))
            import sys
            sys.exit(13)
            

    def predict(self, sequences: np.array, batch_size: int) -> np.array:
        # directly predict application class from source sequences:
        p = np.array(self.model.predict(sequences, batch_size=batch_size, verbose=0))  # one-hot(range([0, 103]))
        indices = [np.argmax(x) for x in p]
        return [i + 1 for i in indices]  # range(y): [1, 104], range(indices): [0, 103]

    def predict_gen(self, generator: EmbeddingSequence) -> np.array:
        # directly predict application class from source sequences:
        p = np.array(self.model.predict_generator(generator, verbose=1))  # one-hot(range([0, 103]))
        indices = [np.argmax(x) for x in p]
        return [i + 1 for i in indices]  # range(y): [1, 104], range(indices): [0, 103]


########################################################################################################################
# Evaluate
########################################################################################################################
def evaluate(model, embeddings, folder_data, samples_per_class, folder_results, dense_layer_size, print_summary,
             num_epochs, batch_size):
    # Set seed for reproducibility
    seed = 204

    ####################################################################################################################
    # Get data
    vsamples_per_class = FLAGS.vsamples

    # Data acquisition
    num_classes = 104

    pickle_preprocess = os.path.join(folder_results, 'input_preprocessed.p')

    if fs.exists(pickle_preprocess):
        print('Loading preprocessed data from', pickle_preprocess)
        [X_seq_train, X_seq_val, X_seq_test, y_1hot_train, y_1hot_val, y_test, maxlen,
         seq_len_train, seq_len_val, seq_len_test, unk_index] = pickle.load(open(pickle_preprocess, 'rb'))
    else:
        y_train = np.empty(0)  # training
        X_train = list()
        folder_data_train = os.path.join(folder_data, 'seq_train')
        y_val = np.empty(0)  # validation
        X_val = list()
        folder_data_val = os.path.join(folder_data, 'seq_val')
        y_test = np.empty(0)  # testing
        X_test = list()
        folder_data_test = os.path.join(folder_data, 'seq_test')
        print('Getting file names for', num_classes, 'classes from folders:')
        print(folder_data_train)
        print(folder_data_val)
        print(folder_data_test)
        for i in range(1, num_classes + 1):  # loop over classes

            # training: Read data file names
            folder = os.path.join(folder_data_train, str(i))  # index i marks the target class
            assert os.path.exists(folder), "Folder: " + folder + ' does not exist'
            print('\ttraining  : Read file names from folder ', folder)
            listing = os.listdir(folder + '/')
            seq_files = [os.path.join(folder, f) for f in listing if f[-4:] == '.rec']

            # training: Randomly pick programs
            assert len(seq_files) >= samples_per_class, "Cannot sample " + str(samples_per_class) + " from " + str(
                len(seq_files)) + " files found in " + folder
            X_train += resample(seq_files, replace=False, n_samples=samples_per_class, random_state=seed)
            y_train = np.concatenate([y_train, np.array([int(i)] * samples_per_class, dtype=np.int32)])  # i becomes target

            # validation: Read data file names
            folder = os.path.join(folder_data_val, str(i))
            assert os.path.exists(folder), "Folder: " + folder + ' does not exist'
            print('\tvalidation: Read file names from folder ', folder)
            listing = os.listdir(folder + '/')
            seq_files = [os.path.join(folder, f) for f in listing if f[-4:] == '.rec']

            # validation: Randomly pick programs
            if vsamples_per_class > 0:
                assert len(seq_files) >= vsamples_per_class, "Cannot sample " + str(vsamples_per_class) + " from " + str(
                    len(seq_files)) + " files found in " + folder
                X_val += resample(seq_files, replace=False, n_samples=vsamples_per_class, random_state=seed)
                y_val = np.concatenate([y_val, np.array([int(i)] * vsamples_per_class, dtype=np.int32)])
            else:
                assert len(seq_files) > 0, "No .rec files found in" + folder
                X_val += seq_files
                y_val = np.concatenate([y_val, np.array([int(i)] * len(seq_files), dtype=np.int32)])

            # test: Read data file names
            folder = os.path.join(folder_data_test, str(i))
            assert os.path.exists(folder), "Folder: " + folder + ' does not exist'
            print('\ttest      : Read file names from folder ', folder)
            listing = os.listdir(folder + '/')
            seq_files = [os.path.join(folder, f) for f in listing if f[-4:] == '.rec']
            assert len(seq_files) > 0, "No .rec files found in" + folder
            X_test += seq_files
            y_test = np.concatenate([y_test, np.array([int(i)] * len(seq_files), dtype=np.int32)])

        # Load dictionary and cutoff statements
        folder_vocabulary = FLAGS.vocabulary_dir
        dictionary_pickle = os.path.join(folder_vocabulary, 'dic_pickle')
        print('\tLoading dictionary from file', dictionary_pickle)
        with open(dictionary_pickle, 'rb') as f:
            dictionary = pickle.load(f)
        unk_index = dictionary[rgx.unknown_token]
        del dictionary

        # Encode source codes and get max. sequence length
        X_seq_train, maxlen_train, seq_len_train = encode_srcs(X_train, 'training', unk_index)
        X_seq_val, maxlen_val, seq_len_val = encode_srcs(X_val, 'validation', unk_index)
        X_seq_test, maxlen_test, seq_len_test = encode_srcs(X_test, 'testing', unk_index)
        maxlen = max(maxlen_train, maxlen_test, maxlen_val)
        print('Max. sequence length overall:', maxlen)
        print('Padding sequences')
        X_seq_train = pad_src(X_seq_train, maxlen, unk_index)
        X_seq_val = pad_src(X_seq_val, maxlen, unk_index)
        X_seq_test = pad_src(X_seq_test, maxlen, unk_index)

        # Get one-hot vectors for classification
        print('YTRAIN\n', y_train)
        y_1hot_train = get_onehot(y_train, num_classes)
        y_1hot_val = get_onehot(y_val, num_classes)

        print('Saving preprocessed inputs to file')
        pickle.dump([X_seq_train, X_seq_val, X_seq_test, y_1hot_train, y_1hot_val, y_test, maxlen,
                     seq_len_train, seq_len_val, seq_len_test, unk_index],
                    open(pickle_preprocess, 'wb'))


    ####################################################################################################################
    # Setup paths

    # Set up names paths
    model_name = model.__name__
    model_path = os.path.join(folder_results,
                              "models/{}.model".format(model_name))
    predictions_path = os.path.join(folder_results,
                                    "predictions/{}.result".format(model_name))

    # If predictions have already been made with these embeddings, load them
    if fs.exists(predictions_path):
        print("\tFound predictions in", predictions_path, ", skipping...")
        with open(predictions_path, 'rb') as infile:
            p = pickle.load(infile)

    else:  # could not find predictions already computed with these embeddings

        # Embeddings
        import tensorflow as tf  # for embeddings lookup
        embedding_matrix_normalized = tf.nn.l2_normalize(embeddings, axis=1)
        vocabulary_size, embedding_dimension = embedding_matrix_normalized.shape
        print('XSEQ:\n', X_seq_train)
        print('EMB:\n', embedding_matrix_normalized)

        gen_test = EmbeddingPredictionSequence(batch_size, X_seq_test, embedding_matrix_normalized, seq_len_test)

        # If models have already been made with these embeddings, load them
        if fs.exists(model_path):
            print("\n\tFound trained model in", model_path, ", skipping...")
            model.restore(model_path)

        else:  # could not find models already computed with these embeddings

            if FLAGS.buckets <= 0 and FLAGS.bucketstr is None:
                gen_train = EmbeddingSequence(batch_size, X_seq_train, y_1hot_train, embedding_matrix_normalized)
            else:
                buckets = FLAGS.bucketstr if FLAGS.bucketstr is not None else FLAGS.buckets
                gen_train = BucketedEmbeddingSequence(buckets, batch_size, seq_len_train, X_seq_train, y_1hot_train, 
                                                      embedding_matrix_normalized)
                                 
            gen_val = EmbeddingSequence(batch_size, X_seq_val, y_1hot_val, embedding_matrix_normalized)

            ############################################################################################################
            # Train

            # Create a new model and train it
            print('\n--- Initializing model...')
            model.init(seed=seed,
                       maxlen=maxlen,
                       embedding_dim=int(embedding_dimension),
                       num_classes=num_classes,
                       dense_layer_size=dense_layer_size)
            if print_summary:
                model.model.summary()
            print('\n--- Training model...')
            model.train_gen(train_generator=gen_train,
                            validation_generator=gen_val,
                            verbose=True,
                            epochs=num_epochs)

            # Save the model
            fs.mkdir(fs.dirname(model_path))
            model.save(model_path)
            print('\tsaved model to', model_path)

        ################################################################################################################
        # Test

        # Test model
        print('\n--- Testing model...')
        p = model.predict_gen(generator=gen_test)

        # cache the prediction
        fs.mkdir(fs.dirname(predictions_path))
        with open(predictions_path, 'wb') as outfile:
            pickle.dump(p, outfile)
        print('\tsaved predictions to', predictions_path)

    ####################################################################################################################
    # Return accuracy
    accuracy = np.array(p) == np.array(y_test)   # prediction accuracy
    return accuracy


########################################################################################################################
# Main
########################################################################################################################
def main(argv):
    del argv    # unused

    ####################################################################################################################
    # Setup
    # Get flag values
    embeddings = task_utils.get_embeddings()
    folder_results = FLAGS.out
    assert len(folder_results) > 0, "Please specify a path to the results folder using --folder_results"
    folder_data = FLAGS.input_data
    dense_layer_size = FLAGS.dense_layer
    print_summary = FLAGS.print_summary
    num_epochs = FLAGS.num_epochs
    batch_size = FLAGS.batch_size
    train_samples = FLAGS.train_samples

    # Acquire data
    if not os.path.exists(os.path.join(folder_data, 'ir_train')):
        # Download data
        task_utils.download_and_unzip('https://polybox.ethz.ch/index.php/s/JOBjrfmAjOeWCyl/download',
                                      'classifyapp_training_data', folder_data)

    task_utils.llvm_ir_to_trainable(os.path.join(folder_data, 'ir_train'))
    assert os.path.exists(os.path.join(folder_data, 'ir_val')), "Folder not found: " + folder_data + '/ir_val'
    task_utils.llvm_ir_to_trainable(os.path.join(folder_data, 'ir_val'))
    assert os.path.exists(os.path.join(folder_data, 'ir_test')), "Folder not found: " + folder_data + '/ir_test'
    task_utils.llvm_ir_to_trainable(os.path.join(folder_data, 'ir_test'))


    # Create directories if they do not exist
    if not os.path.exists(folder_results):
        os.makedirs(folder_results)
    if not os.path.exists(os.path.join(folder_results, "models")):
        os.makedirs(os.path.join(folder_results, "models"))
    if not os.path.exists(os.path.join(folder_results, "predictions")):
        os.makedirs(os.path.join(folder_results, "predictions"))

    ####################################################################################################################
    # Train model
    # Evaluate Classifyapp
    print("\nEvaluating ClassifyappInst2Vec ...")
    classifyapp_accuracy = evaluate(NCC_classifyapp(), embeddings, folder_data, train_samples, folder_results,
                                    dense_layer_size, print_summary, num_epochs, batch_size)

    ####################################################################################################################
    # Print results
    print('\nTest accuracy:', sum(classifyapp_accuracy)*100/len(classifyapp_accuracy), '%')


if __name__ == '__main__':
    app.run(main)
