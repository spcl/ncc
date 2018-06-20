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
"""Training workflow for optimal thread coarsening factor prediction"""
# Based on: https://github.com/ChrisCummins/paper-end2end-dl/blob/master/code/Case%20Study%20B.ipynb

from sklearn.model_selection import KFold
import rgx_utils as rgx
import task_utils
from labm8 import fs
import numpy as np
import pandas as pd
import os
import pickle
import math
from absl import app
from absl import flags

flags.DEFINE_string('input_data', 'task/threadcoarsening', 'Path to input data')
flags.DEFINE_string('out', 'task/threadcoarsening',
                    'Path to folder in which to write saved Keras models and predictions')
flags.DEFINE_string('device', 'all', 'Device to evaluate model on. Options: all, Cypress, Tahiti, Fermi, Kepler')
flags.DEFINE_integer('num_epochs', 50, 'number of training epochs')
flags.DEFINE_integer('batch_size', 64, 'training batch size')
flags.DEFINE_integer('dense_layer', 32, 'dense layer size')
flags.DEFINE_bool('print_summary', False, 'Print summary of Keras model')

FLAGS = flags.FLAGS

_FLAG_TO_DEVICE_NAME = {
    'Cypress': 'AMD Radeon HD 5900',
    'Tahiti': 'AMD Tahiti 7970',
    'Fermi': 'NVIDIA GTX 480',
    'Kepler': 'NVIDIA Tesla K20c'
}


########################################################################################################################
# Utils
########################################################################################################################
cfs = [1, 2, 4, 8, 16, 32]  # thread coarsening factors


def get_onehot(df, platform):
    hot = np.zeros((len(df), len(cfs)), dtype=np.int32)
    for i, cf in enumerate(df["cf_{}".format(platform)]):
        hot[i][cfs.index(cf)] = 1

    return hot


def get_magni_features(df, oracles, platform):
    X_cc, y_cc, = [], []
    for kernel in sorted(set(df["kernel"])):
        _df = df[df["kernel"] == kernel]

        oracle_cf = int(oracles[oracles["kernel"] == kernel]["cf_{}".format(platform)].values[0])

        feature_vectors = np.asarray([
            _df['PCA1'].values,
            _df['PCA2'].values,
            _df['PCA3'].values,
            _df['PCA4'].values,
            _df['PCA5'].values,
            _df['PCA6'].values,
            _df['PCA7'].values,
        ]).T

        X_cc.append(feature_vectors)
        y = []
        cfs__ = []
        for i, cf in enumerate(cfs[:len(feature_vectors)]):
            y_ = 1 if cf < oracle_cf else 0
            y.append(y_)
        y_cc.append(y)

        assert len(feature_vectors) == len(y)

    assert len(X_cc) == len(y_cc) == 17

    return np.asarray(X_cc), np.asarray(y_cc)


def encode_srcs(data_folder, df: pd.DataFrame):
    """
    encode and pad source code for learning
    """
    from keras.preprocessing.sequence import pad_sequences

    # Load dictionary and cutoff statements
    folder_vocabulary = FLAGS.vocabulary_dir
    dictionary_pickle = os.path.join(folder_vocabulary, 'dic_pickle')
    print('\tLoading dictionary from file', dictionary_pickle)
    with open(dictionary_pickle, 'rb') as f:
        dictionary = pickle.load(f)
    unk_index = dictionary[rgx.unknown_token]
    del dictionary

    # Get list of source file names
    data_folder = os.path.join(data_folder, 'kernels_seq')
    input_files = df["kernel"].values   # list of strings of kernel names
    num_files = len(input_files)
    num_unks = 0
    seq_lengths = list()

    print('\n--- Preparing to read', num_files, 'input files from folder', data_folder)
    seqs = list()
    for file in input_files:

        file = os.path.join(data_folder, file + '_seq.csv')
        assert os.path.exists(file), 'input file not found: ' + file
        with open(file, 'r') as f:
            seq = f.read().splitlines()
        assert len(seq) > 0, 'Found empty file: ' + file
        num_unks += seq.count(str(unk_index))
        seq_lengths.append(len(seq))
        seqs.append([int(s) for s in seq])

        print('\tShortest sequence    : {:>5}'.format(min(seq_lengths)))
        maxlen = max(seq_lengths)
        print('\tLongest sequence     : {:>5}'.format(maxlen))
        print('\tMean sequence length : {:>5} (rounded down)'.format(math.floor(np.mean(seq_lengths))))
        print('\tNumber of \'UNK\'      : {:>5}'.format(num_unks))
        print('\tPercentage of \'UNK\'  : {:>8.4} (% among all stmts)'.format((num_unks*100)/sum(seq_lengths)))
        print('\t\'UNK\' index          : {:>5}'.format(unk_index))

    encoded = np.array(pad_sequences(seqs, maxlen=maxlen, value=unk_index))
    return np.vstack([np.expand_dims(x, axis=0) for x in encoded]), maxlen


def platform2str(platform):
    if platform == "Fermi":
        return "NVIDIA GTX 480"
    elif platform == "Kepler":
        return "NVIDIA Tesla K20c"
    elif platform == "Cypress":
        return "AMD Radeon HD 5900"
    elif platform == "Tahiti":
        return "AMD Tahiti 7970"
    else:
        raise LookupError


########################################################################################################################
# Model
########################################################################################################################
class NCC_threadcoarsening:
    __name__ = "NCC_threadcoarsening"
    __basename__ = "ncc_threadcoarsening"

    def init(self, seed: int, maxlen: int, embedding_dim: int, dense_layer_size: int):
        from keras.layers import Input, LSTM, Dense
        from keras.layers.normalization import BatchNormalization
        from keras.models import Model

        np.random.seed(seed)

        # Model
        inp = Input(shape=(maxlen, embedding_dim,), dtype="float32", name="code_in")
        x = LSTM(embedding_dim, implementation=1, return_sequences=True, name="lstm_1")(inp)
        x = LSTM(embedding_dim, implementation=1, name="lstm_2")(x)
        x = BatchNormalization()(x)
        x = Dense(dense_layer_size, activation="relu")(x)
        outputs = Dense(6, activation="sigmoid")(x)
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

    def train(self, sequences: np.array, y_1hot: np.array, verbose: bool, epochs: int, batch_size: int) -> None:
        self.model.fit(sequences, y_1hot, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True)

    def predict(self, sequences: np.array, batch_size: int) -> np.array:
        # directly predict optimal thread coarsening factor from source sequences:
        p = np.array(self.model.predict(sequences, batch_size=batch_size, verbose=0))
        indices = [np.argmax(x) for x in p]
        return [cfs[x] for x in indices]


########################################################################################################################
# Evaluate
########################################################################################################################
# Set seed for reproductibility
seed = 204


def evaluate(model, device, data_folder, out_folder, embeddings, dense_layer_size, print_summary, num_epochs,
             batch_size):

    data = []

    # Create device list
    if device == 'all':
        device_list = ["Cypress", "Tahiti", "Fermi", "Kepler"]
    else:
        device_list = [device]

    for i, platform in enumerate(device_list):
        print('\n------------------------------------------------------------------')
        print('--- Platform', platform, '[', i+1, '/ 4 ]')
        print('------------------------------------------------------------------')
        platform_name = platform2str(platform)

        # Read data
        oracle_file = os.path.join(data_folder, "pact-2014-oracles.csv")
        oracles = pd.read_csv(oracle_file)
        runtimes_file = os.path.join(data_folder, "pact-2014-runtimes.csv")
        df = pd.read_csv(runtimes_file)
        print('\tRead data from', oracle_file, '\n\tand', runtimes_file)

        # Extract data
        oracle_runtimes = np.array([float(x) for x in oracles["runtime_" + platform]])
        y = np.array([int(x) for x in oracles["cf_" + platform]], dtype=np.int32)
        y_1hot = get_onehot(oracles, platform)

        # Encode source codes
        X_seq, maxlen = encode_srcs(data_folder, df)

        # Embeddings
        import tensorflow as tf  # for embeddings lookup
        embedding_matrix_normalized = tf.nn.l2_normalize(embeddings, axis=1)
        vocabulary_size, embedding_dimension = embedding_matrix_normalized.shape
        seq_ = tf.placeholder(dtype=tf.int32)

        # Tensor of shape (num_input_files, sequence length, embbedding dimension)
        embedding_input_ = tf.nn.embedding_lookup(embedding_matrix_normalized, seq_)
        with tf.Session() as sess:
            embedding_input = sess.run(embedding_input_, feed_dict={seq_: X_seq})

        # Leave-one-out cross-validation
        kf = KFold(n_splits=len(y), shuffle=False)

        for j, (train_index, test_index) in enumerate(kf.split(y)):
            print('--- Cross validation step [', j+1, '/ ', len(y), ']')
            kernel = sorted(set(df["kernel"]))[test_index[0]]
            X_cc, y_cc = get_magni_features(df, oracles, platform)

            model_name = model.__name__
            model_basename = model.__basename__

            model_path = os.path.join(out_folder, "models/{model_basename}-{platform}-{j}.model".format(
                model_basename=model_basename, platform=platform, j=j))
            predictions_path = os.path.join(out_folder, "predictions/{model_basename}-{platform}-{j}.result".format(
                model_basename=model_basename, platform=platform, j=j))

            if fs.exists(predictions_path):
                # load result from cache
                print("\tFound predictions in", predictions_path, ", skipping...")
                with open(predictions_path, 'rb') as infile:
                    p = pickle.load(infile)
            else:

                if fs.exists(model_path):
                    # load a trained model from cache
                    print("\n\tFound trained model in", model_path, ", skipping...")
                    model.restore(model_path)
                else:

                    # Initialize model and print summary
                    print('\n--- Training model...')
                    model.init(seed, maxlen, int(embedding_dimension), dense_layer_size)
                    if print_summary:
                        model.model.summary()

                    # Train and cache a model
                    model.train(sequences=embedding_input[train_index, :, :],
                                verbose=True,
                                y_1hot=y_1hot[train_index],
                                epochs=num_epochs,
                                batch_size=batch_size)

                    # cache the model
                    fs.mkdir(fs.dirname(model_path))
                    model.save(model_path)
                    print('\tsaved model to', model_path)

                # test model
                print('\n--- Testing model...')
                p = model.predict(sequences=embedding_input[test_index, :, :], batch_size=batch_size)[0]

                # The runtimes of some coarsening factors are not recorded in the data table. If that is the case for
                # the predicted cf, clamp it down to the highest cf for which the runtime is recorded
                p = min(p, 2 ** (len(X_cc[test_index[0]]) - 1))

                # cache the prediction
                fs.mkdir(fs.dirname(predictions_path))
                with open(predictions_path, 'wb') as outfile:
                    pickle.dump(p, outfile)
                print('\tsaved predictions to', predictions_path)

            o = y[test_index[0]]    # oracle prediction (true value)
            correct = p == o        # predictions' correctness

            # get runtime without thread coarsening
            row = df[(df["kernel"] == kernel) & (df["cf"] == 1)]
            assert (len(row) == 1)  # sanity check
            nocf_runtime = float(row["runtime_" + platform])

            # get runtime of prediction
            row = df[(df["kernel"] == kernel) & (df["cf"] == p)]
            assert (len(row) == 1)  # sanity check
            p_runtime = float(row["runtime_" + platform])

            # get runtime of oracle coarsening factor
            o_runtime = oracle_runtimes[test_index[0]]

            # speedup and % oracle
            s_oracle = nocf_runtime / o_runtime
            p_speedup = nocf_runtime / p_runtime
            p_oracle = o_runtime / p_runtime

            # record result
            data.append({
                "Model": model_name,
                "Platform": platform_name,
                "Kernel": kernel,
                "Oracle-CF": o,
                "Predicted-CF": p,
                "Speedup": p_speedup,
                "Oracle": p_oracle
            })

    return pd.DataFrame(data, columns=[
        "Model", "Platform", "Kernel", "Oracle-CF", "Predicted-CF", "Speedup", "Oracle"])


########################################################################################################################
# Main
########################################################################################################################
def main(argv):
    del argv    # unused

    ####################################################################################################################
    # Setup
    # Get flag values
    embeddings = task_utils.get_embeddings()
    input_data = FLAGS.input_data
    assert os.path.exists(input_data), "Folder not found: " + input_data
    task_utils.llvm_ir_to_trainable(os.path.join(input_data, 'kernels_ir'))
    out = FLAGS.out
    if not os.path.exists(out):
        os.makedirs(out)
    device = FLAGS.device
    assert device in ["all", "Cypress", "Tahiti", "Fermi", "Kepler"], \
        'Choose device among: all, Cypress, Tahiti, Fermi, Kepler'
    dense_layer_size = FLAGS.dense_layer
    print_summary = FLAGS.print_summary
    num_epochs = FLAGS.num_epochs
    batch_size = FLAGS.batch_size

    ####################################################################################################################
    # Reference values
    # Values copied from papers and github
    magni_pl_sp_vals = [1.21, 1.01, 0.86, 0.94]
    magni_sp_mean = 1.005
    deeptune_pl_sp_vals = [1.10, 1.05, 1.10, 0.99]
    deeptune_sp_mean = 1.06
    deeptuneTL_pl_sp_vals = [1.17, 1.23, 1.14, 0.93]
    deeptuneTL_sp_mean = 1.1175

    ####################################################################################################################
    # Train model
    # Evaluate NCC_threadcoarsening
    print("\nEvaluating NCC_threadcoarsening ...")
    ncc_threadcoarsening = evaluate(NCC_threadcoarsening(), device, input_data, out, embeddings, dense_layer_size,
                                    print_summary, num_epochs, batch_size)

    ####################################################################################################################
    # Print results
    print('\n', ncc_threadcoarsening.groupby('Platform')['Platform', 'Speedup', 'Oracle'].mean())
    d = np.array([ncc_threadcoarsening[['Speedup', 'Oracle']].mean()]).T
    print('\n', pd.DataFrame(d, columns=["DeepTuneInst2Vec"], index=["Speedup", "Oracle"]))

    # Model comparison: speedups
    print('\nModel comparison: speedups')
    d = list()
    d.append(np.append(magni_pl_sp_vals, magni_sp_mean))
    d.append(np.append(deeptune_pl_sp_vals, deeptune_sp_mean))
    d.append(np.append(deeptuneTL_pl_sp_vals, deeptuneTL_sp_mean))
    d.append(np.append(ncc_threadcoarsening.groupby(['Platform'])['Speedup'].mean().values,
                       ncc_threadcoarsening['Speedup'].mean()))
    if FLAGS.device == 'all':
        d = np.array(d).T.reshape(5, 4)
        devs = ['AMD Radeon HD 5900', 'AMD Tahiti 7970',
                'NVIDIA GTX 480', 'NVIDIA Tesla K20c', 'Average']
    else:
        d = np.array(d).T.reshape(1, 4)
        devs = [_FLAG_TO_DEVICE_NAME[FLAGS.device]]
    print('\n', pd.DataFrame(d, columns=['Magni et al.', 'DeepTune', 'DeepTuneTL', 'DeepTuneInst2Vec'], index=devs))


if __name__ == '__main__':
    app.run(main)

