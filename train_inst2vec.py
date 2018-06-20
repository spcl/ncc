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
"""Main inst2vec and ncc workflow"""


import os
import pickle
from inst2vec import inst2vec_datagen as i2v_datagen
from inst2vec import inst2vec_preprocess as i2v_prep
from inst2vec import inst2vec_vocabulary as i2v_vocab
from inst2vec import inst2vec_embedding as i2v_emb
from inst2vec import inst2vec_evaluate as i2v_eval
from inst2vec import inst2vec_appflags
from absl import flags, app

FLAGS = flags.FLAGS


def main(argv):
    del argv  # unused

    data_folder = os.path.join(FLAGS.data_folder, FLAGS.data)
    if not os.path.exists(FLAGS.embeddings_file):

        if FLAGS.data == "data":
            # Generate the data set
            i2v_datagen.datagen(data_folder)
        else:
            # Assert the data folder's existence
            assert os.path.exists(data_folder), "Folder " + data_folder + " does not exist"

        # Build XFGs from raw code
        data_folders = i2v_prep.construct_xfg(data_folder)

        # Build vocabulary
        i2v_vocab.construct_vocabulary(data_folder, data_folders)

        # Train embeddings
        embedding_matrix, embeddings_file = i2v_emb.train_embeddings(data_folder, data_folders)

    else:

        print('Loading pre-trained embeddings from', FLAGS.embeddings_file)
        with open(FLAGS.embeddings_file, 'rb') as f:
            embedding_matrix = pickle.load(f)
        embeddings_file = FLAGS.embeddings_file

    # Evaluate embeddings (intrinsic evaluation)
    i2v_eval.evaluate_embeddings(data_folder, embedding_matrix, embeddings_file)


if __name__ == '__main__':
    app.run(main)
