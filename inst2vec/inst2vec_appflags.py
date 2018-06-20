# NCC: Neural Code Comprehension
# https://github.com/spcl/ncc
# Copyright 2018 ETH Zurich
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the follo
# wing conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================
"""Execution flags for inst2vec parameters"""

from absl import flags

# Data set parameters
flags.DEFINE_string('data', 'data', 'Dataset to use')
flags.DEFINE_string('data_folder', '', 'Dataset folder prefix')

# Vocabulary parameters
flags.DEFINE_integer('context_width', 2, 'width of skip-gram context')
flags.DEFINE_integer('cutoff_unknown', 300,
                     'replace stmts which appear less than "cutoff" times by "unknown token')
flags.DEFINE_float('subsampling', 1e-7, 'frequent pairs subsampling')

# Parameters of inst2vec (default)
flags.DEFINE_integer('embedding_size', 200, 'Dimension of embedding space')
flags.DEFINE_integer('mini_batch_size', 64, 'size of mini-batches of data to feed the neural network')
flags.DEFINE_integer('num_sampled', 60, 'number of negative classes to sample/batch for NCE')
flags.DEFINE_integer('num_epochs', 5, 'number of training epochs')
flags.DEFINE_float('learning_rate', 0.001, "learning rate used by Adam's optimizer")
flags.DEFINE_float('beta', 0.0, 'scale of L2 regularization applied to weights (0: no regularization)')

# Embedding training parameters
flags.DEFINE_string('embeddings_folder', 'data/emb',
                    'Folder in which to store embedding training data and their evaluation')
flags.DEFINE_integer('freq_print_loss', 100, 'how many times to print the average loss per epoch (0: no printing)')
flags.DEFINE_integer('step_print_neighbors', -1, 'frequency for printing nearest neighbours ' +
                     '(0: no printing, -1: print at last step)')
flags.DEFINE_bool('restore', False, 'Restore from checkpoint')
flags.DEFINE_bool('profile', False, 'Write traces to Chrome tracing JSON')
flags.DEFINE_bool('xla', False, 'Use XLA JIT compilation (need to compile TF from source)')
flags.DEFINE_string('optimizer', 'adam', 'Choose an optimizer (options: adam, nadam, momentum)')
flags.DEFINE_bool('extreme', False, 'Kill training every step')
flags.DEFINE_bool('softmax', True, 'Use softmax instead of NCE')
flags.DEFINE_string('savebest', None, 'Folder to save the best training results to')

# Parameters of embeddings intrinsic evaluation
flags.DEFINE_string('embeddings_file', '',
                    'Path to embeddings file to be used for evaluation and NCC task training')
flags.DEFINE_string('vocabulary_folder', '',
                    'Path to the vocabulary folder associated with those embeddings')
flags.DEFINE_bool('verbose', False, 'Use verbosity in UMAP')
flags.DEFINE_string('metric', 'euclidean', 'Distance metric for UMAP')
flags.DEFINE_bool('tsne', True, 'Use t-SNE')
flags.DEFINE_bool('newtags', False, 'Use type-based tags')
flags.DEFINE_string('taglist', 'tags.np', 'Use existing taglist (or save to it)')
