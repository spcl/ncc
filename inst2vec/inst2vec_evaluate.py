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
"""inst2vec embedding intrinsic evaluation"""
import os
import pickle
import re
import tensorflow as tf
import numpy as np
import rgx_utils as rgx
from inst2vec import inst2vec_utils as i2v_utils
from inst2vec import inst2vec_analogygen as analogygen
from inst2vec import inst2vec_appflags
from bokeh.plotting import figure, output_file, show
from bokeh.models import CategoricalColorMapper, ColumnDataSource
from bokeh.palettes import Category20
import umap
from sklearn.manifold import TSNE
from absl import flags

FLAGS = flags.FLAGS


########################################################################################################################
# Reading, writing and dumping files
########################################################################################################################
def write_score_summary(scores, analogy_types, filename):
    """
    Write score summary to a string
    :param scores: list of pairs
                   (number of correctly answered questions in category, number of questions in category)
    :param analogy_types:
    :param filename:
    :return: score summary (string)
    """
    # Helper: column width in printout
    w = '120'

    # Header
    out = '-' * int(w) + '\n' + filename + ': score summary' + '\n' + '-' * int(w) + '\n'
    out += '{:<65}\t{:>12}\t{:>12}\t{:>15}\n'.format(
        'Analogy type', '#corr answ', '#questions', 'corr answ [%]')

    # Score summary
    for i, analogy_type in enumerate(analogy_types):
        s = scores[i][0]
        nq = scores[i][1]
        perc = s * 100.0 / nq

        # Write the line containing all information pertaining to this analogy type
        out += '{:<65}\t{:>12}\t{:>12}\t{:>15.4g}\n'.format(analogy_type, s, nq, perc)

    return out


########################################################################################################################
# Analogies: helper functions
########################################################################################################################
def nop(*args, **kwargs):
    return


def load_analogy_questions(analogy_questions_file, dictionary):
    """
    :param analogy_questions_file:
    :param dictionary: dictionary where dictionary[word] = index in data
    :return: analogies: list of
                        numpy array [n_questions, 4]
                        containing analogy questions in form of word ids
    :return analogy_types: list of descriptions of types of analogies

    """
    print('\tLoading analogy questions from file ', analogy_questions_file)

    # Read analogy questions
    with open(analogy_questions_file) as f:
        raw_data = f.read().splitlines()

    # Helper variables
    analogies = list()
    analogy_types = list()
    arr = []                # temporary storage for analogy questions of the same "analogy type"
    n_questions = 0

    # Construct analogy questions in terms of word indices
    i = 1
    while True:

        line = raw_data[i]

        # Ignore empty lines and super-comment lines
        if len(line) > 0 and line[0:2] != '##':
            if line[0] == '#':

                # beginning of a new type of analogy
                if len(arr) > 0:
                    # append temporary storage array containing
                    analogies.append(np.array(arr, dtype=np.int32))
                    print('Appending', len(arr), 'questions')
                else:
                    if len(analogy_types) > 0:
                        # remove previous "title"
                        print('Removing', analogy_types[-1])
                        del analogy_types[-1]

                # reset temporary storage array to "empty"
                arr = []

                # append new analogy category title
                print('\tFound', line[1:])
                analogy_types.append(line[1:].strip())

            else:

                # a new analogy question element
                n_questions += 1
                word_A = line.strip();          assert len(word_A) > 0, "Malformed question at line " + str(i)
                word_B = raw_data[i+1].strip(); assert len(word_A) > 0, "Malformed question at line " + str(i)
                word_X = raw_data[i+2].strip(); assert len(word_A) > 0, "Malformed question at line " + str(i)
                word_Y = raw_data[i+3].strip(); assert len(word_A) > 0, "Malformed question at line " + str(i)
                words = [word_A, word_B, word_X, word_Y]

                # skip to the end of the question
                i += 3

                # check whether all the words in this question appear in our data set
                add = True
                for w in words:
                    if w not in dictionary.keys():
                        add = False
                        break

                # add this analogy question when appropriate
                if add:
                    arr.append([dictionary[words[0]], dictionary[words[1]],
                                dictionary[words[2]], dictionary[words[3]]])

                if n_questions % 1000 == 0:
                    print('\t\tprocessed', n_questions, 'analogies')

        i += 1
        if i == len(raw_data[1:]):
            break

    # Append the last temporary array
    if len(arr) > 0:
        # append temporary storage array containing
        print('Appending', len(arr), 'questions')
        analogies.append(np.array(arr, dtype=np.int32))
    else:
        print('Empty category', analogy_types[-1])
        if len(analogy_types) > 0:
            # remove previous "title"
            print('Removing', analogy_types[-1])
            del analogy_types[-1]

    # Print the number of questions read
    print('\tFound {:>10,d} analogy-questions in file'.format(n_questions))

    # Make sure there are no duplicate analogy-questions
    super_a = analogies[0]
    for a in analogies[1:]:
        super_a = np.concatenate((super_a, a), axis=0)
    nrows = super_a.shape[0]
    for row in range(nrows):
        for j in range(row + 1, nrows):
            if np.array_equal(super_a[row], super_a[j]):  # compare rows
                # assert False, "Found duplicate questions in file " + analogy_questions_file
                print("Found duplicate questions in file " + analogy_questions_file)

    # Print the number of "relevant questions"
    print('\tof which {:>10,d} are compatible with this dataset'.format(nrows))

    return analogies, analogy_types, n_questions, nrows


def load_analogies(data_folder):

    ####################################################################################################################
    # Generate analogy "questions" and write them to a file
    eval_folder = os.path.join(FLAGS.embeddings_folder, "eval")
    folder_analogies = os.path.join(eval_folder, "analogy")

    if not os.path.exists(folder_analogies):
        os.makedirs(folder_analogies)
    analogy_questions_file = os.path.join(folder_analogies, "questions.txt")
    if not os.path.exists(analogy_questions_file):
        print('\n--- Generating analogy questions and write them to a file')
        analogygen.generate_analogy_questions(analogy_questions_file)

    ####################################################################################################################
    # Read analogy "questions" from file
    folder_vocabulary = os.path.join(data_folder, "vocabulary")
    dictionary_pickle = os.path.join(folder_vocabulary, 'dic_pickle')
    print('\tLoading dictionary from file', dictionary_pickle)
    with open(dictionary_pickle, 'rb') as f:
        dictionary = pickle.load(f)

    analogy_questions_file_dump = os.path.join(folder_analogies, "questions")
    if not os.path.exists(analogy_questions_file_dump):

        # Read analogies from external file
        print('\n--- Read analogies from file ', analogy_questions_file)
        analogies, analogy_types, n_questions_total, n_questions_relevant = \
            load_analogy_questions(analogy_questions_file, dictionary)

        # Dump analogies into a file to be reused
        print('\n--- Writing analogies into file ', analogy_questions_file_dump)
        i2v_utils.safe_pickle([analogies, analogy_types, n_questions_total, n_questions_relevant], analogy_questions_file_dump)

    else:

        # Load analogies from binary file
        print('\n--- Loading analogies from file ', analogy_questions_file_dump)
        with open(analogy_questions_file_dump, 'rb') as f:
            analogies, analogy_types, n_questions_total, n_questions_relevant = pickle.load(f)

    # Print info
    print('\tFound    {:>10,d} analogy-questions, '.format(n_questions_total))
    print('\tof which {:>10,d} are compatible with this vocabulary'.format(n_questions_relevant))

    return analogies, analogy_types, n_questions_total, n_questions_relevant


def evaluate_analogies(W, reverse_dictionary, analogies, analogy_types, results_filename, session=None, print=print):
    """
    Build evaluation graph and evaluate the representation of analogies in the embeddings
    :param W: embeddings (not normalized)
              Ndarray of shape: (vocabulary_size, embedding_dimension)
    :param reverse_dictionary:  reverse_dictionary[index (int)] = statement (string)
                                length = vocabulary_size
    :param analogies: list of ndarrays
                      len(analogies) = number of different analogy types
                      elements are ndarrays of shape (#questions, 4)
                      which correspond to analogy questions represented as indices
    :param analogy_types: list of string containing descriptions of types of analogies
                          len(analogy_types) = number of different analogy types
    :param results_filename: name of file in which to write results of the evaluation (string)
    :param session:
    :param print:
    """
    # Print message
    print('\tEvaluating analogies (using cosine similarity)...\n')

    # Build the computational graph
    # Placeholders to fill in with indices of words in an analogy question
    # (dim = [N], N = number of analogies to evaluate in one batch) (or rather: unknown at this point)
    analogy_a = tf.placeholder(dtype=tf.int32)
    analogy_b = tf.placeholder(dtype=tf.int32)
    analogy_c = tf.placeholder(dtype=tf.int32)

    # Embeddings (dim = [vocabulary size, embedding dimension])
    embedding_matrix = tf.nn.l2_normalize(W, axis=1)
    vocabulary_size_dic = len(reverse_dictionary)
    vocabulary_size, embedding_dimension = W.shape
    assert vocabulary_size_dic == vocabulary_size, \
        "Vocabulary size of embedding matrix (" + str(vocabulary_size) + \
        ") does not match vocabulary size of dictionary (" + str(vocabulary_size_dic) + ")"

    # Embedding vectors corresponding to the input words (dim = [N, embedding dimension])
    embedded_a = tf.nn.embedding_lookup(embedding_matrix, analogy_a)
    embedded_b = tf.nn.embedding_lookup(embedding_matrix, analogy_b)
    embedded_c = tf.nn.embedding_lookup(embedding_matrix, analogy_c)

    # Predicted embedding vectors (i.e. computed answer to the analogy question)
    # (dim = [N, embedding dimension])
    pred_d = embedded_c + (embedded_b - embedded_a)

    # Distance between each pair [pred_d, word in vocabulary]
    # (dim = [N (rows), vocabulary size (cols)])
    dist_pred_vocab = tf.matmul(pred_d, embedding_matrix, transpose_b=True)

    # Find the top words closest to the computed result (pred_d)
    n_top = 5

    # For each question in the processing_batch (i.e., each row in dist_pred_vocab)
    # Find the words with the lowest distance to pred_d
    # pred_dist holds values of distances (dim = [N, k])
    # pred_idx holds indices in dist_pred_vocab, which corresponds to the indices of words in the vocab (dim = [N, k])
    pred_dist, pred_idx = tf.nn.top_k(dist_pred_vocab, n_top)  # cosine similarity is in [-1, 1]

    # Evaluate analogies
    scores = list()
    correct_answers_q = list()
    correct_answers_a = list()
    incorrect_answers_q = list()
    incorrect_answers_a = list()

    def inner_evaluate_analogies(sess):

        # Loop over the different types of analogies
        for i in range(len(analogies)):

            print('Evaluating analogies of type {:<80} ({:>2} of {:>2})'.format(analogy_types[i], i, len(analogies)))

            # Helper variables
            correct_answers = 0  # accumulate the number of correct answers in this analogy type
            n_questions = len(analogies[i])  # number of questions to evaluate in this analogy type

            # Compute answers
            sub = analogies[i]  # subset of "analogies": the questions corresponding to this analogy type

            dist, idx = sess.run([pred_dist, pred_idx], {
                analogy_a: sub[:, 0],
                analogy_b: sub[:, 1],
                analogy_c: sub[:, 2]
            })

            # Compute score
            correct_answers_q_ = list()  # list of correctly answered questions
            # (temporary array, elements are np array of dim [1, 4])
            correct_answers_a_ = list()  # list of top 5 answers to correctly answered questions
            # (temporary array, elements are np array of dim [1, 4])
            incorrect_answers_q_ = list()  #
            incorrect_answers_a_ = list()  #

            # loop over the questions (sub.shape[0] == n_questions)
            for q in range(sub.shape[0]):

                # boolean
                answered_correctly = False

                # loop over top closest words
                for j in range(n_top):

                    # if the correct answer is among the n_top closest words
                    if idx[q, j] == sub[q, 3]:
                        answered_correctly = True
                        break
                    elif idx[q, j] in sub[q, :3]:
                        continue    # skip words which are already in the question
                    else:
                        continue    # not the correct answer

                # Do scoring
                if answered_correctly:
                    correct_answers += 1
                    correct_answers_q_.append(sub[q, :])  # add the question to the correctly answered questions
                    correct_answers_a_.append(idx[q, :])  # add the top closest words to the correct answers
                else:
                    incorrect_answers_q_.append(sub[q, :])
                    incorrect_answers_a_.append(idx[q, :])

            scores.append([correct_answers, n_questions])
            correct_answers_q.append(correct_answers_q_)
            correct_answers_a.append(correct_answers_a_)
            incorrect_answers_q.append(incorrect_answers_q_)
            incorrect_answers_a.append(incorrect_answers_a_)

    if session is None:
        with tf.Session() as sess:
            inner_evaluate_analogies(sess)
    else:
        inner_evaluate_analogies(session)

    # Print score to file
    print('\n\tPrinting evaluation scores to file ', results_filename)
    w = '120'  # column width in printout
    line_q = '\n{:<' + w + '}\n{:<' + w + '}\n{:<' + w + '}\nexpected answer:\n\t\t{:<' + w + '}\n'
    line_a = '\t\t{:<' + w + '}\n'
    line_a *= n_top
    line_a = 'Nearest neighbors:\n' + line_a
    with open(results_filename, 'w') as f:

        # Write file header
        f.write('-' * int(w) + '\n' + 'Score summary' + '\n' + '-' * int(w) + '\n')
        f.write('{:<85}\t{:<18}\t{:<12}\t{:<20}\n'.format(
            'Analogy type',
            '#correct answers', '#questions', 'correct answers [%]'))

        # Summary
        # Loop over analogy types
        for i in range(len(analogies)):
            # Get scores
            anal_type_ = analogy_types[i]
            score_ = scores[i][0]
            n_q_ = scores[i][1]
            perc_ = score_ * 100.0 / n_q_

            # Write the line containing all information pertaining to this analogy type
            f.write('{:<85}\t{:>18}\t{:>12}\t{:>20}\n'.format(
                anal_type_, score_, n_q_, perc_))

        # Write file header
        f.write('\n\n')

        # Detailed score
        # Loop over analogy types
        for i in range(len(analogies)):
            # Write the line containing all information pertaining to this analogy type
            f.write('-' * int(w) + '\n' + analogy_types[i] + '\n' + '-' * int(w))

            f.write('\n--- Correct predictions:')
            if len(correct_answers_q[i]) > 0:
                f.write('\n')
                for q in range(len(correct_answers_q[i])):
                    f.write(line_q.format(reverse_dictionary[correct_answers_q[i][q][0]],
                                          reverse_dictionary[correct_answers_q[i][q][1]],
                                          reverse_dictionary[correct_answers_q[i][q][2]],
                                          reverse_dictionary[correct_answers_q[i][q][3]]))
                    f.write(line_a.format(reverse_dictionary[correct_answers_a[i][q][0]],
                                          reverse_dictionary[correct_answers_a[i][q][1]],
                                          reverse_dictionary[correct_answers_a[i][q][2]],
                                          reverse_dictionary[correct_answers_a[i][q][3]],
                                          reverse_dictionary[correct_answers_a[i][q][4]]))
            else:
                f.write('None')

            f.write('\n--- Incorrect predictions:')
            if len(incorrect_answers_q[i]) > 0:
                f.write('\n')
                for q in range(len(incorrect_answers_q[i])):
                    f.write(line_q.format(reverse_dictionary[incorrect_answers_q[i][q][0]],
                                          reverse_dictionary[incorrect_answers_q[i][q][1]],
                                          reverse_dictionary[incorrect_answers_q[i][q][2]],
                                          reverse_dictionary[incorrect_answers_q[i][q][3]]))
                    f.write(line_a.format(reverse_dictionary[incorrect_answers_a[i][q][0]],
                                          reverse_dictionary[incorrect_answers_a[i][q][1]],
                                          reverse_dictionary[incorrect_answers_a[i][q][2]],
                                          reverse_dictionary[incorrect_answers_a[i][q][3]],
                                          reverse_dictionary[incorrect_answers_a[i][q][4]]))
            else:
                f.write('None')

    # Total number of correct answers
    return [(s[0], s[1]) for s in scores]


def analogies(eval_folder, embeddings, embeddings_file, dictionary, reverse_dictionary):
    """
    Evaluate embeddings with respect to analogies
    :param eval_folder: folder in which to write analogy results
    :param embeddings: embedding matrix to evaluate
    :param embeddings_file: file in which the embedding matrix is stored
    :param dictionary: [keys=statement, values==statement index]
    :param reverse_dictionary: [keys=statement index, values=statement]
    """
    # Create folder in which to write analogy results
    folder_analogies = os.path.join(eval_folder, "analogy")
    if not os.path.exists(folder_analogies):
        os.makedirs(folder_analogies)

    # Generate analogy "questions" and write them to a file
    analogy_questions_file = os.path.join(folder_analogies, "questions.txt")
    if not os.path.exists(analogy_questions_file):
        print('\n--- Generate analogy questions and write them to a file')
        analogygen.generate_analogy_questions(analogy_questions_file)

    # Load analogies
    analogy_questions_file_dump = os.path.join(folder_analogies, "questions")
    if not os.path.exists(analogy_questions_file_dump):

        # Read analogies from external file
        print('\n--- Read analogies from file ', analogy_questions_file)
        analogies, analogy_types, n_questions_total, n_questions_relevant = \
            load_analogy_questions(analogy_questions_file, dictionary)

        # Dump analogies into a file to be reused
        print('\n--- Writing analogies into file ', analogy_questions_file_dump)
        i2v_utils.safe_pickle([analogies, analogy_types, n_questions_total, n_questions_relevant],
                              analogy_questions_file_dump)

    else:

        # Load analogies from binary file
        print('\n--- Loading analogies from file ', analogy_questions_file_dump)
        with open(analogy_questions_file_dump, 'rb') as f:
            analogies, analogy_types, n_questions_total, n_questions_relevant = pickle.load(f)

    # Print info
    print('\tFound    {:>10,d} analogy-questions in total, '.format(n_questions_total))
    print('\tof which {:>10,d} are compatible with this vocabulary'.format(n_questions_relevant))

    # Evaluate
    summary = ''
    score_list = list()

    # Evaluate analogies in the embedding space
    analogy_eval_file = os.path.join(folder_analogies, 'res_' + embeddings_file[:-2].replace('/', '_') + '.txt')
    print('\n--- Starting analogy evaluation')

    # List of pairs (number of correctly answered questions in category, number of questions in category)
    scores = evaluate_analogies(embeddings, reverse_dictionary, analogies, analogy_types, analogy_eval_file)
    score_list.append(scores)
    summary += write_score_summary(scores, analogy_types, embeddings_file)

    # Print summary
    print(summary)


########################################################################################################################
# Semantic tests: helper variables and functions
########################################################################################################################
semantic_categories = {
    'add': ['<%ID> = add'],
    'fadd': ['<%ID> = fadd'],
    'sub': ['<%ID> = sub'],
    'fsub': ['<%ID> = fsub'],
    'mul': ['<%ID> = mul'],
    'fmul': ['<%ID> = fmul'],
    'ret': ['ret '],
    'fdiv': ['<%ID> = fdiv'],
    'udiv': ['<%ID> = udiv'],
    'sdiv': ['<%ID> = sdiv'],
    'bitwise binary': ['<%ID> = (shl|lshr|ashr|and|or|xor)']
}


def lookup(stmt, embeddings, dictionary):
    """
    Look up a statement's embedded vector in the embedding matrix
    :param stmt: statement to be looked up (string)
    :param embeddings: embedding matrix
    :param dictionary: [keys=statement, values==statement index]
    :return: embedded vector
    """
    return embeddings[dictionary[stmt], :]


def dist(a, b, W, dic):
    """
    Return the distance between the embedded vectors of two statements
    :param a: statement (string)
    :param b: statement (string)
    :return: distance
    """
    return np.dot(lookup(a, W, dic), lookup(b, W, dic))


def test_distances(category, dictionary, W):
    """
    Perform a distance test for a certain category of statements
    :param category: key in dictionary 'semantic_categories'
    :param dictionary: [keys=statement, values==statement index]
    :param W: embedding matrix
    :return: score, summary string to print
    """
    # Create in and out statements
    in_category = list()
    out_category = list()
    matches = semantic_categories[category]
    for s in dictionary.keys():
        for m in matches:
            added = False
            if re.match(m, s):
                in_category.append(dictionary[s])
                added = True
                break
        if not added:
            out_category.append(dictionary[s])

    # Print
    out_ = 'Statements in category    : {:>6,d}\n'.format(len(in_category))
    out_ += 'Statements out of category: {:>6,d}\n'.format(len(out_category))

    # Loop and test
    res = list()
    for i, in1 in enumerate(in_category):
        if i % 5 == 0:
            out_ += 'processed ' + str(i) + ' outer stmts\n'
        for in2 in in_category:
            if in1 != in2:
                for out in out_category:
                    # d(fadd, br) > d(br, invoke)
                    # d(f*, br) > d(invoke, br)
                    res.append(np.dot(W[out, :], W[in1, :]) < np.dot(W[in2, :], W[in1, :]))

    return res, out_


def semantic_test(eval_folder, embeddings, embeddings_file, dictionary):
    """
    Evaluate embeddings with semantic tests
    :param eval_folder: folder in which to write analogy results
    :param embeddings: embedding matrix to evaluate
    :param embeddings_file: file in which the embedding matrix is stored
    :param dictionary: [keys=statement, values==statement index]
    """
    print('\n--- Starting semantic tests')
    # Carry out tests:
    out = ''
    for category in semantic_categories.keys():
        out_ = 'Testing category \"' + category + '\" ...'
        print(out_)
        out += out_
        res, out_ = test_distances(category, dictionary, embeddings)
        print(out_)
        out += out_
        if len(res) > 0:
            out_ = 'Score: {:3}%'.format(res.count(True)*100/len(res))
        else:
            out_ = 'Empty category'
        print(out_)
        out += out_

    # Create folder in which to write semantic test results
    folder_semtests = os.path.join(eval_folder, "semtests")
    if not os.path.exists(folder_semtests):
        os.makedirs(folder_semtests)

    # Print results to file
    res_file = os.path.join(folder_semtests, 'res_' + embeddings_file[:-2].replace('/', '_') + '.txt')
    with open(res_file, 'w') as f:
        f.write(out)


########################################################################################################################
# Clustering plot: helper functions
########################################################################################################################
def get_stmt_tag(stmt):
    for fam in rgx.llvm_IR_stmt_families:
        if re.match(fam[3], stmt, re.MULTILINE):
            return fam[0]
    return None


def get_stmt_newtag(stmt):
    for fam in rgx.llvm_IR_stmt_tags:
        if re.match(fam[0], stmt, re.MULTILINE):
            return fam[2]
    return None


def create_tags(rev_dic, data):
    if FLAGS.newtags:
        targets = list(set([tag[2] for tag in rgx.llvm_IR_stmt_tags]))
        print('Found %d labels' % len(targets))
        labels = [get_stmt_newtag(rev_dic[stmt]) for stmt in range(len(data))]
    else:
        targets = list(set([tag[0] for tag in rgx.llvm_IR_stmt_families]))
        print('Found %d labels' % len(targets))
        labels = [get_stmt_tag(rev_dic[stmt]) for stmt in range(len(data))]

    return targets, labels


def plot_clustering(eval_folder, embeddings, embeddings_file, reverse_dictionary):
    """
    Evaluate embeddings by visualizing the clustering plot
    :param eval_folder: folder in which to write analogy results
    :param embeddings: embedding matrix to evaluate
    :param embeddings_file: file in which the embedding matrix is stored
    :param reverse_dictionary: [keys=statement index, values=statement]
    """
    print('\n--- Starting clustering plot')

    # Create folder in which to save plots
    folder_clusterplot = os.path.join(eval_folder, "clusterplot")
    if not os.path.exists(folder_clusterplot):
        os.makedirs(folder_clusterplot)

    if FLAGS.taglist is None:
        print('Taglist must be defined')
        return 1

    print('Loading/creating labels')
    if FLAGS.newtags:
        flags_file = FLAGS.taglist + '.new'
    else:
        flags_file = FLAGS.taglist
    if os.path.exists(flags_file):
        print('Loaded from tags file', flags_file)
        [targets, labels] = pickle.load(open(flags_file, 'rb'))
    else:
        print('Recomputing tags file')
        targets, labels = create_tags(reverse_dictionary, embeddings)
        pickle.dump([targets, labels], open(flags_file, 'wb'))

    if FLAGS.tsne:
        embedding = TSNE(metric=FLAGS.metric, verbose=FLAGS.verbose).fit_transform(embeddings)
        np_file = os.path.join(folder_clusterplot, 'tsne_' + embeddings_file[:-2].replace('/', '_') + '.np')
        html_file = os.path.join(folder_clusterplot, 'tsne_' + embeddings_file[:-2].replace('/', '_') + '.html')
    else:
        embedding = umap.UMAP(metric=FLAGS.metric, verbose=FLAGS.verbose).fit_transform(embeddings)
        np_file = os.path.join(folder_clusterplot, 'umap_' + embeddings_file[:-2].replace('/', '_') + '.np')
        html_file = os.path.join(folder_clusterplot, 'umap_' + embeddings_file[:-2].replace('/', '_') + '.html')

    # Save plots to file
    embedding.tofile(np_file)
    output_file(html_file)
    print('Plotting')

    source = ColumnDataSource(dict(
            x=[e[0] for e in embedding],
            y=[e[1] for e in embedding],
            label=labels))

    cmap = CategoricalColorMapper(factors=targets, palette=Category20[len(targets)])

    p = figure(title="test umap")
    p.circle(x='x',
             y='y',
             source=source,
             color={"field": 'label', "transform": cmap},
             legend='label')
    show(p)


########################################################################################################################
# Main function for embeddings evaluation
########################################################################################################################
def evaluate_embeddings(data_folder, embeddings, embeddings_file):
    """
    Main function for embeddings evaluation
    :param data_folder: string containing the path to the parent directory of raw data sub-folders
    :param embeddings: embedding matrix to evaluate
    :param embeddings_file: file in which the embedding matrix is stored

    Folders produced:
        data_folder/FLAGS.embeddings_folder/eval/analogy
        data_folder/FLAGS.embeddings_folder/eval/clusterplot
        data_folder/FLAGS.embeddings_folder/eval/semtests
    """
    ####################################################################################################################
    # Setup
    # Create evaluation folder
    outfolder = FLAGS.embeddings_folder
    eval_folder = os.path.join(outfolder, "eval")
    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder)

    # Load vocabulary
    print('\n--- Loading vocabulary')
    if FLAGS.vocabulary_folder != '':
        folder_vocabulary = FLAGS.vocabulary_folder
    else:
        folder_vocabulary = os.path.join(data_folder, 'vocabulary')
    dictionary_pickle = os.path.join(folder_vocabulary, 'dic_pickle')
    print('\tLoading dictionary from file', dictionary_pickle)
    with open(dictionary_pickle, 'rb') as f:
        dictionary = pickle.load(f)
    print('\tBuilding reverse dictionary...')
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    ####################################################################################################################
    # Analogies
    analogies(eval_folder, embeddings, embeddings_file, dictionary, reverse_dictionary)

    ####################################################################################################################
    # Semantic tests
    semantic_test(eval_folder, embeddings, embeddings_file, dictionary)

    ####################################################################################################################
    # Clustering plot
    plot_clustering(eval_folder, embeddings, embeddings_file, reverse_dictionary)
