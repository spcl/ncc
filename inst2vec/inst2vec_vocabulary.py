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
"""Construct vocabulary from XFGs and indexify the data set"""


from inst2vec import inst2vec_utils as i2v_utils
import rgx_utils as rgx
import collections
import pickle
import csv
import os
import sys
import math
import struct
import re
import networkx as nx
from scipy import sparse
import random
from absl import flags

FLAGS = flags.FLAGS


########################################################################################################################
# Counting and statistics
########################################################################################################################
def vocabulary_statistics(vocabulary_dic, descr):
    """
    Print some statistics on the vocabulary
    :param vocabulary_dic: dictionary [key=statement in vocabulary , value=number of occurences in data set]
    :param descr: step destriction (string)
    """
    # Get number of lines and the vocabulary size
    number_lines = sum(vocabulary_dic.values())
    vocabulary_size = len(vocabulary_dic.keys())

    # Construct output
    out = '\tAfter ' + descr + ':\n' \
        + '\t--- {:<26}: {:>8,d}\n'.format('Number of lines', number_lines) \
        + '\t--- {:<26}: {:>8,d}\n'.format('Vocabulary size', vocabulary_size)
    print(out)


########################################################################################################################
# Reading, writing and dumping files
########################################################################################################################
def get_file_names(folder):
    """
    Get the names of the individual LLVM IR files in a folder
    :param folder: name of the folder in which the data files to be read are located
    :return: a list of strings representing the file names
    """
    print('Reading file names from all files in folder ', folder)

    # Helper variables
    file_names = dict()
    file_count = 0
    listing = os.listdir(folder + '/')
    to_subtract = file_count

    # Loop over files in folder
    for file in listing:
        if file[0] != '.' and file[-3:] == '.ll':

            # If this isn't a hidden file and it is an LLVM IR file ('.ll' extension),
            # Add file name to dictionary
            file_names[file_count] = file

            # Increment counters
            file_count += 1

    print('    Number of files read from', folder, ': ', file_count - to_subtract)

    return file_names


def print_vocabulary(mylist_freq, filename):
    """
    Prints vocabulary and statistics related to it to a file
    :param mylist_freq: dictionary [stmt, number of occurences]
    :param filename: name of file in which to print
    """
    # Print vocabulary in alphabetical order with number of occurences
    print('Printing vocabulary information to file', filename)
    with open(filename + '_freq.txt', 'w') as f:
        f.write('{:>6}   {}\n'.format('# occ', 'statement (in alphabetical order)'))
        for key, value in sorted(mylist_freq.items()):
            f.write('{:>6}   {}\n'.format(str(value), key))

    # Prepare to print statistics
    mylist_families_l1 = rgx.get_list_tag_level_1()
    to_iterate1 = list()
    for i in range(len(mylist_families_l1)):
        to_iterate1.append([mylist_families_l1[i], rgx.get_count(mylist_freq, mylist_families_l1[i], 1)])
    to_iterate1.sort(key=lambda tup: tup[1], reverse=True)

    # Print statistics
    with open(filename + '_class.txt', 'w') as f:
        f.write('{:>6}   {:<30}{:<25}{}\n'.format('# occ', 'tag level 1', 'tag level 2', 'tag level 3'))

        # Print all level 1
        for tag1 in to_iterate1:
            f.write('{:>6}   {:<30}\n'.format(str(tag1[1]), tag1[0]))

            # Get stats l2
            mylist_families_l2 = rgx.get_list_tag_level_2(tag1[0])
            to_iterate2 = list()
            for i in range(len(mylist_families_l2)):
                to_iterate2.append([mylist_families_l2[i], rgx.get_count(mylist_freq, mylist_families_l2[i], 2)])
            to_iterate2.sort(key=lambda tup: tup[1], reverse=True)

            # Print all level 2
            for tag2 in to_iterate2:
                f.write('{:>6}   {:<30}{:<25}\n'.format(str(tag2[1]), '----------------------------', tag2[0]))

                # Get stats l3
                mylist_families_l3 = rgx.get_list_tag_level_3(tag2[0])
                to_iterate3 = list()
                for i in range(len(mylist_families_l3)):
                    to_iterate3.append([mylist_families_l3[i], rgx.get_count(mylist_freq, mylist_families_l3[i], 3)])
                to_iterate3.sort(key=lambda tup: tup[1], reverse=True)

                # Print all level 3
                for tag3 in to_iterate3:
                    f.write('{:>6}   {:<30}{:<25}{}\n'.format(str(tag3[1]),
                                                              '----------------------------',
                                                              '-----------------------', tag3[0]))


def make_one_line_stmt(stmt):
    """
    Some statements contain carriage returns.
    Yet they should be printed into only one line in the vocabulary metdata file to be read by Tensorboard.
    Apply this function to transform them
    :param stmt: a string containing carriage returns
    :return: modified statement
    """
    stmt = re.sub('\n', ' \ ', stmt)
    return stmt


def print_vocabulary_metadata(reverse_dictionary, source_data_freq, filename):
    """
    Print the vocabulary's metadata into a tab-separated value file
    to be loaded by Tensorboard
    :param reverse_dictionary: dictionary [key=index, value=statement]
    :param source_data_freq: dictionary [key=statement, value=number of occurences]
    :param filename: file name
    """
    to_track = ''

    print('Printing metadata information to file ', filename)
    with open(filename, 'w') as f:
        # Write file header
        f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('stmt', 'count', 'tag1', 'tag2', 'tag3', 'newtagA', 'newtagB'))

        # Loop over all words in dictionary
        vocabulary_size = len(reverse_dictionary)
        for i in range(vocabulary_size):

            if i % 100 == 0:
                print('Processed {:>6,d} words out of {:>6,d} ...'.format(i, vocabulary_size))

            word = reverse_dictionary[i]
            count = source_data_freq[word]

            # Debugging
            if len(to_track) > 0:
                if word == to_track:
                    print('Found stmt', to_track)

            # Find tags corresponding to this word in llvm_IR_stmt_families
            for fam in rgx.llvm_IR_stmt_families:
                if re.match(fam[3], word, re.MULTILINE):
                    t1 = fam[0]
                    t2 = fam[1]
                    t3 = fam[3]
                    break
            else:
                assert False, "No OLD tag found for stmt " + word

            # Find tags corresponding to this word in llvm_IR_stmt_tags
            for t in rgx.llvm_IR_stmt_tags:
                if re.match(t[0], word, re.MULTILINE):
                    tnA = t[1]
                    tnB = t[2]
                    break
            else:
                assert False, "No NEW tag found for stmt \"" + word + "\""

            # Write the line containing all information pertaining to this word
            if '\n' in word:
                word = make_one_line_stmt(word)
            f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(word, count, t1, t2, t3, tnA, tnB))


########################################################################################################################
# Helper functions for vocabulary construction
########################################################################################################################
def add_to_vocabulary(cumul_dic, stmt_list):
    """

    :param cumul_dic:
    :param stmt_list:
    :return:
    """
    count_stmts = collections.Counter(stmt_list)

    for s in count_stmts.keys():
        if s in cumul_dic.keys():
            cumul_dic[s] += count_stmts[s]
        else:
            cumul_dic[s] = count_stmts[s]

    return cumul_dic


def prune_vocabulary(data, cutoff):
    """
    Prune the all the words which appear less than cutoff times in the data from the vocabulary
    :param data: dictionary [statement, frequency]
    :param cutoff: prune any stmt which appears less than cutoff times in "source_data_list"
                   if = 0, do no pruning
    :return:
    """
    stmts_cut_off = list()
    if cutoff > 0:
        print("Start pruning vocabulary with cutoff value", cutoff, "...")

        # Add 'unknown' entry to vocabulary dictionary
        data[rgx.unknown_token] = 0

        i = 0
        n_data = len(data)
        for s in data.keys():
            # Print ever so often to mark progress
            if i % 1e5 == 0:
                print('statement {:>12,d} of {:>12,d} ...'.format(i, n_data))
            if data[s] <= cutoff:
                stmts_cut_off.append(s)
                data[rgx.unknown_token] += data[s]  # add to number of unknowns
                data[s] = 0
            i += 1

        # Create new dictionary without all the entries which were set to 0
        data = {k: data[k] for k in data if data[k] > 0}

    else:
        print("Cutoff is null, skip pruning ...")

    return data, stmts_cut_off


def build_dictionary(words):
    """
    Process raw inputs into a data set
    :param words: list of strings where each element is a word/token
    :return: dictionary
    data        -- list of indices corresponding to the input words
    count       -- list of length n_words containing the most common words
                   every element is a tuple ('word', index in dictionary)
    dictionary  -- dictionary where dictionary[word] = index in data
    reversed_dictionary -- reversed_dictionary[index] = word
    """
    # Create a dictionary with an entry for each of the possible words
    print('Create dictionary of statement indices ...')
    dictionary = dict()
    for word in words.keys():
        dictionary[word] = len(dictionary)

    return dictionary


########################################################################################################################
# Helper functions for pair construction
########################################################################################################################
def build_H_dictionary(D, skip_window, folder, filename, dictionary, stmts_cut_off):
    """
    Build H-dictionary [keys=indexed data pairs, values=number of occurences in file] from dual-XFG and list of cut off
    statements and write them to a file
    :param D: Dual graph
    :param skip_window: context window width
    :param folder: folder in which to write adjacency-matrix files
    :param filename: base filename
    :param dictionary: [keys=statements, values=index]
    :param stmts_cut_off: list of statements cut off in the pruning step
    :return: H_dic: [keys=(index of target statement, index of context-statement), values=number of occurences in files]
    """
    # Create index-node dictionary
    nodelist = list(D.nodes())

    # Treat as a big matrix
    if len(nodelist) > 15e3:
        graph_is_big = True
    else:
        graph_is_big = False

    if graph_is_big:
        print('got node list, length=', len(nodelist))

    # Get adjacency matrix level 1
    if graph_is_big:
        adj_mat_file = os.path.join(folder, filename + '_AdjMat' +'.npz')
        if os.path.exists(adj_mat_file):
            print('Load adjmat from', adj_mat_file)
            A1 = sparse.load_npz(adj_mat_file)
        else:
            A1 = nx.adjacency_matrix(D)
            if sys.getsizeof(A1) < 45e5:
                print('Save adjmat to', adj_mat_file)
                sparse.save_npz(adj_mat_file, A1)
    else:
        A1 = nx.adjacency_matrix(D)

    # Context adjacency
    A = A1
    A_context = A1

    if skip_window > 1:

        # Compute context-adjacency
        for i in range(skip_window-1):
            if graph_is_big:
                A_file = os.path.join(folder, filename + '_A' + str(i) + '.npz')
                if os.path.exists(A_file):
                    print('Load A mat from', A_file)
                    A = sparse.load_npz(A_file)
                else:
                    A *= A1
                    if sys.getsizeof(A1) < 45e5:
                        print('Saving A mat to', A_file)
                        sparse.save_npz(A_file, A)
            else:
                A *= A1

            A_context += A

            if graph_is_big:
                print('completed step', i)
        del A1, A

    # if context_width = 1, then A_context is simply A1

    A_num_rows = A_context.shape[0]
    A_indices = A_context.indices
    A_row_starts = A_context.indptr
    H_dic = dict()

    # Loop over rows
    for i in range(A_num_rows):

        if graph_is_big and i % 5e3 == 0:
                print('Adjmat rows:', i, '/', A_num_rows)

        col_indices = A_indices[A_row_starts[i]:A_row_starts[i+1]]
        for j in col_indices:

            if i != j:

                # Add nodes
                target = re.sub('§\d+$', '', nodelist[i])
                context = re.sub('§\d+$', '', nodelist[j])

                if target != context:

                    # cut off?
                    if target in stmts_cut_off:
                        target = rgx.unknown_token
                        if context in stmts_cut_off:
                            break  # we don't want a pair (UNK, UNK)
                    if context in stmts_cut_off:
                        context = rgx.unknown_token

                    # target-context index pair
                    if target not in dictionary.keys() or context not in dictionary.keys():
                        if target not in dictionary.keys():
                            print('WARNING, not in dictionary:', target)
                        if context not in dictionary.keys():
                            print('WARNING, not in dictionary:', context)
                    else:
                        t = dictionary[target]
                        c = dictionary[context]
                        if (t, c) in H_dic:
                            H_dic[(t, c)] += 1
                        else:
                            H_dic[(t, c)] = 1

    return H_dic


def generate_data_pairs_from_H_dictionary(H_dic, t):
    """
    Generate data pairs from H-dictionary by reading them from a file and applying subsampling
    :param H_dic: [keys=(index of target statement, index of context-statement), values=number of occurences in files]
    :param t: subsampling threshold
    :return: data_pairs: subsampled data pairs in a list
    """
    data_pairs = list()
    file = 0

    # Loop over the "in-context" graphs
    print('Generating data pairs from dic dump with subsampling threshold', t)

    n_possible_pairs = sum(list(H_dic.values()))
    var = t * n_possible_pairs

    for ct, rep in H_dic.items():

        # Construct counter and discard probability
        if t > 0:
            p_discard = max(1.0 - math.sqrt(var / rep), 0)
        else:
            p_discard = 0

        c = ct[0]
        t = ct[1]
        for i in range(rep):
            # Add this edge's nodes to the list as a data pair if it is not discarded by susampling
            if random.random() > p_discard:
                data_pairs.append([c, t])
            if random.random() > p_discard:
                data_pairs.append([t, c])

    # Increment counter
    file += 1

    # Return
    print('\nNumber of generated data pairs in file  : {:>12,d}'.format(len(data_pairs)))
    return data_pairs


########################################################################################################################
# Main function for vocabulary construction
########################################################################################################################
def construct_vocabulary(data_folder, folders):
    """
    Construct vocabulary from XFGs and indexify the data set
    :param data_folder: string containing the path to the parent directory of data sub-folders
    :param folders: list of sub-folders containing pre-processed LLVM IR code

    Files produced for vocabulary:
        data_folder/vocabulary/cutoff_stmts_pickle
        data_folder/vocabulary/cutoff_stmts.csv
        data_folder/vocabulary/dic_pickle
        data_folder/vocabulary/dic.csv
        data_folder/vocabulary/vocabulary_metadata_for_tboard
        data_folder/vocabulary/vocabulary_statistics_class.txt
        data_folder/vocabulary/vocabulary_statistics_freq.txt
    Files produced for pair-building:
        data_folder/*_datasetprep_adjmat/
        data_folder/*_datasetprep_cw_X/file_H_dic_cw_X.p
    Files produced for indexification:
        data_folder/*_dataset_cw_X/data_pairs_cw_3.rec
    """

    # Get options and flags
    context_width = FLAGS.context_width
    cutoff_unknown = FLAGS.cutoff_unknown
    subsample_threshold = FLAGS.subsampling

    # Vocabulary folder
    folder_vocabulary = os.path.join(data_folder, 'vocabulary')
    if not os.path.exists(folder_vocabulary):
        os.makedirs(folder_vocabulary)

    ####################################################################################################################
    # Build vocabulary
    dictionary_csv = os.path.join(folder_vocabulary, 'dic.csv')
    dictionary_pickle = os.path.join(folder_vocabulary, 'dic_pickle')
    cutoff_stmts_pickle = os.path.join(folder_vocabulary, 'cutoff_stmts_pickle')
    if not os.path.exists(dictionary_csv):

        # Combine the source data lists
        print('\n--- Combining', len(folders), 'folders into one data set from which we build a vocabulary')
        source_data_list_combined = dict()  # keys: statements as strings, values: number of occurences
        num_statements_total = 0

        for folder in folders:

            folder_preprocessed = folder + '_preprocessed'
            transformed_folder = os.path.join(folder_preprocessed, 'data_transformed')
            file_names_dict = get_file_names(folder)
            file_names = file_names_dict.values()
            num_files = len(file_names)
            count = 0

            for file_name in file_names:

                source = os.path.join(transformed_folder, file_name[:-3] + '.p')

                if os.path.exists(source):
                    with open(source, 'rb') as f:

                        # Load lists of statements
                        print('Fetching statements from file {:<60} ({:>2} / {:>2})'.format(
                            source, count, num_files))
                        source_data_list_ = pickle.load(f)

                        # Add to cummulated list
                        source_data_list_combined = add_to_vocabulary(source_data_list_combined, source_data_list_)

                        # Get numbers
                        num_statements_in_file = len(source_data_list_)
                        num_statements_total += num_statements_in_file
                        print('\tRead        {:>10,d} statements in this file'.format(num_statements_in_file))
                        print('\tAccumulated {:>10,d} statements so far'.format(num_statements_total))
                        del source_data_list_
                        count += 1

        # Get statistics of the combined list before pruning
        print('\n--- Compute some statistics on the combined data')
        vocabulary_statistics(source_data_list_combined, descr="combining data folders")

        # Prune data
        source_data_list_combined, stmts_cut_off = prune_vocabulary(source_data_list_combined, cutoff_unknown)

        # Get statistics of the combined list after pruning
        print('\n--- Compute some statistics on the combined data')
        vocabulary_statistics(source_data_list_combined, descr="pruning combined data")

        # Build the vocabulary
        print('\n--- Building the vocabulary and indices')

        # Set the vocabulary size
        vocabulary_size = len(source_data_list_combined)

        # Build data set: use ordering from original files, here statement-strings are being translated to indices
        number_statements = sum(list(source_data_list_combined.values()))
        dictionary = build_dictionary(source_data_list_combined)

        # Print information about the vocabulary to console
        out = '\tAfter building indexed vocabulary:\n' \
              + '\t--- {:<26}: {:>8,d}\n'.format('Number of stmts', number_statements) \
              + '\t--- {:<26}: {:>8,d}\n'.format('Vocabulary size', vocabulary_size)
        print(out)

        # Print information about the vocabulary to file
        vocab_info_file = os.path.join(folder_vocabulary, 'vocabulary_statistics')
        print_vocabulary(source_data_list_combined, vocab_info_file)

        # Print dictionary
        print('Writing dictionary to file', dictionary_pickle)
        i2v_utils.safe_pickle(dictionary, dictionary_pickle)
        print('Writing dictionary to file', dictionary_csv)
        with open(dictionary_csv, 'w', newline='') as f:
            fieldnames = ['#statement', 'index']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            data = [dict(zip(fieldnames, [k.replace('\n ', '\\n '), v])) for k, v in dictionary.items()]
            writer.writerows(data)

        # Print cut off statements
        print('Writing cut off statements to file', cutoff_stmts_pickle)
        i2v_utils.safe_pickle(stmts_cut_off, cutoff_stmts_pickle)
        cutoff_stmts_csv = os.path.join(folder_vocabulary, 'cutoff_stmts.csv')
        print('Writing cut off statements to file', cutoff_stmts_csv)
        with open(cutoff_stmts_csv, 'w', newline='\n') as f:
            for c in stmts_cut_off:
                f.write(c + '\n')
        del cutoff_stmts_csv

        # Print metadata file used by TensorBoard
        print('Building reverse dictionary...')
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        vocab_metada_file = os.path.join(folder_vocabulary, 'vocabulary_metadata_for_tboard')
        print_vocabulary_metadata(reverse_dictionary, source_data_list_combined, vocab_metada_file)

        # Let go of variables that aren't needed anymore so as to reduce memory usage
        del source_data_list_combined

    ####################################################################################################################
    # Generate data-pair dictionaries

    # Load dictionary and cutoff statements
    print('\n--- Loading dictionary from file', dictionary_pickle)
    with open(dictionary_pickle, 'rb') as f:
        dictionary = pickle.load(f)
    print('Loading cut off statements from file', cutoff_stmts_pickle)
    with open(cutoff_stmts_pickle, 'rb') as f:
        stmts_cut_off = pickle.load(f)
    stmts_cut_off = set(stmts_cut_off)

    # Generate
    print('\n--- Generating data pair dictionary from dual graphs and dump to files')

    for folder in folders:

        folder_preprocessed = folder + '_preprocessed'
        folder_Dfiles = os.path.join(folder_preprocessed, 'xfg_dual')
        D_files_ = os.listdir(folder_Dfiles + '/')
        D_files = [Df for Df in D_files_ if Df[-2:] == '.p']
        num_D_files = len(D_files)
        folder_H = folder + '_datasetprep_cw_' + str(context_width)
        folder_mat = folder + '_datasetprep_adjmat'
        if not os.path.exists(folder_H):
            os.makedirs(folder_H)
        if not os.path.exists(folder_mat):
            os.makedirs(folder_mat)

        for i, D_file in enumerate(D_files):

            # "In-context" dictionary
            base_filename = D_file[:-2]
            D_file_open = os.path.join(folder_Dfiles, D_file)
            to_dump = os.path.join(folder_H, base_filename + "_H_dic_cw_" + str(context_width) + '.p')
            if not os.path.exists(to_dump):

                # Load dual graph
                print('Build H_dic from:', D_file_open, '(', i, '/', num_D_files, ')')
                with open(D_file_open, 'rb') as f:
                    D = pickle.load(f)

                # Build H-dictionary
                H_dic = build_H_dictionary(D, context_width, folder_mat, base_filename, dictionary, stmts_cut_off)
                print('Print to', to_dump)
                i2v_utils.safe_pickle(H_dic, to_dump)

            else:
                print('Found context-dictionary dump:', to_dump, '(', i, '/', num_D_files, ')')

    ####################################################################################################################
    # Generate data_pairs.rec from data pair dictionary dumps

    # Generate
    print('\n--- Writing .rec files')

    for folder in folders:

        # H dic dump files
        folder_H = folder + '_datasetprep_cw_' + str(context_width)
        H_files_ = os.listdir(folder_H + '/')
        H_files = [Hf for Hf in H_files_ if "_H_dic_cw_" + str(context_width) in Hf and Hf[-2:] == '.p']
        num_H_files = len(H_files)

        # Record files
        folder_REC = folder + '_dataset_cw_' + str(context_width)
        file_rec = os.path.join(folder_REC, 'data_pairs_cw_' + str(context_width) + '.rec')
        if not os.path.exists(folder_REC):
            os.makedirs(folder_REC)

        if not os.path.exists(file_rec):

            # Clear contents
            f = open(file_rec, 'wb')
            f.close()

            data_pairs_in_folder = 0
            for i, H_file in enumerate(H_files):

                dic_dump = os.path.join(folder_H, H_file)

                print('Building data pairs from file', dic_dump, '(', i, '/', num_H_files, ')')
                with open(dic_dump, 'rb') as f:
                    H_dic = pickle.load(f)

                # Get pairs [target, context] from graph and write them to file
                data_pairs = generate_data_pairs_from_H_dictionary(H_dic, subsample_threshold)
                data_pairs_in_folder += len(data_pairs)

                print('writing to fixed-length file: ', file_rec)

                # Start read and write
                counter = 0
                with open(file_rec, 'ab') as rec:

                    # Loop over pairs
                    num_pairs = len(data_pairs)
                    for p in data_pairs:

                        # Print progress ever so often
                        if counter % 10e5 == 0 and counter != 0:
                            print('wrote pairs: {:>10,d} / {:>10,d} ...'.format(counter, num_pairs))

                        # Write and increment counter
                        assert int(p[0]) < 184, "Found index " + str(int(p[0]))
                        assert int(p[1]) < 184, "Found index " + str(int(p[1]))
                        rec.write(struct.pack('II', int(p[0]), int(p[1])))
                        counter += 1

            print('Pairs in folder', folder, ':', data_pairs_in_folder)

        else:

            filesize_bytes = os.path.getsize(file_rec)
            # Number of pairs is filesize_bytes / 2 (pairs) / 4 (32-bit integers)
            file_pairs = int(filesize_bytes / 8)
            print('Found', file_rec, 'with #pairs:', file_pairs)
