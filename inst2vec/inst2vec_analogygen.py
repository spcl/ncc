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
"""Helper module for analogy generation"""

from jinja2 import Template

########################################################################################################################
# Helper variables
########################################################################################################################
alignment = ["2", "4", "8", "16", "32"]  # ["1", "2", "4", "8", "16", "32", "64", "128"]


########################################################################################################################
# Helper functions for analogy generation
########################################################################################################################
def write_ready_analogy(list_analogies, description, file):
    """
    Print a ready analogy to a file
    :param list_analogies: list of list of 4 strings
    :param description: string describing the analogy category
    :param file: file to print to
    :return: number of printed analogies
    """
    # Helper variables for printing format
    line = '{}\n{}\n{}\n{}\n\n'

    # Get analogy parameters
    num_ops = len(list_analogies)

    # Print to file
    with open(file, 'a') as f:
        f.write("\n\n# " + description + ": " + str(num_ops) + "\n\n")
        for i in range(num_ops):
            f.write(line.format(list_analogies[i][0], list_analogies[i][1], list_analogies[i][2], list_analogies[i][3]))

    # Return number of generated analogies
    return num_ops


def write_analogy_from_pairs(analogy_pair, description, file):
    """
    Write analogies to file based on pairs of corresponding statements
    :param analogy_pair: list of pairs of strings
    :param description: string describing the analogy category
    :param file: file to print to
    :return: number of printed analogies
    """
    # Helper variables for printing format
    line = '{}\n{}\n{}\n{}\n\n'

    # Get analogy parameters
    analogies = pairs_from_array(analogy_pair)
    num_ops = len(analogies)

    # Print to file
    with open(file, 'a') as f:
        f.write("\n\n# " + description + ": " + str(num_ops) + "\n\n")
        for i in range(num_ops):
            f.write(line.format(analogies[i][0][0], analogies[i][0][1], analogies[i][1][0], analogies[i][1][1]))

    # Return number of generated analogies
    return num_ops


def write_analogy(pars, description, file):
    """
    Write analogies to file
    :param pars: list of elements necessary to generate analogies
    :param description: string describing the analogy category
    :param file: file to print to
    :return: number of printed analogies
    """
    # Helper variables for printing format
    line = '{}\n{}\n{}\n{}\n\n'

    # Get analogy parameters
    var_a, var_b, var_x, var_y, num_ops, template = pars

    # Print to file
    with open(file, 'a') as f:
        f.write("\n\n# " + description + ": " + str(num_ops) + "\n\n")
        for i in range(num_ops):
            f.write(line.format(template.render(var_a[i]), template.render(var_b[i]), template.render(var_x[i]),
                                template.render(var_y[i])))

    # Free memory
    del var_a, var_b, var_x, var_y, template, pars

    # Return number of generated analogies
    return num_ops


def pairs_from_array(a):
    """
    Given an array of strings, create a list of pairs of elements from the array
    Creates all possible combinations without symmetry (given pair [a,b], it does not create [b,a])
    nor repetition (e.g. [a,a])
    :param a: Array of strings
    :return: list of pairs of strings
    """
    pairs = list()
    for i in range(len(a)):
        for j in range(len(a[i+1:])):
            pairs.append([a[i], a[i+1+j]])
    return pairs


########################################################################################################################
# Analogy categories
########################################################################################################################
analogy_categories = {
    # Syntactic type
    'Integer binary operations (type semantic analogy)'                 : 'syntactic - type',
    'Floating point binary operations (type semantic analogy)'          : 'syntactic - type',
    'Floating point / Integer binary operations (type semantic analogy)': 'syntactic - type',
    'Insertelement - Extractelement operations (type)'                  : 'syntactic - type',
    # Syntactic other
    'Insertvalue - Extractvalue operations (index analogy)'             : 'syntactic - other',
    'Insertelement - Extractelement operations (index analogy)'         : 'syntactic - other',
    'Floating point ops (fast-math analogies)'                          : 'syntactic - other',
    # Semantic: inverse
    'Bitcast x to y - y to x (inverse operations analogy)'              : 'semantic - inverse',
    'Arithmetic integer binary operations (inverse operations analogy)' : 'semantic - inverse',
    'Arithmetic flpt binary operations (inverse operations analogy)'    : 'semantic - inverse',
    'Trunc - s/zext (inverse operations analogy)'                       : 'semantic - inverse',
    'Fptou/si - s/uitofp (inverse operations analogy)'                  : 'semantic - inverse',
    'Inttoptr - ptrtoint (inverse operations analogy)'                  : 'semantic - inverse',
    # Semantic: equivalents
    'Structure - Vector equivalents (1)'                                : 'semantic - equivalents',
    'Structure - Vector equivalents (2)'                                : 'semantic - equivalents',
    'Structure - Vector equivalents (3)'                                : 'semantic - equivalents'
}


########################################################################################################################
# Analogy template: Integer binary operations (type semantic analogy)
########################################################################################################################
def gen_anlgy_type_int_bin_op():
    """
    Generate analogy, example:
    <%ID> = add i8 <%ID>, <%ID>
    <%ID> = add i64 <%ID>, <%ID>
    <%ID> = sub i8 <%ID>, <%ID>
    expected answer: <%ID> = sub i64 <%ID>, <%ID>
    """

    # Variables containing variations
    variations_int_t_op_a = list()
    variations_int_t_op_b = list()
    variations_int_t_op_x = list()
    variations_int_t_op_y = list()

    # Counter
    num_int_t_ops = 0

    # Template
    t_int_t_op = Template("<%ID> = {{int_ops}} {{int_op_option}}{{int_type}} {{int_operandA}}, {{int_operandB}}")

    # Expression variations
    int_ops = ["add", "sub", "mul", "udiv", "sdiv", "urem", "srem"]
    int_types = ["i1", "i2", "i4", "i8", "i32", "i64", "i128",
                 "<4 x i8>", "<8 x i8>", "<16 x i8>", "<32 x i8>",
                 "<4 x i16>", "<8 x i16>", "<16 x i16>",
                 "<4 x i32>", "<8 x i32>", "<16 x i32>", "<32 x i32>",
                 "<2 x i64>", "<4 x i64>", "<8 x i64>"]
    int_addsubmul_options = ['', 'nuw ', 'nsw ', 'nuw nsw ']
    int_div_options = ['', 'exact ']
    int_operandsA = ['<INT>', '<%ID>']
    int_operandsB = ['<INT>', '<%ID>']

    # Construct
    for operA in int_operandsA:
        for oper in int_operandsB:
            for p in pairs_from_array(int_ops):
                for v in pairs_from_array(int_types):

                    var0 = list()
                    if p[0] in ["add", "sub", "mul"]:
                        var0 = int_addsubmul_options
                    elif p[0] in ["udiv", "sdiv"]:
                        var0 = int_div_options
                    else:
                        var0 = ['']

                    var1 = list()
                    if p[1] in ["add", "sub", "mul"]:
                        var1 = int_addsubmul_options
                    elif p[1] in ["udiv", "sdiv"]:
                        var1 = int_div_options
                    else:
                        var1 = ['']

                    for v0 in var0:
                        for v1 in var1:

                            # Increase counter
                            num_int_t_ops += 1

                            # Construct variations
                            variations_int_t_op_a.append({"int_ops": p[0], "int_op_option": v0, "int_type": v[0],
                                                          "int_operandA": operA, "int_operandB": oper})
                            variations_int_t_op_b.append({"int_ops": p[0], "int_op_option": v0, "int_type": v[1],
                                                          "int_operandA": operA, "int_operandB": oper})
                            variations_int_t_op_x.append({"int_ops": p[1], "int_op_option": v1, "int_type": v[0],
                                                          "int_operandA": operA, "int_operandB": oper})
                            variations_int_t_op_y.append({"int_ops": p[1], "int_op_option": v1, "int_type": v[1],
                                                          "int_operandA": operA, "int_operandB": oper})

    return variations_int_t_op_a, variations_int_t_op_b, variations_int_t_op_x, variations_int_t_op_y, num_int_t_ops, \
           t_int_t_op


########################################################################################################################
# Analogy template: Floating point binary operations (type semantic analogy)
########################################################################################################################
def gen_anlgy_type_flpt_bin_op():
    """
    Generate analogy, example:
    <%ID> = fadd float <%ID>, <%ID>
    <%ID> = fadd double <%ID>, <%ID>
    <%ID> = fsub float <%ID>, <%ID>
    expected answer: <%ID> = fsub double <%ID>, <%ID>
    """

    # Variables containing variations
    variations_fl_t_op_a = list()
    variations_fl_t_op_b = list()
    variations_fl_t_op_x = list()
    variations_fl_t_op_y = list()

    # Counter
    num_fl_t_op = 0

    # Template
    t_fl_t_op = Template("<%ID> = {{float_op}} {{fast_math_flag}}{{float_type}} {{float_operandA}}, {{float_operandB}}")

    # Expression variations
    float_ops = ["fadd", "fsub", "fmul", "fdiv"]
    float_types = ["float", "<2 x float>", "<4 x float>", "<8 x float>",
                   "double", "<2 x double>", "<4 x double>", "<8 x double>"]
    fast_math_flags = ['', 'nnan ', 'ninf ', 'nsz ', 'arcp ', 'contract ', 'afn ', 'reassoc ', 'fast ']
    float_operandsA = ['<%ID>', "<FLOAT>"]
    float_operandsB = ['<%ID>', "<FLOAT>"]

    # Construct
    for p in pairs_from_array(float_ops):
        for t in pairs_from_array(float_types):
            for f in fast_math_flags:
                for oA in float_operandsA:
                    for o in float_operandsB:

                        num_fl_t_op += 1
                        variations_fl_t_op_a.append({"float_op": p[0], "fast_math_flag": f, "float_type": t[0],
                                                     "float_operandA": oA, "float_operandB": o})
                        variations_fl_t_op_b.append({"float_op": p[0], "fast_math_flag": f, "float_type": t[1],
                                                     "float_operandA": oA, "float_operandB": o})
                        variations_fl_t_op_x.append({"float_op": p[1], "fast_math_flag": f, "float_type": t[0],
                                                     "float_operandA": oA, "float_operandB": o})
                        variations_fl_t_op_y.append({"float_op": p[1], "fast_math_flag": f, "float_type": t[1],
                                                     "float_operandA": oA, "float_operandB": o})

    return variations_fl_t_op_a, variations_fl_t_op_b, variations_fl_t_op_x, variations_fl_t_op_y, num_fl_t_op, t_fl_t_op


########################################################################################################################
# Analogy template: Floating point ops (fast-math analogies)
########################################################################################################################
def gen_anlgy_type_flpt_bin_op_opt():
    """
    Generate analogy, example:
    <%ID> = fadd float <%ID>, <%ID>
    <%ID> = fadd double <%ID>, <%ID>
    <%ID> = fsub float <%ID>, <%ID>
    expected answer: <%ID> = fsub double <%ID>, <%ID>
    """

    # Variables containing variations
    variations_fl_t_op_a = list()
    variations_fl_t_op_b = list()
    variations_fl_t_op_x = list()
    variations_fl_t_op_y = list()

    # Counter
    num_fl_t_op = 0

    # Template
    t_fl_t_op = Template("<%ID> = {{float_op}} {{fast_math_flag}}{{float_type}} {{float_operandA}}, {{float_operandB}}")

    # Expression variations
    float_ops = ["fadd", "fsub", "fmul", "fdiv"]
    float_types = ["float", "<2 x float>", "<4 x float>", "<8 x float>", " <16 x float> ",
                   "double", "<2 x double>", "<4 x double>", "<8 x double>"]
    fast_math_flags = ['', 'fast ']
    float_operands = ['<%ID>', "<FLOAT>"]

    # Construct
    for p in pairs_from_array(float_ops):
        for t in float_types:
            for oA in float_operands:
                for o in float_operands:

                    num_fl_t_op += 1
                    variations_fl_t_op_a.append({"float_op": p[0], "fast_math_flag": fast_math_flags[0],
                                                 "float_type": t, "float_operandA": oA, "float_operandB": o})
                    variations_fl_t_op_b.append({"float_op": p[0], "fast_math_flag": fast_math_flags[1],
                                                 "float_type": t, "float_operandA": oA, "float_operandB": o})
                    variations_fl_t_op_x.append({"float_op": p[1], "fast_math_flag": fast_math_flags[0],
                                                 "float_type": t, "float_operandA": oA, "float_operandB": o})
                    variations_fl_t_op_y.append({"float_op": p[1], "fast_math_flag": fast_math_flags[1],
                                                 "float_type": t, "float_operandA": oA, "float_operandB": o})

    return variations_fl_t_op_a, variations_fl_t_op_b, variations_fl_t_op_x, variations_fl_t_op_y, num_fl_t_op, t_fl_t_op


########################################################################################################################
# Analogy template: Floating point / Integer binary operations (type semantic analogy)
########################################################################################################################
def gen_anlgy_type_flpt_int_bin_op():
    """
    Generate analogy, example:
    <%ID> = add i64 <%ID>, <%ID>
    <%ID> = fadd float <%ID>, <%ID>
    <%ID> = sub i64 <%ID>, <%ID>
    expected answer: <%ID> = fsub float <%ID>, <%ID>
    """
    # Variables containing variations
    variations_int_t_op_a = list()
    variations_fl_t_op_b = list()
    variations_int_t_op_x = list()
    variations_fl_t_op_y = list()

    # Counter
    num_intfl_t_ops = 0

    # Templates
    t_intfl_t_op = Template("<%ID> = {{op}} {{option}}{{type}} {{operandA}}, {{operandB}}")

    # Expression variations
    equiv_ops = {"add": "fadd",
                 "sub": "fsub",
                 "mul": "fmul"}
    types = {"i32": "float",
             "<4 x i32>": "<4 x float>",
             "<8 x i32>": "<8 x float>",
             "i64": "double",
             "<2 x i64>": "<2 x double>",
             "<4 x i64>": "<4 x double>",
             "<8 x i64>": "<8 x double>"}
    operands = {'<%ID>': '<%ID>',
                '<INT>': '<FLOAT>'}

    # Construct
    for pint in pairs_from_array(list(equiv_ops.keys())):
        for operA in operands.keys():
            for oper in operands.keys():
                for v in types.keys():

                    num_intfl_t_ops += 1
                    variations_int_t_op_a.append({"op": pint[0], "option": '', "type": v,
                                                  "operandA": operA, "operandB": oper})
                    variations_fl_t_op_b.append({"op": equiv_ops[pint[0]], "option": '', "type": types[v],
                                                 "operandA": operands[operA], "operandB": operands[oper]})
                    variations_int_t_op_x.append({"op": pint[1], "option": '', "type": v,
                                                  "operandA": operA, "operandB": oper})
                    variations_fl_t_op_y.append({"op": equiv_ops[pint[1]], "option": '', "type": types[v],
                                                 "operandA": operands[operA], "operandB": operands[oper]})

    return variations_int_t_op_a, variations_fl_t_op_b, variations_int_t_op_x, variations_fl_t_op_y, num_intfl_t_ops, \
           t_intfl_t_op


########################################################################################################################
# Analogy template: Insertelement - Extractelement operations (type)
########################################################################################################################
def gen_anlgy_insert_extract_type():
    """
    Generate analogy, example:
    <%ID> = insertelement <2 x float> <%ID>, float <%ID>, <TYP> 1
    <%ID> = insertelement <4 x float> <%ID>, float <%ID>, <TYP> 1
    <%ID> = extractelement <2 x float> <%ID>, <TYP> 1
    expected answer: <%ID> = extractelement <4 x float> <%ID>, <TYP> 1
    """
    # Variables containing variations
    variations_ie_a = list()
    variations_ie_b = list()
    variations_ie_x = list()
    variations_ie_y = list()

    # Counter
    num_ie = 0

    # Template
    t_ie = Template("<%ID> = {{op}} {{type}} {{middle_part}} <TYP> {{index}}")

    # Expression variations
    ops = ['insertelement', 'extractelement']
    types = [["<2 x float>", "float"], ["<4 x float>", "float"], ["<8 x float>", "float"],
             ["<2 x double>", "double"],  ["<4 x double>", "double"], ["<8 x double>", "double"],
             ["<4 x i1>", "i1"],
             ["<8 x i8>", "i8"], ["<32 x i8>", "i8"],
             ["<2 x i16>", "i16"], ["<4 x i16>", "i16"], ["<8 x i16>", "i16"], ["<16 x i16>", "i16"],
             ["<2 x i32>", "i32"], ["<4 x i32>", "i32"], ["<8 x i32>", "i32"], ["<16 x i32>", "i32"],
             ["<2 x i64>", "i64"], ["<4 x i64>", "i64"], ["<8 x i64>", "i64"], ["<16 x i64>", "i64"]]
    middle_part_insert_a = ['<%ID>, ', 'undef,']
    middle_part_insert_b = [' <%ID>,', '<FLOAT>,']
    middle_part_extract = '<%ID>,'
    indices = ["0", "1", "2", "3", "4", "5", "6", "7"]

    # Construct
    for t in pairs_from_array(types):
        for i in indices:
            for df in middle_part_insert_a:
                for df_ in middle_part_insert_b:

                    num_ie += 1
                    variations_ie_a.append({"op": ops[0], "type": t[0][0], "middle_part": df + t[0][1] + df_,
                                            "index": i})
                    variations_ie_b.append({"op": ops[0], "type": t[1][0], "middle_part": df + t[1][1] + df_,
                                            "index": i})
                    variations_ie_x.append({"op": ops[1], "type": t[0][0], "middle_part": middle_part_extract,
                                            "index": i})
                    variations_ie_y.append({"op": ops[1], "type": t[1][0], "middle_part": middle_part_extract,
                                            "index": i})

    return variations_ie_a, variations_ie_b, variations_ie_x, variations_ie_y, num_ie, t_ie


########################################################################################################################
# Analogy template: Insertelement - Extractelement operations (index analogy)
########################################################################################################################
def gen_anlgy_insert_extract_index():
    """
    Generate analogy, example:
    <%ID> = insertelement <2 x float> <%ID>, float <%ID>, <TYP> 0
    <%ID> = insertelement <2 x float> <%ID>, float <%ID>, <TYP> 1
    <%ID> = extractelement <2 x float> <%ID>, <TYP> 0
    <%ID> = extractelement <2 x float> <%ID>, <TYP> 1
    """
    # Variables containing variations
    variations_ie_el_ind_a = list()
    variations_ie_el_ind_b = list()
    variations_ie_el_ind_x = list()
    variations_ie_el_ind_y = list()

    # Counter
    num_ie_el_ind = 0

    # Template
    t_ie_el_ind = Template("<%ID> = {{op}} {{type}} {{middle_part}} <TYP> {{index}}")

    # Expression variations
    ops = ['insertelement', 'extractelement']
    types = [["<2 x float>", "float"], ["<4 x float>", "float"], ["<8 x float>", "float"],
             ["<2 x double>", "double"], ["<4 x double>", "double"], ["<8 x double>", "double"],
             ["<4 x i1>", "i1"],
             ["<8 x i8>", "i8"], ["<32 x i8>", "i8"],
             ["<2 x i16>", "i16"], ["<4 x i16>", "i16"], ["<8 x i16>", "i16"], ["<16 x i16>", "i16"],
             ["<2 x i32>", "i32"], ["<4 x i32>", "i32"], ["<8 x i32>", "i32"], ["<16 x i32>", "i32"],
             ["<2 x i64>", "i64"], ["<4 x i64>", "i64"], ["<8 x i64>", "i64"], ["<16 x i64>", "i64"]]
    middle_part_insert_a = ['<%ID>, ', 'undef,']
    middle_part_insert_b = [' <%ID>,', '<FLOAT>,']
    middle_part_extract = '<%ID>,'
    indices = ["0", "1", "2", "3", "4", "5", "6", "7"]

    # Construct
    for t in types:
        for i in pairs_from_array(indices):
            for df in middle_part_insert_a:
                for df_ in middle_part_insert_b:

                    num_ie_el_ind += 1
                    variations_ie_el_ind_a.append({"op": ops[0], "type": t[0], "middle_part": df + t[1] + df_,
                                                   "index": i[0]})
                    variations_ie_el_ind_b.append({"op": ops[0], "type": t[0], "middle_part": df + t[1] + df_,
                                                   "index": i[1]})
                    variations_ie_el_ind_x.append({"op": ops[1], "type": t[0], "middle_part": middle_part_extract,
                                                   "index": i[0]})
                    variations_ie_el_ind_y.append({"op": ops[1], "type": t[0], "middle_part": middle_part_extract,
                                                   "index": i[1]})

    return variations_ie_el_ind_a, variations_ie_el_ind_b, variations_ie_el_ind_x, variations_ie_el_ind_y, \
           num_ie_el_ind, t_ie_el_ind


########################################################################################################################
# Analogy template: Structure - Vector equivalents (a)
########################################################################################################################
def gen_anlgy_equiv_types_a():
    """
    Generate analogy, example:
    <%ID> = extractvalue { i32, i32, i32, i32 } <%ID>, 0
    <%ID> = extractelement <4 x i32> <%ID>, <TYP> 0
    <%ID> = extractvalue { i32, i32, i32, i32 } <%ID>, 1
    <%ID> = extractelement <4 x i32> <%ID>, <TYP> 1
    """

    # Variables containing variations
    variations_aggr_a = list()
    variations_aggr_b = list()
    variations_aggr_x = list()
    variations_aggr_y = list()

    # Counter
    num_aggr = 0

    # Template
    t_aggr = Template("<%ID> = {{op_beg}} {{type}}{{op_end}}")

    # Expression variations
    ops = [['extractvalue', ' <%ID>, 0', 'extractelement', ' <%ID>, <TYP> 0'],
           ['extractvalue', ' <%ID>, 1', 'extractelement', ' <%ID>, <TYP> 1'],
           ['extractvalue', ' <%ID>, 2', 'extractelement', ' <%ID>, <TYP> 2'],
           ['extractvalue', ' <%ID>, 3', 'extractelement', ' <%ID>, <TYP> 3'],
           ['alloca', ', align 4', 'alloca', ', align 4'],
           ['alloca', ', align 8', 'alloca', ', align 8'],
           ['alloca', ', align 16', 'alloca', ', align 16'],
           ['phi', '[ <%ID>, <%ID> ], [ <%ID>, <%ID> ]', 'phi', '[ <%ID>, <%ID> ], [ <%ID>, <%ID> ]'],
           ['phi', '[ <%ID>, <%ID> ], [ <%ID>, <%ID> ]', 'phi',
            '[ <%ID>, <%ID> ], [ <float <FLOAT>, float <FLOAT>>, <%ID> ]'],
           ['phi', '[ <%ID>, <%ID> ], [ <%ID>, <%ID> ]', 'phi',
            '[ <double <FLOAT>, double <FLOAT>>, <%ID> ], [ <%ID>, <%ID> ]']]
    types = [["{ i32, i32, i32, i32 }", "<4 x i32>"],
             ["{ i64, i64 }", "<2 x i64>"],
             ["{ float, float }", "<2 x float>"],
             ["{ { float, float } }", "<2 x float>"],
             ["{ double, double }", "<2 x double>"],
             ["{ { double, double } }", "<2 x double>"],
             ["{ float, float }*", "<2 x float>*"],
             ["{ { float, float } }*", "<2 x float>*"],
             ["{ double, double }*", "<2 x double>*"],
             ["{ { double, double } }*", "<2 x double>*"]]

    # Construct
    for op in pairs_from_array(ops):
        for t in types:

            num_aggr += 1
            variations_aggr_a.append({"op_beg": op[0][0], "type": t[0], "op_end": op[0][1]})
            variations_aggr_b.append({"op_beg": op[0][2], "type": t[1], "op_end": op[0][3]})
            variations_aggr_x.append({"op_beg": op[1][0], "type": t[0], "op_end": op[1][1]})
            variations_aggr_y.append({"op_beg": op[1][2], "type": t[1], "op_end": op[1][3]})

    return variations_aggr_a, variations_aggr_b, variations_aggr_x, variations_aggr_y, num_aggr, t_aggr


########################################################################################################################
# Analogy template: Structure - Vector equivalents (b)
########################################################################################################################
def gen_anlgy_equiv_types_b():
    """
    Generate analogy, example:
    <%ID> = load { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* <%ID>, align 1
    <%ID> = load <4 x i32>, <4 x i32>* <%ID>, align 1
    <%ID> = load { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* <%ID>, align 4
    <%ID> = load <4 x i32>, <4 x i32>* <%ID>, align 4
    """
    # Variables containing variations
    variations_aggrb_a = list()
    variations_aggrb_b = list()
    variations_aggrb_x = list()
    variations_aggrb_y = list()

    # Counter
    num_aggrb = 0

    # Template
    t_aggrb = Template("<%ID> = {{op_beg}} {{type}}, {{type}}* <%ID>, {{op_end}}")

    # Expression variations
    ops = [['load', 'align 1'],
           ['load', 'align 4'],
           ['load', 'align 8'],
           ['load', 'align 16'],
           ['getelementptr', 'i64 <INT>, i64 <INT>'],
           ['getelementptr', 'i64 <INT>, i32 <INT>'],
           ['getelementptr', 'i64 <%ID>, i32 <INT>, i32 <INT>'],
           ['getelementptr', 'i64 <%ID>'],
           ['getelementptr inbounds', 'i64 <INT>, i64 <INT>'],
           ['getelementptr inbounds', 'i64 <INT>, i32 <INT>'],
           ['getelementptr inbounds', 'i64 <%ID>, i32 <INT>, i32 <INT>'],
           ['getelementptr inbounds', 'i64 <%ID>']]
    types = [["{ i32, i32, i32, i32 }", "<4 x i32>"],
             ["{ float, float }", "<2 x float>"],
             ["{ { float, float } }", "<2 x float>"],
             ["{ double, double }", "<2 x double>"],
             ["{ { double, double } }", "<2 x double>"]]

    # Construct
    for op in pairs_from_array(ops):
        for t in types:

            num_aggrb += 1
            variations_aggrb_a.append({"op_beg": op[0][0], "type": t[0], "op_end": op[0][1]})
            variations_aggrb_b.append({"op_beg": op[0][0], "type": t[1], "op_end": op[0][1]})
            variations_aggrb_x.append({"op_beg": op[1][0], "type": t[0], "op_end": op[1][1]})
            variations_aggrb_y.append({"op_beg": op[1][0], "type": t[1], "op_end": op[1][1]})

    return variations_aggrb_a, variations_aggrb_b, variations_aggrb_x, variations_aggrb_y, num_aggrb, t_aggrb


########################################################################################################################
# Analogy template: Arithmetic integer binary operations (inverse operations analogy) (c)
########################################################################################################################
def gen_anlgy_equiv_types_c():
    """
    Generate analogy, example:
    <%ID> = bitcast i8* <%ID> to <2 x double>*
    <%ID> = bitcast i8* <%ID> to { { double, double } }*
    <%ID> = bitcast i8* <%ID> to <2 x double>*
    <%ID> = bitcast i8* <%ID> to { double, double }*
    """
    # Variables containing variations
    variations_aggrc_a = list()
    variations_aggrc_b = list()
    variations_aggrc_x = list()
    variations_aggrc_y = list()

    # Counter
    num_aggrc = 0

    # Template
    t_aggrc = Template("<%ID> = bitcast {{type1}} <%ID> to {{type2}}")

    # Expression variations
    types = [["i8*", "<2 x double>*", "{ { double, double } }*"],
             ["i8*", "<2 x double>*", "{ double, double }*"],
             ["i64*", "<2 x double>*", "{ double, double }*"],
             ["<2 x i64>", "<2 x double>", "{ double, double }"]]

    # Construct
    for t in pairs_from_array(types):

        num_aggrc += 1
        variations_aggrc_a.append({"type1": t[0][0], "type2": t[0][1]})
        variations_aggrc_b.append({"type1": t[0][0], "type2": t[0][2]})
        variations_aggrc_x.append({"type1": t[1][0], "type2": t[1][1]})
        variations_aggrc_y.append({"type1": t[1][0], "type2": t[1][2]})

    return variations_aggrc_a, variations_aggrc_b, variations_aggrc_x, variations_aggrc_y, num_aggrc, t_aggrc


########################################################################################################################
# Analogy template: Arithmetic integer binary operations (inverse operations analogy) (a)
########################################################################################################################
def gen_anlgy_op_inv_a():
    """
    Generate analogy, example:
    <%ID> = add i1 <%ID>, <%ID>
    <%ID> = sub i1 <%ID>, <%ID>
    <%ID> = mul i1 <%ID>, <%ID>
    <%ID> = udiv i1 <%ID>, <%ID>
    """
    # Variables containing variations
    variations_inv_bin_ops_a = list()
    variations_inv_bin_ops_b = list()
    variations_inv_bin_ops_x = list()
    variations_inv_bin_ops_y = list()

    # Counter
    num_inv_bin_ops = 0

    # Template
    t_inv_bin_ops = Template("<%ID> = {{inv_bin_opss}} {{inv_bin_ops_option}}{{int_type}} <%ID>, {{inv_bin_opserand}}")

    # Expression variations
    inv_bin_opss = [["add", "sub", "mul", "udiv"],
                    ["add", "sub", "mul", "sdiv"]]
    int_types = ["i1", "i2", "i4", "i8", "i32", "i64", "i128",
                 "<4 x i8>", "<8 x i8>", "<16 x i8>", "<32 x i8>",
                 "<4 x i16>", "<8 x i16>", "<16 x i16>",
                 "<4 x i32>", "<8 x i32>", "<16 x i32>", "<32 x i32>",
                 "<2 x i64>", "<4 x i64>", "<8 x i64>"]
    int_addsubmul_options = ['', 'nuw ', 'nsw ', 'nuw nsw ']
    int_div_options = ['', 'exact ']

    # Construct
    for op in inv_bin_opss:
        for v in int_types:
            for f1 in int_addsubmul_options:
                for f3 in int_div_options:

                    num_inv_bin_ops += 1
                    variations_inv_bin_ops_a.append({"inv_bin_opss": op[0], "inv_bin_ops_option": f1, "int_type": v,
                                                     "inv_bin_opserand": '<%ID>'})
                    variations_inv_bin_ops_b.append({"inv_bin_opss": op[1], "inv_bin_ops_option": f1, "int_type": v,
                                                     "inv_bin_opserand": '<%ID>'})
                    variations_inv_bin_ops_x.append({"inv_bin_opss": op[2], "inv_bin_ops_option": f1, "int_type": v,
                                                     "inv_bin_opserand": '<%ID>'})
                    variations_inv_bin_ops_y.append({"inv_bin_opss": op[3], "inv_bin_ops_option": f1, "int_type": v,
                                                     "inv_bin_opserand": '<%ID>'})

        return variations_inv_bin_ops_a, variations_inv_bin_ops_b, variations_inv_bin_ops_x, variations_inv_bin_ops_y, \
               num_inv_bin_ops, t_inv_bin_ops


########################################################################################################################
# Analogy template: Arithmetic flpt binary operations (inverse operations analogy) (b)
########################################################################################################################
def gen_anlgy_type_op_inv_b():
    """
    Generate analogy, example:
    <%ID> = fadd float <%ID>, <%ID>
    <%ID> = fsub float <%ID>, <%ID>
    <%ID> = fmul float <%ID>, <%ID>
    <%ID> = fdiv float <%ID>, <%ID>
    """
    # Variables containing variations
    variations_fl_bin_inv_ops_a = list()
    variations_fl_bin_inv_ops_b = list()
    variations_fl_bin_inv_ops_x = list()
    variations_fl_bin_inv_ops_y = list()

    # Counter
    num_fl_bin_inv_ops = 0

    # Template
    t_fl_bin_inv_ops = Template("<%ID> = {{float_op}} {{fast_math_flag}}{{float_type}} {{float_operandA}}, {{float_operandB}}")

    # expression variations
    float_ops = ["fadd", "fsub", "fmul", "fdiv"]
    float_types = ["float", "<2 x float>", "<4 x float>", "<8 x float>",
                   "double", "<2 x double>", "<4 x double>", "<8 x double>"]
    fast_math_flags = ['', 'nnan ', 'ninf ', 'nsz ', 'arcp ', 'contract ', 'afn ', 'reassoc ', 'fast ']
    float_operands = ['<%ID>', "<FLOAT>"]

    # Construct
    for t in float_types:
        for f in fast_math_flags:
            for oA in float_operands:
                for oB in float_operands:

                    num_fl_bin_inv_ops += 1
                    variations_fl_bin_inv_ops_a.append({"float_op": float_ops[0], "fast_math_flag": f, "float_type": t,
                                                        "float_operandA": oA, "float_operandB": oB})
                    variations_fl_bin_inv_ops_b.append({"float_op": float_ops[1], "fast_math_flag": f, "float_type": t,
                                                        "float_operandA": oA, "float_operandB": oB})
                    variations_fl_bin_inv_ops_x.append({"float_op": float_ops[2], "fast_math_flag": f, "float_type": t,
                                                        "float_operandA": oA, "float_operandB": oB})
                    variations_fl_bin_inv_ops_y.append({"float_op": float_ops[3], "fast_math_flag": f, "float_type": t,
                                                        "float_operandA": oA, "float_operandB": oB})

    return variations_fl_bin_inv_ops_a, variations_fl_bin_inv_ops_b, variations_fl_bin_inv_ops_x, \
           variations_fl_bin_inv_ops_y, num_fl_bin_inv_ops, t_fl_bin_inv_ops


########################################################################################################################
# Generate analogies and print them to file
########################################################################################################################
def generate_analogy_questions(analogy_questions_file):
    """
    Generate analogy questions ("A is to B what C is to D") consisting of 4 LLVM IR statements and print them to a file
    :param analogy_questions_file: name of file to print these analogies into (string)
    :return total number of analogies created
    """

    # Print
    print('\tPrinting analogy questions to file ', analogy_questions_file)
    tot_analogies = 0

    # Clean up file contents
    f = open(analogy_questions_file, 'w')
    f.close()

    ####################################################################################################################
    # syntactic type
    # Integer binary operations (type semantic analogy)
    descr = 'Integer binary operations (type semantic analogy)'
    print('\tGenerating:', descr)
    num_anlgy = write_analogy(gen_anlgy_type_int_bin_op(), descr, analogy_questions_file)
    tot_analogies += num_anlgy
    print('\tnumber generated analogies: {:>8,}'.format(num_anlgy))

    # Floating point binary operations (type semantic analogy):
    descr = 'Floating point binary operations (type semantic analogy)'
    print('\tGenerating:', descr)
    num_anlgy = write_analogy(gen_anlgy_type_flpt_bin_op(), descr, analogy_questions_file)
    tot_analogies += num_anlgy
    print('\tnumber generated analogies: {:>8,}'.format(num_anlgy))

    # Integer <-> Floating point binary operations (type semantic analogy):
    descr = 'Floating point / Integer binary operations (type semantic analogy)'
    print('\tGenerating:', descr)
    num_anlgy = write_analogy(gen_anlgy_type_flpt_int_bin_op(), descr, analogy_questions_file)
    tot_analogies += num_anlgy
    print('\tnumber generated analogies: {:>8,}'.format(num_anlgy))

    # Insertelement - Extractelement operations (type)
    descr = 'Insertelement - Extractelement operations (type)'
    print('\tGenerating:', descr)
    num_anlgy = write_analogy(gen_anlgy_insert_extract_type(), descr, analogy_questions_file)
    tot_analogies += num_anlgy
    print('\tnumber generated analogies: {:>8,}'.format(num_anlgy))

    ####################################################################################################################
    # syntactic
    # Fast math flags
    descr = 'Floating point ops (fast-math analogies)'
    print('\tGenerating:', descr)
    num_anlgy = write_analogy(gen_anlgy_type_flpt_bin_op_opt(), descr, analogy_questions_file)
    tot_analogies += num_anlgy
    print('\tnumber generated analogies: {:>8,}'.format(num_anlgy))

    # Insertelement - Extractelement operations (index analogy)
    descr = 'Insertelement - Extractelement operations (index analogy)'
    print('\tGenerating:', descr)
    num_anlgy = write_analogy(gen_anlgy_insert_extract_index(), descr, analogy_questions_file)
    tot_analogies += num_anlgy
    print('\tnumber generated analogies: {:>8,}'.format(num_anlgy))

    # Insertvalue - Extractvalue operations (index analogy)
    descr = 'Insertvalue - Extractvalue operations (index analogy)'
    print('\tGenerating:', descr)
    anlgies = [
        ["<%ID> = insertvalue { double, double } undef, double <%ID>, 0",
         "<%ID> = insertvalue { double, double } <%ID>, double <%ID>, 1",
         "<%ID> = extractvalue { double, double } <%ID>, 0",
         "<%ID> = extractvalue { double, double } <%ID>, 1"],
        ["<%ID> = insertvalue { float*, i64 } undef, float* <%ID>, 0",
         "<%ID> = extractvalue { float*, i64 } <%ID>, 0",
         "<%ID> = insertvalue { float*, i64 } <%ID>, i64 <%ID>, 1",
         "<%ID> = extractvalue { float*, i64 } <%ID>, 1"],
        ["<%ID> = insertvalue { i32*, i64 } undef, i32* <%ID>, 0",
         "<%ID> = extractvalue { i32*, i64 } <%ID>, 0",
         "<%ID> = insertvalue { i32*, i64 } <%ID>, i64 <%ID>, 1",
         "<%ID> = extractvalue { i32*, i64 } <%ID>, 1"],
        ["<%ID> = insertvalue { i8*, i32 } undef, i8* <%ID>, 0",
         "<%ID> = extractvalue { i8*, i32 } <%ID>, 0",
         "<%ID> = insertvalue { i8*, i32 } <%ID>, i32 <%ID>, 1",
         "<%ID> = extractvalue { i8*, i32 } <%ID>, 1"]
    ]
    num_anlgy = write_ready_analogy(anlgies, descr, analogy_questions_file)
    tot_analogies += num_anlgy
    print('\tnumber generated analogies: {:>8,}'.format(num_anlgy))

    ####################################################################################################################
    # syntactic: inverse operations
    # Bitcast x to y - y to x (inverse operations analogy)
    descr = 'Bitcast x to y - y to x (inverse operations analogy)'
    print('\tGenerating:', descr)
    anlgy_pair = [
                     ["<%ID> = bitcast <2 x double>* <%ID> to { double, double }*",
                      "<%ID> = bitcast { double, double }* <%ID> to <2 x double>*"],
                     ["<%ID> = bitcast <2 x i64>* <%ID> to { double, double }*",
                      "<%ID> = bitcast { double, double }* <%ID> to <2 x i64>*"],
                     ["<%ID> = bitcast <2 x float>* <%ID> to { float, float }*",
                      "<%ID> = bitcast { float, float }* <%ID> to <2 x float>*"],
                     ["<%ID> = bitcast i8* <%ID> to { double, double }*",
                      "<%ID> = bitcast { double, double }* <%ID> to i8*"],
                     ["<%ID> = bitcast i8* <%ID> to { opaque*, opaque* }*",
                      "<%ID> = bitcast { opaque*, opaque* }* <%ID> to i8*"],
                     ["<%ID> = bitcast { <{ opaque, opaque*, opaque*, i8, [7 x i8] }>* }** <%ID> to i8*",
                      "<%ID> = bitcast i8* <%ID> to { <{ opaque, opaque*, opaque*, i8, [7 x i8] }>* }**"],
                     ["<%ID> = bitcast { double, double }* <%ID> to <2 x double>*",
                      "<%ID> = bitcast <2 x double>* <%ID> to { double, double }*"],
                     ["<%ID> = bitcast { double, double }* <%ID> to <2 x i64>*",
                      "<%ID> = bitcast <2 x i64>* <%ID> to { double, double }*"],
                     ["<%ID> = bitcast { double, double }* <%ID> to i8*",
                      "<%ID> = bitcast i8* <%ID> to { double, double }*"],
                     ["<%ID> = bitcast { float, float }* <%ID> to <2 x float>*",
                      "<%ID> = bitcast <2 x float>* <%ID> to { float, float }*"],
                     ["<%ID> = bitcast { float, float }* <%ID> to i8*",
                      "<%ID> = bitcast i8* <%ID> to { float, float }*"],
                     ["<%ID> = bitcast { i64*, i64 }* <%ID> to { i64*, i64 }*",
                      "<%ID> = bitcast { { i64*, i64 } }* <%ID> to { i64*, i64 }*"],
                     ["<%ID> = bitcast { i8 }* <%ID> to { { { { i32*, i64 } } } }*",
                      "<%ID> = bitcast { { { { i32*, i64 } } } }* <%ID> to { i8 }*"],
                     ["<%ID> = bitcast { i8 }* <%ID> to { { { double*, i64, i64 } } }*",
                      "<%ID> = bitcast { { { double*, i64, i64 } } }* <%ID> to { i8 }*"],
                     ["<%ID> = bitcast { opaque*, opaque* }* <%ID> to i8*",
                      "<%ID> = bitcast i8* <%ID> to { opaque*, opaque* }*"],
                     ["<%ID> = bitcast { { i32*, i64, i64 } }* <%ID> to { { { i32*, i64, i64 } } }*",
                      "<%ID> = bitcast { { { i32*, i64, i64 } } }* <%ID> to { { i32*, i64, i64 } }*"],
                     ["<%ID> = bitcast { { i8* }, i64, { i64, [8 x i8] } }* <%ID> to i8*",
                      "<%ID> = bitcast i8* <%ID> to { { i8* }, i64, { i64, [8 x i8] } }*"],
                     ["<%ID> = bitcast { { { double*, i64, i64 } } }* <%ID> to i8*",
                      "<%ID> = bitcast { i8 }* <%ID> to { { { double*, i64, i64 } } }*"],
                     ["<%ID> = bitcast { { { { i32*, i64 } } } }* <%ID> to { i8 }*",
                      "<%ID> = bitcast { i8 }* <%ID> to { { { { i32*, i64 } } } }*"],
                     ["<%ID> = bitcast i8* <%ID> to { { { { { { i64, i64, i8* } } } } } }*",
                      "<%ID> = bitcast { { { { { { i64, i64, i8* } } } } } }* <%ID> to i8*"],
                     ["<%ID> = bitcast i8* <%ID> to { { { { { { i64, i64, i8* } } } } } }**",
                      "<%ID> = bitcast { { { { { { i64, i64, i8* } } } } } }** <%ID> to i8*"],
                     ["<%ID> = bitcast i8** <%ID> to { { { { { { i64, i64, i8* } } } } } }**",
                      "<%ID> = bitcast { { { { { { i64, i64, i8* } } } } } }** <%ID> to i8**"],
                     ["<%ID> = bitcast <2 x double>* <%ID> to i8*",
                      "<%ID> = bitcast i8* <%ID> to <2 x double>*"],
                     ["<%ID> = bitcast <16 x i8> <%ID> to <2 x i64>",
                      "<%ID> = bitcast <2 x i64> <%ID> to <16 x i8>"],
                     ["<%ID> = bitcast <2 x double> <%ID> to <4 x float>",
                      "<%ID> = bitcast <4 x float> <%ID> to <2 x double>"],
                     ["<%ID> = bitcast <2 x i64> <%ID> to <4 x i32>",
                      "<%ID> = bitcast <4 x i32> <%ID> to <2 x i64>"],
                     ["<%ID> = bitcast <2 x i64> <%ID> to <16 x i8>",
                      "<%ID> = bitcast <16 x i8> <%ID> to <2 x i64>"],
                     ["<%ID> = bitcast <4 x double> <%ID> to <4 x i64>",
                      "<%ID> = bitcast <4 x i64> <%ID> to <4 x double>"],
                     ["<%ID> = bitcast <4 x float>* <%ID> to i8*",
                      "<%ID> = bitcast i8* <%ID> to <4 x float>*"],
                     ["<%ID> = bitcast <4 x float> <%ID> to <2 x double>",
                      "<%ID> = bitcast <2 x double> <%ID> to <4 x float>"],
                     ["<%ID> = bitcast <4 x float> <%ID> to <4 x i32>",
                      "<%ID> = bitcast <4 x i32> <%ID> to <4 x float>"],
                     ["<%ID> = bitcast <4 x i32> <%ID> to <2 x i64>",
                      "<%ID> = bitcast <2 x i64> <%ID> to <4 x i32>"],
                     ["<%ID> = bitcast <4 x i32> <%ID> to <16 x i8>",
                      "<%ID> = bitcast <16 x i8> <%ID> to <4 x i32>"],
                     ["<%ID> = bitcast <4 x i32> <%ID> to <4 x float>",
                      "<%ID> = bitcast <4 x float> <%ID> to <4 x i32>"],
                     ["<%ID> = bitcast <4 x i64> <%ID> to <4 x double>",
                      "<%ID> = bitcast <4 x double> <%ID> to <4 x i64>"],
                     ["<%ID> = bitcast <8 x float> <%ID> to <8 x i32>",
                      "<%ID> = bitcast <8 x i32> <%ID> to <8 x float>"],
                     ["<%ID> = bitcast <8 x i32> <%ID> to <8 x float>",
                      "<%ID> = bitcast <8 x float> <%ID> to <8 x i32>"],
                     ["<%ID> = bitcast double* <%ID> to i64*",
                      "<%ID> = bitcast i64* <%ID> to double*"],
                     ["<%ID> = bitcast double* <%ID> to i8*",
                      "<%ID> = bitcast i8* <%ID> to double*"],
                     ["<%ID> = bitcast float <%ID> to i32",
                      "<%ID> = bitcast i32 <%ID> to float"],
                     ["<%ID> = bitcast double <%ID> to i64",
                      "<%ID> = bitcast i64 <%ID> to double"],
                     ["<%ID> = bitcast float* <%ID> to i8*",
                      "<%ID> = bitcast i8* <%ID> to float*"],
                     ["<%ID> = bitcast i16* <%ID> to i8*",
                      "<%ID> = bitcast i8* <%ID> to i16*"],
                     ["<%ID> = bitcast i32 <%ID> to float",
                      "<%ID> = bitcast float <%ID> to i32"],
                     ["<%ID> = bitcast i32* <%ID> to <2 x i64>*",
                      "<%ID> = bitcast <2 x i64>* <%ID> to i32*"],
                     ["<%ID> = bitcast i32** <%ID> to i64*",
                      "<%ID> = bitcast i64* <%ID> to i32**"],
                     ["<%ID> = bitcast i32* <%ID> to i64*",
                      "<%ID> = bitcast i64* <%ID> to i32*"],
                     ["<%ID> = bitcast i32* <%ID> to i8*",
                      "<%ID> = bitcast i8* <%ID> to i32*"],
                     ["<%ID> = bitcast i32** <%ID> to i8*",
                      "<%ID> = bitcast i8* <%ID> to i32**"],
                     ["<%ID> = bitcast i32** <%ID> to i8**",
                      "<%ID> = bitcast i8** <%ID> to i32**"],
                     ["<%ID> = bitcast i32* <%ID> to i8**",
                      "<%ID> = bitcast i8** <%ID> to i32*"]
    ]
    num_anlgy = write_analogy_from_pairs(anlgy_pair, descr, analogy_questions_file)
    tot_analogies += num_anlgy
    print('\tnumber generated analogies: {:>8,}'.format(num_anlgy))

    # Arithmetic integer binary operations (inverse operations analogy)
    descr = 'Arithmetic integer binary operations (inverse operations analogy)'
    print('\tGenerating:', descr)
    num_anlgy = write_analogy(gen_anlgy_op_inv_a(), descr, analogy_questions_file)
    tot_analogies += num_anlgy
    print('\tnumber generated analogies: {:>8,}'.format(num_anlgy))

    # Arithmetic floating point binary operations (inverse operations analogy)
    descr = 'Arithmetic flpt binary operations (inverse operations analogy)'
    print('\tGenerating:', descr)
    num_anlgy = write_analogy(gen_anlgy_type_op_inv_b(), descr, analogy_questions_file)
    tot_analogies += num_anlgy
    print('\tnumber generated analogies: {:>8,}'.format(num_anlgy))

    # Trunc - s/zext (inverse operations analogy)
    descr = 'Trunc - s/zext (inverse operations analogy)'
    print('\tGenerating:', descr)
    tot_analogies += num_anlgy
    anlgy_pair = [
        ["<%ID> = trunc <4 x i64> <%ID> to <4 x i32>",
         "<%ID> = sext <4 x i32> <%ID> to <4 x i64>"],
        ["<%ID> = trunc i128 <%ID> to i64",
         "<%ID> = sext i64 <%ID> to i128"],
        ["<%ID> = trunc i128 <%ID> to i64",
         "<%ID> = zext i64 <%ID> to i128"],
        ["<%ID> = trunc i16 <%ID> to i8",
         "<%ID> = zext i8 <%ID> to i16"],
        ["<%ID> = trunc i32 <%ID> to i16",
         "<%ID> = sext i16 <%ID> to i32"],
        ["<%ID> = trunc i32 <%ID> to i16",
         "<%ID> = zext i16 <%ID> to i32"],
        ["<%ID> = trunc i32 <%ID> to i8",
         "<%ID> = sext i8 <%ID> to i32"],
        ["<%ID> = trunc i32 <%ID> to i8",
         "<%ID> = zext i8 <%ID> to i32"],
        ["<%ID> = trunc i64 <%ID> to i16",
         "<%ID> = sext i16 <%ID> to i64"],
        [" <%ID> = trunc i64 <%ID> to i16",
         "<%ID> = zext i16 <%ID> to i64"],
        [" <%ID> = trunc i64 <%ID> to i32",
         "<%ID> = sext i32 <%ID> to i64"],
        ["<%ID> = trunc i64 <%ID> to i32",
         "<%ID> = zext i32 <%ID> to i64"],
        ["<%ID> = trunc i64 <%ID> to i8",
         "<%ID> = sext i8 <%ID> to i64"],
        ["<%ID> = trunc i8 <%ID> to i1",
         "<%ID> = zext i1 <%ID> to i8"]
    ]
    num_anlgy = write_analogy_from_pairs(anlgy_pair, descr, analogy_questions_file)
    print('\tnumber generated analogies: {:>8,}'.format(num_anlgy))

    # Fptou/si - s/uitofp (inverse operations analogy)
    descr = 'Fptou/si - s/uitofp (inverse operations analogy)'
    print('\tGenerating:', descr)
    anlgy_pair = [
            ["<%ID> = fptoui float <%ID> to i64",
             "<%ID> = uitofp i64 <%ID> to float"],
            ["<%ID> = fptosi double <%ID> to i32",
             "<%ID> = sitofp i32 <%ID> to double"],
            ["<%ID> = fptosi double <%ID> to i64",
             "<%ID> = sitofp i64 <%ID> to double"],
            ["<%ID> = fptosi float <%ID> to i32",
             "<%ID> = sitofp i32 <%ID> to float"],
    ]
    num_anlgy = write_analogy_from_pairs(anlgy_pair, descr, analogy_questions_file)
    tot_analogies += num_anlgy
    print('\tnumber generated analogies: {:>8,}'.format(num_anlgy))

    ####################################################################################################################
    # Inttoptr - ptrtoint (inverse operations analogy)
    descr = 'Inttoptr - ptrtoint (inverse operations analogy)'
    print('\tGenerating:', descr)
    anlgy_pair = [
        ["<%ID> = inttoptr i64 <%ID> to <{ opaque, opaque*, opaque*, i8, [7 x i8] }>*",
         "<%ID> = ptrtoint <{ opaque, opaque*, opaque*, i8, [7 x i8] }>* <%ID> to i64"],
        ["<%ID> = inttoptr i64 <%ID> to { <{ opaque, opaque*, opaque*, i8, [7 x i8] }>* }*",
         "<%ID> = ptrtoint { <{ opaque, opaque*, opaque*, i8, [7 x i8] }>* }* <%ID> to i64"],
        ["<%ID> = inttoptr i64 <%ID> to { double, double }*",
         "<%ID> = ptrtoint { double, double }* <%ID> to i64"],
        ["<%ID> = inttoptr i64 <%ID> to { float, float }*",
         "<%ID> = ptrtoint { float, float }* <%ID> to i64"],
        ["<%ID> = inttoptr i64 <%ID> to { i32, i32, i32, { [4 x i8*] }, { [4 x i8*] }, { opaque*, { { i32 (...)**, i64 }, i64 }* }, i32, opaque*, opaque* }**",
         "<%ID> = ptrtoint { i32, i32, i32, { [4 x i8*] }, { [4 x i8*] }, { opaque*, { { i32 (...)**, i64 }, i64 }* }, i32, opaque*, opaque* }** <%ID> to i64"],
        ["<%ID> = inttoptr i64 <%ID> to { i64, opaque, { i64 }, { i64 }, { { opaque*, opaque* } }, { i64 }, [8 x i8] }*",
         "<%ID> = ptrtoint { i64, opaque, { i64 }, { i64 }, { { opaque*, opaque* } }, { i64 }, [8 x i8] }* <%ID> to i64"],
        ["<%ID> = inttoptr i64 <%ID> to { opaque*, opaque* }*",
         "<%ID> = ptrtoint { opaque*, opaque* }* <%ID> to i64"],
        ["<%ID> = inttoptr i64 <%ID> to { { { double*, i64 } } }*",
         "<%ID> = ptrtoint { { { double*, i64 } } }* <%ID> to i64"],
        ["<%ID> = inttoptr i64 <%ID> to { { { double*, i64, i64 } } }*",
         "<%ID> = ptrtoint { { { double*, i64, i64 } } }* <%ID> to i64"],
        ["<%ID> = inttoptr i64 <%ID> to { { { i32*, i64 } } }*",
         "<%ID> = ptrtoint { { { i32*, i64 } } }* <%ID> to i64"],
        ["<%ID> = inttoptr i64 <%ID> to { { { { { { i64, i64, i8* } } } } } }*",
         "<%ID> = ptrtoint { { { { { { i64, i64, i8* } } } } } }* <%ID> to i64"],
        ["<%ID> = inttoptr i64 <%ID> to double*",
         "<%ID> = ptrtoint double* <%ID> to i64"],
        ["<%ID> = inttoptr i64 <%ID> to float*",
         "<%ID> = ptrtoint float* <%ID> to i64"],
        ["<%ID> = inttoptr i64 <%ID> to i32*",
         "<%ID> = ptrtoint i32* <%ID> to i64"],
        ["<%ID> = inttoptr i64 <%ID> to i64*",
         "<%ID> = ptrtoint i64* <%ID> to i64"],
        ["<%ID> = inttoptr i64 <%ID> to i8**",
         "<%ID> = ptrtoint i8** <%ID> to i64"],
        ["<%ID> = inttoptr i64 <%ID> to i8*",
         "<%ID> = ptrtoint i8* <%ID> to i64"]
    ]
    num_anlgy = write_analogy_from_pairs(anlgy_pair, descr, analogy_questions_file)
    tot_analogies += num_anlgy
    print('\tnumber generated analogies: {:>8,}'.format(num_anlgy))

    ####################################################################################################################
    # syntactic: structure/vector equivalents
    # Structure - Vector equivalents (a)
    descr = 'Structure - Vector equivalents (a)'
    print('\tGenerating:', descr)
    num_anlgy = write_analogy(gen_anlgy_equiv_types_a(), descr, analogy_questions_file)
    tot_analogies += num_anlgy
    print('\tnumber generated analogies: {:>8,}'.format(num_anlgy))

    # Structure - Vector equivalents (b)
    descr = 'Structure - Vector equivalents (b)'
    print('\tGenerating:', descr)
    num_anlgy = write_analogy(gen_anlgy_equiv_types_b(), descr, analogy_questions_file)
    tot_analogies += num_anlgy
    print('\tnumber generated analogies: {:>8,}'.format(num_anlgy))

    # Structure - Vector equivalents (c)
    descr = 'Structure - Vector equivalents (c)'
    print('\tGenerating:', descr)
    num_anlgy = write_analogy(gen_anlgy_equiv_types_c(), descr, analogy_questions_file)
    tot_analogies += num_anlgy
    print('\tnumber generated analogies: {:>8,}'.format(num_anlgy))

    # Print and return
    print('\tAnalogies printed: {:>10,d} analogies were generated'.format(tot_analogies))
    return tot_analogies
