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
"""Compile classifyapp data to LLVM IR"""

from absl import app, flags
import os
import random
import subprocess
import template_vars


class CompilerArgumentGenerator(object):
    def __init__(self):
        self.compiler = template_vars.ValueListVar(
            ['g++ -fplugin=dragonegg.so -S -fplugin-arg-dragonegg-emit-ir -std=c++11', 
             'clang++ -S -emit-llvm -std=c++11'])
        self.optimization = template_vars.ValueListVar(['-O0','-O1','-O2','-O3'])
        self.fastmath = template_vars.ValueListVar(['', '-ffast-math'])
        self.native = template_vars.ValueListVar(['', '-march=native'])

    # Returns a tuple (cmdline, output_filename) -- for indexing purposes
    def get_cmdline(self, input_path, input_filename, additional_flags):        
        # file.cpp -> file_RANDOM.ll
        output_filename = (input_filename[:-4] + '_' +
                           template_vars.RandomStrVar()[0] + '.ll')

        args = [self.compiler, self.optimization, self.fastmath, self.native]
        arg_strs = [str(random.choice(arg)) for arg in args]
        return (' '.join(arg_strs) + ' ' + input_path+'/'+input_filename + 
                ' '+ additional_flags + ' -o ', output_filename)


flags.DEFINE_string('input_path', None, 'Input path of rendered .cpp files')
flags.DEFINE_string('output_path', None, 'Output path to store .ll files')
flags.DEFINE_string('compile_one', None, 'Define a single template to compile')
flags.DEFINE_integer('ir_per_file', 32, 'Number of .ll files generated per input')
flags.DEFINE_string('compiler_flags', '', 'Additional compiler flags')
FLAGS = flags.FLAGS


def _createdir(dname):
    try:
        os.makedirs(dname)
    except FileExistsError:
        pass


def main(argv):
    del argv

    cag = CompilerArgumentGenerator()

    cwd = os.path.dirname(os.path.abspath(__file__))

    inpath = cwd + '/code'
    outpath = cwd + '/llvm_ir'
    if FLAGS.input_path is not None:
        inpath = FLAGS.input_path
    if FLAGS.output_path is not None:
        outpath = FLAGS.output_path

    # Create output directory
    _createdir(outpath)

    classlist = [f for f in os.listdir(inpath)]
    if FLAGS.compile_one is not None:
        classlist = [FLAGS.compile_one]

    for clazz in classlist:
        filelist = [f for f in os.listdir(inpath + '/' + clazz) if '.cpp' in f]
        _createdir(outpath + '/' + clazz)

        for file in filelist:
            for i in range(FLAGS.ir_per_file):
                cmdline, outfile = \
                    cag.get_cmdline('%s/%s' % (inpath, clazz), file, 
                                    FLAGS.compiler_flags)

                # Append mapping of file to cmdline in logfile
                with open('%s/%s/flags.log' % (outpath, clazz), 'a') as f:
                    f.write(outfile + ': ' + cmdline + '\n')

                fullopath = '%s/%s/%s' % (outpath, clazz, outfile)
                print(cmdline + fullopath)
                subprocess.call(cmdline + fullopath, shell=True)


if __name__ == '__main__':
    app.run(main)
