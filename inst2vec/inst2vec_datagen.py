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
"""Generate dataset for inst2vec training"""

import wget
import zipfile
import os
import shutil


########################################################################################################################
# Main function for data set generation
########################################################################################################################
def datagen(data_folder):
    """
    Download and unzip training data for inst2vec
    :param data_folder: folder in which to put the downloaded data
    """

    ####################################################################################################################
    # Download and unzip inst2vec training data
    download_and_unzip('http://spclstorage.inf.ethz.ch/projects/ncc/datasets/amdappsdk.zip', 'AMD', data_folder)
    download_and_unzip('http://spclstorage.inf.ethz.ch/projects/ncc/datasets/blas.zip', 'BLAS', data_folder)
    download_and_unzip('http://spclstorage.inf.ethz.ch/projects/ncc/datasets/eigen-synth.zip', 'eigen_synthetic', data_folder)
    download_and_unzip('http://spclstorage.inf.ethz.ch/projects/ncc/datasets/gemm-synth.zip', 'gemm_synthetic', data_folder)
    download_and_unzip('http://spclstorage.inf.ethz.ch/projects/ncc/datasets/linuxkernel.zip', 'linux-4.15', data_folder)
    download_and_unzip('http://spclstorage.inf.ethz.ch/projects/ncc/datasets/opencv.zip', 'opencv', data_folder)
    download_and_unzip('http://spclstorage.inf.ethz.ch/projects/ncc/datasets/polybenchgpu.zip', 'polybenchGPU', data_folder)
    download_and_unzip('http://spclstorage.inf.ethz.ch/projects/ncc/datasets/rodinia.zip', 'rodinia_3.1', data_folder)
    download_and_unzip('http://spclstorage.inf.ethz.ch/projects/ncc/datasets/shoc.zip', 'shoc', data_folder)
    download_and_unzip('http://spclstorage.inf.ethz.ch/projects/ncc/datasets/stencil-synth.zip', 'stencil_synthetic', data_folder)
    download_and_unzip('http://spclstorage.inf.ethz.ch/projects/ncc/datasets/tensorflow.zip', 'tensorflow', data_folder)

    ####################################################################################################################
    # Remove __MACOSX directory resulting from unzipping
    if os.path.exists(os.path.join(data_folder, '__MACOSX')):
        shutil.rmtree(os.path.join(data_folder, '__MACOSX'))


########################################################################################################################
# Helper function
########################################################################################################################
def download_and_unzip(url, dataset_name, data_folder):
    """
    Download and unzip data set folder from url
    :param url: from which to download
    :param dataset_name: name of data set (for printing)
    :param data_folder: folder in which to put the downloaded data
    """
    print('Downloading', dataset_name, 'data set...')
    data_zip = wget.download(url, out=data_folder)
    print('\tunzipping...')
    zip_ = zipfile.ZipFile(data_zip, 'r')
    zip_.extractall(data_folder)
    zip_.close()
    print('\tdone')
