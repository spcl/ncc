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
"""Template variations"""

import itertools
import random
import string
import math


class TemplateVar(object):
    def __iter__(self):
        raise NotImplementedError
    def __getitem__(self, index):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError


class RangeVar(TemplateVar):
    def __init__(self, start, stop, skip, *args, **kwargs):
        self.start = start
        self.stop = stop
        self.skip = skip
        return super().__init__(*args, **kwargs)

    def __len__(self):
        return len(range(self.start, self.stop, self.skip))

    def __getitem__(self, index):
        return list(range(self.start, self.stop, self.skip))[index]

    def __iter__(self):
        yield from range(self.start, self.stop, self.skip)


class ValueListVar(TemplateVar):
    def __init__(self, values, *args, **kwargs):
        self.values = values
        return super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def __iter__(self):
        yield from iter(self.values)


class BoolVar(ValueListVar):
    def __init__(self):
        super().__init__(['false', 'true'])


class PermutationVar(TemplateVar):
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return math.factorial(len(self.values))

    def __getitem__(self, index):
        return list(itertools.permutations(self.values))[index]

    def __iter__(self):
        yield from itertools.permutations(self.values)


class RandomStrVar(TemplateVar):
    def __len__(self):
        return 1
    
    def _randomize(self):
        return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))

    def __getitem__(self, index):
        return self._randomize()

    def __iter__(self):
        while True:
            yield self._randomize()


_TYPES = [
    'int8_t', 'int16_t', 'int32_t', 'int64_t',
    'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
    'float', 'double', 'std::complex<float>', 'std::complex<double>'
]


class TypeVar(ValueListVar):
    def __init__(self):
        return super().__init__(_TYPES)