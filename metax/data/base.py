"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import abc
import os
from pathlib import Path
from typing import Dict, NamedTuple, Tuple, Union

from chex import Array

DATAPATH = Path(os.path.expanduser("~/data/jax"))


class Dataloader(abc.ABC):
    def __init__(self, input_shape: Tuple[int], output_dim: int):
        self.input_shape = input_shape
        self.output_dim = output_dim

    @abc.abstractproperty
    def __len__(self):
        pass

    @abc.abstractproperty
    def sample_input(self):
        # Sample input should include batch dimension
        pass

    @abc.abstractmethod
    def __iter__(self):
        pass


class Dataset(NamedTuple):
    x: Array
    y: Array
    info: Dict = dict()


class MultitaskDataset(NamedTuple):
    x: Array
    y: Array
    task_id: Array
    info: Dict = dict()


class MetaDataset(NamedTuple):
    train: Union[Dataset, MultitaskDataset]
    test: Union[Dataset, MultitaskDataset]
