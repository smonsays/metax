"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import math

import chex
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np


class PytreeReshaper:
    def __init__(self, tree_shapes):
        self.shapes, self.treedef = jtu.tree_flatten(
            tree_shapes, is_leaf=is_tuple_of_ints
        )
        sizes = [math.prod(shape) for shape in self.shapes]

        self.split_indeces = list(np.cumsum(sizes)[:-1])
        self.num_elements = sum(sizes)

    def __call__(self, array_flat):
        arrays_split = jnp.split(array_flat, self.split_indeces)
        arrays_reshaped = [a.reshape(shape) for a, shape in zip(arrays_split, self.shapes)]

        return jtu.tree_unflatten(self.treedef, arrays_reshaped)

    @staticmethod
    def flatten(pytree):
        return jnp.concatenate([jnp.ravel(e) for e in jtu.tree_flatten(pytree)[0]])


def is_tuple_of_ints(x):
    return isinstance(x, tuple) and all(isinstance(v, int) for v in x)


def tree_index(pytree, idx):
    return jtu.tree_map(lambda x: x[idx], pytree)


def tree_length(pytree):
    """
    Get size of leading dim assuming all leaves have the same.
    """
    chex.assert_equal_shape(jtu.tree_leaves(pytree), dims=0)

    return len(jtu.tree_leaves(pytree)[0])
