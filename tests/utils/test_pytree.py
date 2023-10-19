"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import unittest

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from metax.utils.pytree import PytreeReshaper


class PytreeReshaperTestCase(unittest.TestCase):
    def test_invariance(self):
        @hk.without_apply_rng
        @hk.transform
        def net(x):
            return hk.nets.MLP((8, 8))(x)

        pytree1 = net.init(jax.random.PRNGKey(0), jnp.empty((1, 1)))
        reshaper = PytreeReshaper(jtu.tree_map(jnp.shape, pytree1))
        pytree2 = net.init(jax.random.PRNGKey(1), jnp.empty((1, 1)))
        chex.assert_trees_all_close(reshaper(reshaper.flatten(pytree2)), pytree2)
