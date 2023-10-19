"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import Optional

import haiku as hk
import jax.numpy as jnp
import numpy as np
from jax import lax


class LinearBlock(hk.Module):
    def __init__(
        self,
        output_size: int,
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        batch_norm: bool = False,
        reparametrized_linear: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        if reparametrized_linear:
            self.linear = ReparametrizedLinear(output_size, with_bias, w_init, b_init)
        else:
            self.linear = hk.Linear(output_size, with_bias, w_init, b_init)
        if batch_norm:
            self.batch_norm = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.0)
        else:
            self.batch_norm = None

    def __call__(self, inputs: jnp.array, is_training: bool):
        out = inputs
        out = self.linear(out)
        if self.batch_norm is not None:
            out = self.batch_norm(out, is_training)

        return out


class ReparametrizedLinear(hk.Linear):
    """
    Reparametrized version of the linear layer that performs the
    fan-in scaling of weights as part of the forward pass instead
    of when initialising the weights. This can be useful for when
    weights are generated by a hypernetwork.
    """
    def __call__(
        self,
        inputs: jnp.ndarray,
        *,
        precision: Optional[lax.Precision] = None,
    ) -> jnp.ndarray:
        """
        Computes a linear transform of the input.
        """
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype

        w_init = self.w_init
        if w_init is None:
            w_init = hk.initializers.TruncatedNormal(stddev=1.0)

        w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)

        # Scale weights by typical init factor
        scale = 1. / np.sqrt(self.input_size)
        out = jnp.dot(inputs, scale * w, precision=precision)

        if self.with_bias:
            b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
            b = jnp.broadcast_to(b, out.shape)
            out = out + b

        return out
