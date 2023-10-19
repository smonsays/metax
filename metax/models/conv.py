"""

MIT License

Copyright (c) Simon Schug

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import math
from typing import Optional

import haiku as hk
import jax.numpy as jnp
from jax import nn


class ConvBlock(hk.Module):
    def __init__(self, channels, decay, create_scale_and_offset, max_pool, name=None):
        super().__init__(name=name)
        self.channels = channels
        self.decay = decay
        self.max_pool = max_pool

        self.conv = hk.Conv2D(
            self.channels,
            kernel_shape=3,
            stride=1 if self.max_pool else 2,
            with_bias=True,
            w_init=hk.initializers.VarianceScaling(
                1.0, "fan_avg", "truncated_normal"
            ),  # Glorot normal
            name="conv",
        )

        self.norm = hk.BatchNorm(
            create_scale=create_scale_and_offset,
            create_offset=create_scale_and_offset,
            decay_rate=self.decay,
            name="norm",
        )

    def __call__(self, inputs, is_training, gain=None, shift=None):
        outputs = self.conv(inputs)
        outputs = self.norm(outputs, is_training, scale=gain, offset=shift)
        outputs = nn.relu(outputs)

        if self.max_pool:
            outputs = hk.max_pool(outputs, 2, 2, padding="VALID")

        return outputs


class Conv4(hk.Module):
    def __init__(
        self,
        channels,
        decay,
        readout=None,
        max_pool=True,
        create_scale_and_offset=True,
        normalize_outputs=False,
        name=None,
    ):
        super().__init__(name=name)
        self.channels = channels
        self.decay = decay
        self.readout = readout
        self.max_pool = max_pool
        self.normalize_outputs = normalize_outputs

        self.layers = tuple([
            ConvBlock(
                channels=self.channels,
                decay=self.decay,
                create_scale_and_offset=create_scale_and_offset,
                max_pool=max_pool,
                name="layer{}".format(i),
            )
            for i in range(4)
        ])

    def __call__(
        self,
        inputs: jnp.ndarray,
        is_training: bool,
        gain: Optional[jnp.ndarray] = None,
        shift: Optional[jnp.ndarray] = None,
    ):
        out = inputs
        for i, layer in enumerate(self.layers):
            g = gain[i] if gain is not None else None
            s = shift[i] if shift is not None else None
            out = layer(out, is_training, gain=g, shift=s)

        out = out.reshape(inputs.shape[:-3] + (-1,))

        if self.normalize_outputs:
            out /= math.sqrt(out.shape[-1])

        if self.readout is not None:
            out = hk.Linear(self.readout, with_bias=True)(out)

        return out
