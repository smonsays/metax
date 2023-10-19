"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from functools import partial

import jax
import jax.numpy as jnp

from metax.data.base import Dataset
from metax.data.dataset.base import DatasetGenerator


class Sinusoid(DatasetGenerator):
    def __init__(self):
        super().__init__(input_shape=(1, ), output_dim=1)

    @partial(jax.jit, static_argnames=("self", "num_tasks", "num_samples", "mode"))
    def sample(self, rng, num_tasks, num_samples, mode=None):
        @jnp.vectorize
        def sinusoid(inputs, amplitude, phase):
            targets = amplitude * jnp.sin(inputs + phase)

            return targets

        rng_x, rng_amp, rng_phase = jax.random.split(rng, num=3)
        inputs = jax.random.uniform(rng_x, shape=(num_tasks, num_samples, 1), minval=-5.0, maxval=5.0)
        amplitudes = jax.random.uniform(rng_amp, shape=(num_tasks, 1, 1), minval=0.1, maxval=0.5)
        phases = jax.random.uniform(rng_phase, shape=(num_tasks, 1, 1), minval=0.0, maxval=jnp.pi)

        return Dataset(x=inputs, y=sinusoid(inputs, amplitudes, phases))
