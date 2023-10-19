"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import List

import jax
import jax.numpy as jnp

from metax.data.base import MultitaskDataset
from metax.data.dataset.base import DatasetGenerator


class Family(DatasetGenerator):
    def __init__(self, fun_types: List = ["harm", "lin", "poly", "saw", "sine"]):
        super().__init__(input_shape=(1, ), output_dim=1)
        self.fun_types = fun_types

    def _rescale_linear(self, x, old_min, old_max, new_min, new_max):
        """
        Linearly rescale an array to the range [new_min, new_max].
        """
        slope = (new_max - new_min) / (old_max - old_min)
        intercept = new_min - slope * old_min

        return slope * x + intercept

    def sample_harmonics(self, rng, num_tasks, num_samples):
        @jnp.vectorize
        def harmonic(input, a1, a2, b1, b2, frequency):
            return a1 * jnp.sin(frequency * input + b1) + a2 * jnp.sin(2 * frequency * input + b2)

        rng_x, rng_freq, rng_a1, rng_a2, rng_b1, rng_b2 = jax.random.split(rng, 6)

        inputs = jax.random.uniform(rng_x, shape=(num_tasks, num_samples, 1), minval=-5.0, maxval=5.0)
        frequency = jax.random.uniform(rng_freq, shape=(num_tasks, 1, 1), minval=0.1, maxval=1.0)
        a1 = jax.random.uniform(rng_a1, shape=(num_tasks, 1, 1), minval=0.0, maxval=0.5)
        a2 = jax.random.uniform(rng_a2, shape=(num_tasks, 1, 1), minval=0.0, maxval=0.5)
        b1 = jax.random.uniform(rng_b1, shape=(num_tasks, 1, 1), minval=0.0, maxval=2.0 * jnp.pi)
        b2 = jax.random.uniform(rng_b2, shape=(num_tasks, 1, 1), minval=0.0, maxval=2.0 * jnp.pi)

        return inputs, harmonic(inputs, a1, a2, b1, b2, frequency)

    def sample_linears(self, rng, num_tasks, num_samples):
        @jnp.vectorize
        def linear(input, intercept, slope):
            return intercept + slope * input

        rng_x, rng_intercept, rng_slope = jax.random.split(rng, 3)
        inputs = jax.random.uniform(rng_x, shape=(num_tasks, num_samples, 1), minval=-5.0, maxval=5.0)
        intercepts = jax.random.uniform(rng_intercept, shape=(num_tasks, 1, 1), minval=-0.5, maxval=0.5)
        slopes = jax.random.uniform(rng_slope, shape=(num_tasks, 1, 1), minval=-0.1, maxval=0.1)

        return inputs, linear(inputs, intercepts, slopes)

    def sample_polynomials(self, rng, num_tasks, num_samples):
        @jnp.vectorize
        def polynomial(input, a, b, c):
            return a + b * input + c * input**2

        rng_x, rng_a, rng_b, rng_c = jax.random.split(rng, 4)

        inputs = jax.random.uniform(rng_x, shape=(num_tasks, num_samples, 1), minval=-5.0, maxval=5.0)
        a = jax.random.uniform(rng_a, shape=(num_tasks, 1, 1), minval=-5.0, maxval=5.0)
        b = jax.random.uniform(rng_b, shape=(num_tasks, 1, 1), minval=-5.0, maxval=5.0)
        c = jax.random.uniform(rng_c, shape=(num_tasks, 1, 1), minval=-5.0, maxval=5.0)

        targets = polynomial(inputs, a, b, c)

        # Normalize targets to range -1, 1
        mins = jnp.min(targets, axis=1, keepdims=True)
        maxs = jnp.max(targets, axis=1, keepdims=True)
        targets = self._rescale_linear(targets, mins, maxs, -1.0, 1.0)

        return inputs, targets

    def sample_sawtooths(self, rng, num_tasks, num_samples):
        @jnp.vectorize
        def sawtooth(input, amplitude, phase, width=1):
            """
            scipy.signal.sawtooth is not jittable, so we implement our own version here.
            """
            input = input - phase
            period = 2 * jnp.pi
            phase = input / period
            phase_shifted = phase - jnp.floor(phase)
            return amplitude * (2 * (phase_shifted - width / 2) / width)

        rng_x, rng_amp, rng_phase = jax.random.split(rng, 3)
        inputs = jax.random.uniform(rng_x, shape=(num_tasks, num_samples, 1), minval=-5.0, maxval=5.0)
        amplitude = jax.random.uniform(rng_amp, shape=(num_tasks, 1, 1), minval=0.1, maxval=1.0)
        phase = jax.random.uniform(rng_phase, shape=(num_tasks, 1, 1), minval=0.0, maxval=2.0 * jnp.pi)

        return inputs, sawtooth(inputs, amplitude, phase)

    def sample_sinusoids(self, rng, num_tasks, num_samples):
        @jnp.vectorize
        def sinusoid(input, amplitude, phase):
            return amplitude * jnp.sin(input + phase)

        rng_x, rng_amp, rng_phase = jax.random.split(rng, 3)

        inputs = jax.random.uniform(rng_x, shape=(num_tasks, num_samples, 1), minval=-5.0, maxval=5.0)
        amplitudes = jax.random.uniform(rng_amp, shape=(num_tasks, 1, 1), minval=0.1, maxval=1.0)
        phases = jax.random.uniform(rng_phase, shape=(num_tasks, 1, 1), minval=0.0, maxval=jnp.pi)

        return inputs, sinusoid(inputs, amplitudes, phases)

    def sample(self, rng, num_tasks, num_samples, mode=None):
        rng_harm, rng_lin, rng_poly, rng_saw, rng_sine, rng_perm = jax.random.split(rng, 6)

        # Sample tasks for each family in fun_types
        inputs, targets = [], []
        if "harm" in self.fun_types:
            x, y = self.sample_harmonics(
                rng_harm, num_tasks=num_tasks, num_samples=num_samples
            )
            inputs.append(x)
            targets.append(y)

        if "lin" in self.fun_types:
            x, y = self.sample_linears(
                rng_lin, num_tasks=num_tasks, num_samples=num_samples
            )
            inputs.append(x)
            targets.append(y)

        if "poly" in self.fun_types:
            x, y = self.sample_polynomials(
                rng_poly, num_tasks=num_tasks, num_samples=num_samples
            )
            inputs.append(x)
            targets.append(y)

        if "saw" in self.fun_types:
            x, y = self.sample_sawtooths(
                rng_saw, num_tasks=num_tasks, num_samples=num_samples
            )
            inputs.append(x)
            targets.append(y)

        if "sine" in self.fun_types:
            x, y = self.sample_sinusoids(
                rng_sine, num_tasks=num_tasks, num_samples=num_samples
            )
            inputs.append(x)
            targets.append(y)

        # Combine all tasks and randomly permute the order of tasks
        inputs = jnp.concatenate(inputs)
        targets = jnp.concatenate(targets)

        task_ids = jnp.repeat(jnp.repeat(jnp.arange(len(self.fun_types)), num_tasks)[:, None], num_samples, axis=1)  # Consistent leading dims
        idx = jax.random.permutation(rng_perm, num_tasks)

        return MultitaskDataset(x=inputs[idx][:num_tasks], y=targets[idx][:num_tasks], task_id=task_ids[idx][:num_tasks])


class Harmonic(Family):
    def __init__(self):
        super().__init__(fun_types=["harm"])


class Linear(Family):
    def __init__(self):
        super().__init__(fun_types=["lin"])


class Polynomial(Family):
    def __init__(self):
        super().__init__(fun_types=["poly"])


class Sawtooth(Family):
    def __init__(self):
        super().__init__(fun_types=["saw"])


class SinusoidFamily(Family):
    def __init__(self):
        # NOTE: This sinusoid is slightly different from the canonical sinewave regression
        super().__init__(fun_types=["sine"])
