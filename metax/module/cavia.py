
"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp

from .base import MetaModule


class CAVIAMetaParams(NamedTuple):
    base: hk.Params


class CAVIAParams(NamedTuple):
    context: jnp.array


class CAVIAMetaState(NamedTuple):
    base: hk.State


class CAVIAState(NamedTuple):
    pass


class CAVIA(MetaModule):
    def __init__(self, loss_fn_inner, loss_fn_outer, base_model, embedding_dim):
        super().__init__(loss_fn_inner, loss_fn_outer)
        self.embedding_dim = embedding_dim
        self.base_model = base_model

    def __call__(self, rng, state, hstate, params, hparams, input, is_training):
        rng_base, = jax.random.split(rng, 1)

        # Concatenate context parameters to input assuming input is flattened
        input_context = jnp.concatenate(
            (input, jnp.resize(params.context, (input.shape[0], self.embedding_dim))), axis=-1
        )

        out, hstate_base = self.base_model.apply(
            hparams.base, hstate.base, rng_base, input_context, is_training
        )

        return out, (CAVIAState(), CAVIAMetaState(base=hstate_base))

    def reset_hparams(self, rng, sample_input):
        rng_base, = jax.random.split(rng, 1)

        sample_input_context = jnp.concatenate(
            (jnp.zeros((sample_input.shape[0], self.embedding_dim)), sample_input),
            axis=-1
        )
        hparams_base, state_base = self.base_model.init(
            rng_base, sample_input_context, is_training=True
        )

        return CAVIAMetaParams(base=hparams_base), CAVIAMetaState(base=state_base)

    def reset_params(self, rng, hparams, hstate, sample_input):
        params_context = jnp.zeros((self.embedding_dim))

        return CAVIAParams(context=params_context), CAVIAState()
