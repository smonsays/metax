"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import Dict, NamedTuple

import jax
import jax.numpy as jnp

from metax import energy, models

from .base import MetaModule


class GainModMetaParams(NamedTuple):
    body: Dict
    gain_init: Dict
    head_init: Dict
    shift_init: Dict


class GainModParams(NamedTuple):
    gain: Dict
    head: Dict
    shift: Dict


class GainModMetaState(NamedTuple):
    body: Dict
    head: Dict


class GainModState(NamedTuple):
    head: Dict


class GainMod(MetaModule):
    def __init__(
        self,
        loss_fn_inner,
        loss_fn_outer,
        body,
        hidden_dims,
        output_dim,
        adapt_head,
        reg_strength,
    ):
        super().__init__(loss_fn_inner, loss_fn_outer)

        if reg_strength is not None:
            key_map = {"gain": "gain_init", "shift": "shift_init"}

            if adapt_head:
                key_map["head"] = "head_init"

            self.loss_fn_inner += energy.iMAML(
                reg_strength=reg_strength,
                key_map=key_map,
                reduction="sum"
            )

        self.body = body
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.adapt_head = adapt_head
        self.head = models.Linear(output_dim)

    def __call__(self, rng, state, hstate, params, hparams, input, is_training):

        rng_body, rng_head = jax.random.split(rng)

        features, state_body = self.body.apply(
            params=hparams.body,
            state=hstate.body,
            rng=rng_body,
            inputs=input,
            is_training=is_training,
            shift=params.shift,
            gain=params.gain
        )

        # NOTE: Both hstate and state contain state_head, which one is used depends on `adapt_head`
        if self.adapt_head:
            out, state_head = self.head.apply(
                params.head, state.head, rng_head, features, is_training
            )
        else:
            out, state_head = self.head.apply(
                hparams.head_init, hstate.head, rng_head, features, is_training
            )

        return out, (GainModState(state_head), GainModMetaState(state_body, state_head))

    def reset_hparams(self, rng, sample_input):
        rng_body, rng_head = jax.random.split(rng, 2)

        hparams_body, state_body = self.body.init(rng_body, sample_input, is_training=True)
        hparams_gain = {l: jnp.ones(h_dim) for l, h_dim in enumerate(self.hidden_dims)}
        hparams_shift = {l: jnp.zeros(h_dim) for l, h_dim in enumerate(self.hidden_dims)}

        head_dummy_input = jnp.empty((sample_input.shape[0], self.hidden_dims[-1]))
        hparams_head, state_head = self.head.init(rng_head, head_dummy_input, is_training=True)

        hparams = GainModMetaParams(
            body=hparams_body,
            gain_init=hparams_gain,
            head_init=hparams_head,
            shift_init=hparams_shift,
        )

        hstate = GainModMetaState(state_body, state_head)

        return hparams, hstate

    def reset_params(self, rng, hparams, hstate, sample_input):
        head_dummy_input = jnp.empty((sample_input.shape[0], self.hidden_dims[-1]))
        params_head_random, state_head = self.head.init(rng, head_dummy_input, is_training=True)

        if self.adapt_head:
            params_head = hparams.head_init
        else:
            params_head = dict()

        params = GainModParams(gain=hparams.gain_init, head=params_head, shift=hparams.shift_init)
        state = GainModState(head=state_head)

        return params, state
