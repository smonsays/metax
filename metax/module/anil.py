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

from metax import models

from .base import MetaModule


class AlmostNoInnerLoopMetaParams(NamedTuple):
    body: hk.Params
    head_init: hk.Params


class AlmostNoInnerLoopParams(NamedTuple):
    head: hk.Params


class AlmostNoInnerLoopMetaState(NamedTuple):
    body: hk.State


class AlmostNoInnerLoopState(NamedTuple):
    head: hk.State


class AlmostNoInnerLoop(MetaModule):
    def __init__(self, loss_fn_inner, loss_fn_outer, body, output_dim):
        super().__init__(loss_fn_inner, loss_fn_outer)

        self.body = body
        self.head = models.Linear(output_dim)
        self.output_dim = output_dim

    def __call__(self, rng, state, hstate, params, hparams, input, is_training):
        rng_body, rng_head = jax.random.split(rng)

        features, state_body = self.body.apply(
            hparams.body, hstate.body, rng_body, input, is_training
        )
        output, state_head = self.head.apply(
            params.head, state.head, rng_head, features, is_training
        )

        hstate = AlmostNoInnerLoopMetaState(body=state_body)
        state = AlmostNoInnerLoopState(head=state_head)

        return output, (state, hstate)

    def reset_hparams(self, rng, sample_input):
        rng_body, rng_head = jax.random.split(rng)

        hparams_body, state_body = self.body.init(rng_body, sample_input, is_training=True)
        head_sample_input, _ = self.body.apply(
            hparams_body, state_body, rng_body, sample_input, is_training=True
        )
        hparams_head, _ = self.head.init(rng_head, head_sample_input, is_training=True)

        hparams = AlmostNoInnerLoopMetaParams(body=hparams_body, head_init=hparams_head)
        hstate = AlmostNoInnerLoopMetaState(body=state_body)

        return hparams, hstate

    def reset_params(self, rng, hparams, hstate, sample_input):
        return AlmostNoInnerLoopParams(head=hparams.head_init), AlmostNoInnerLoopState(head={})
