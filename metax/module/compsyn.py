"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import Dict, NamedTuple

import jax.numpy as jnp
import jax.tree_util as jtu

from metax import energy

from .base import MetaModule


class ComplexSynapseMetaParams(NamedTuple):
    omega: Dict
    log_lambda: Dict


class ComplexSynapseParams(NamedTuple):
    base_learner: Dict


class ComplexSynapseMetaState(NamedTuple):
    # NOTE: base_learner state could also be seen as task-shared and handled here
    pass


class ComplexSynapseState(NamedTuple):
    base_learner: Dict


class ComplexSynapse(MetaModule):
    def __init__(self, loss_fn_inner, loss_fn_outer, base_learner, l2_reg, l2_fixed):
        super().__init__(loss_fn_inner=loss_fn_inner, loss_fn_outer=loss_fn_outer)
        self.base_learner = base_learner
        self.l2_reg = l2_reg
        self.l2_fixed = l2_fixed

        if self.l2_fixed:
            # iMAML-style regularizer
            self.loss_fn_inner += energy.iMAML(
                reg_strength=l2_reg,
                key_map={"base_learner": "omega"},
                reduction="sum"
            )
        else:
            # Meta-learn the reg-strength
            self.loss_fn_inner += energy.ComplexSynapse(
                key_map={"base_learner": {"omega": "omega", "log_lambda": "log_lambda"}},
                reduction="sum"
            )

    def __call__(self, rng, state, hstate, params, hparams, input, is_training):
        output, state = self.base_learner.apply(
            params.base_learner, state.base_learner, rng, input, is_training
        )

        return output, (ComplexSynapseState(state), hstate)

    def reset_hparams(self, rng, sample_input):
        omega, _ = self.base_learner.init(rng, sample_input, is_training=True)

        if self.l2_fixed:
            log_lambda = dict()
        else:
            log_lambda = jtu.tree_map(
                lambda x: jnp.log(self.l2_reg) * jnp.ones_like(x), omega
            )

        hparams = ComplexSynapseMetaParams(omega, log_lambda)
        hstate = ComplexSynapseMetaState()

        return hparams, hstate

    def reset_params(self, rng, hparams, hstate, sample_input):
        _, state_base_learner = self.base_learner.init(rng, sample_input, is_training=True)
        return ComplexSynapseParams(hparams.omega), ComplexSynapseState(state_base_learner)
