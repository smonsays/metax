"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import Dict, NamedTuple

from metax import energy

from .base import MetaModule


class LearnedInitMetaParams(NamedTuple):
    base_learner_init: Dict


class LearnedInitParams(NamedTuple):
    base_learner: Dict


class LearnedInitMetaState(NamedTuple):
    pass


class LearnedInitState(NamedTuple):
    base_learner: Dict


class LearnedInit(MetaModule):
    def __init__(self, loss_fn_inner, loss_fn_outer, base_learner, reg_strength):
        super().__init__(loss_fn_inner, loss_fn_outer)
        self.base_learner = base_learner

        if reg_strength is not None:
            # Use iMAML regularizer towards meta-learned init
            key_map = {"base_learner": "base_learner_init"}

            self.loss_fn_inner += energy.iMAML(
                reg_strength=reg_strength,
                key_map=key_map,
                reduction="sum"
            )

    def __call__(self, rng, state, hstate, params, hparams, input, is_training):
        output, state = self.base_learner.apply(
            params.base_learner, state.base_learner, rng, input, is_training
        )
        return output, (LearnedInitState(state), hstate)

    def reset_hparams(self, rng, sample_input):
        params_base_learner, _ = self.base_learner.init(rng, sample_input, is_training=True)

        # Re-using params container here to simplify implementation of reptile
        return LearnedInitMetaParams(params_base_learner), LearnedInitMetaState()

    def reset_params(self, rng, hparams, hstate, sample_input):
        _, state_base_learner = self.base_learner.init(rng, sample_input, is_training=True)

        return LearnedInitParams(hparams.base_learner_init), LearnedInitState(state_base_learner)
