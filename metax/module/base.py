"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import abc
from typing import NamedTuple, Tuple

import chex

from metax.energy import EnergyFunction

Params = State = HParams = HState = NamedTuple  # implemented by each MetaModule


class MetaModule(abc.ABC):
    """
    Abstract base class defining meta modules.
    """
    def __init__(self, loss_fn_inner: EnergyFunction, loss_fn_outer: EnergyFunction) -> None:
        self.loss_fn_inner = loss_fn_inner
        self.loss_fn_outer = loss_fn_outer

    @abc.abstractmethod
    def __call__(
        self,
        rng: chex.Array,
        state: State,
        hstate: HState,
        params: Params,
        hparams: HParams,
        input: chex.Array,
    ) -> Tuple[chex.Array, Tuple[State, HState]]:
        pass

    @abc.abstractmethod
    def reset_hparams(self, rng: chex.Array, sample_input: chex.Array) -> Tuple[HParams, HState]:
        pass

    @abc.abstractmethod
    def reset_params(
        self, rng: chex.Array, hparams: HParams, sample_input: chex.Array
    ) -> Tuple[Params, State]:
        pass
