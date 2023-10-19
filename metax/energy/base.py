"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from __future__ import annotations

import abc
from typing import NamedTuple, Optional, List, Tuple, Dict

import chex
import jax
import jax.numpy as jnp


class EnergyFunction(abc.ABC):
    """
    Abstract base class for modular loss functions that couple params and hparams.
    """

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    def __add__(self, other: EnergySum):
        return EnergySum(self, other)

    @abc.abstractmethod
    def __call__(
        self,
        rng: chex.PRNGKey,
        pred: chex.Array,
        target: chex.Array,
        params: NamedTuple,
        hparams: NamedTuple,
        state: NamedTuple,
        hstate: NamedTuple,
        info: Dict = dict(),
    ):
        pass

    def _reduce(self, value: chex.Array, mask: Optional[chex.Array] = None):
        if mask is not None:
            if self.reduction == "mean":
                return jnp.sum(mask * value) / jnp.sum(mask)
            elif self.reduction == "sum":
                return jnp.sum(mask * value)
            else:
                return mask * value
        else:
            if self.reduction == "mean":
                return jnp.mean(value)
            elif self.reduction == "sum":
                return jnp.sum(value)
            else:
                return value


class EnergySum(EnergyFunction):
    def __init__(self, *energies: List[EnergyFunction]) -> None:
        super().__init__()
        self.energy_list = energies

    def __call__(
        self,
        rng: chex.PRNGKey,
        pred: chex.Array,
        target: chex.Array,
        params: NamedTuple,
        hparams: NamedTuple,
        state: NamedTuple,
        hstate: NamedTuple,
        info: Dict = dict(),
    ) -> Tuple[chex.Array, Dict]:
        if rng is None:
            rngs_call = [None] * len(self.energy_list)
        else:
            rngs_call = jax.random.split(rng, len(self.energy_list))

        metrics = dict()
        loss = 0.0
        for energy, rng_call in zip(self.energy_list, rngs_call):
            l, m = energy(rng_call, pred, target, params, hparams, state, hstate, info)
            loss += l
            # NOTE: In case of using multiple losses with the same name in metric, this will produce
            #       undesired behaviour and overwrite loss metrics with the same name
            metrics.update(m)

        return loss, metrics
