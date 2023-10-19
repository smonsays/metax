"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import Dict, NamedTuple, Tuple

import chex
import jax
import optax

from .base import EnergyFunction
from .metrics import accuracy, accuracy_metrics


class CrossEntropy(EnergyFunction):
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
        loss = optax.softmax_cross_entropy_with_integer_labels(pred, target)
        loss = self._reduce(loss)

        return loss, {"acc": accuracy(pred, info.get("hard_targets", target)), "loss": loss}


class CrossEntropyMasked(EnergyFunction):
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
        loss = optax.softmax_cross_entropy_with_integer_labels(pred, target)
        loss = self._reduce(loss, info["mask"])

        acc_metrics = accuracy_metrics(pred, info.get("hard_targets", target), info["mask"])

        return loss, {**acc_metrics, "loss": loss}


class KLDivergence(EnergyFunction):
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
        loss = optax.convex_kl_divergence(jax.nn.log_softmax(pred), target)
        loss = self._reduce(loss)

        return loss, {"acc": accuracy(pred, info.get("hard_targets", target)), "loss": loss}


class SquaredError(EnergyFunction):
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
        loss = optax.l2_loss(pred, target)
        loss = self._reduce(loss)

        return loss, {"loss": loss}


class SquaredErrorMasked(EnergyFunction):
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
        loss = optax.l2_loss(pred, target)
        loss = self._reduce(loss, info["mask"][:, None])

        return loss, {"loss": loss}
