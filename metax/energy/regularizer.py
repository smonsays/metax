"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import Any, Dict, List, Tuple

import chex
import jax.numpy as jnp
import jax.tree_util as jtu

from metax import utils

from .base import EnergyFunction


class ComplexSynapse(EnergyFunction):
    def __init__(self, key_map: Dict[str, Dict[str, str]], reduction: str = "mean") -> None:
        """
        Args:
            key_map: specifies the corresponding hparams to each param in a dict of dicts
                {"param_name": {"omega": "hparam_name_omega", "log_lambda": "hparam_name_lambda"}}
        """
        super().__init__(reduction)
        self.key_map = key_map

    @staticmethod
    def elem_fn(param: chex.Array, omega: chex.Array, log_lambda: chex.Array) -> chex.Array:
        return jnp.exp(log_lambda) * (param - omega) ** 2

    def __call__(
        self,
        rng: chex.PRNGKey,
        pred: chex.Array,
        target: chex.Array,
        params: Any,
        hparams: Any,
        state: Any,
        hstate: Any,
        info: Dict[str, Any] = dict(),
    ) -> Tuple[chex.Array, Dict[str, chex.Array]]:
        loss_tree = [
            jtu.tree_map(
                self.elem_fn,
                getattr(params, params_key),
                getattr(hparams, hparams_key_dict["omega"]),
                getattr(hparams, hparams_key_dict["log_lambda"]),
            )
            for params_key, hparams_key_dict in self.key_map.items()
        ]
        loss = self._reduce(utils.flatcat(loss_tree))

        return loss, {"loss_reg": loss}


class iMAML(EnergyFunction):
    def __init__(
        self, reg_strength: float, key_map: Dict[str, str], reduction: str = "mean"
    ) -> None:
        """
        Args:
            key_map: specifies the corresponding hparam to each param in a dict
                {"param_name": "hparam_name"}
        """
        super().__init__(reduction)
        self.reg_strength = reg_strength
        self.key_map = key_map

    @staticmethod
    def elem_fn(param: chex.Array, omega: chex.Array) -> chex.Array:
        return (param - omega) ** 2

    def __call__(
        self,
        rng: chex.PRNGKey,
        pred: chex.Array,
        target: chex.Array,
        params: Any,
        hparams: Any,
        state: Any,
        hstate: Any,
        info: Dict[str, Any] = dict(),
    ) -> Tuple[chex.Array, Dict[str, chex.Array]]:
        loss_tree = [
            jtu.tree_map(self.elem_fn, getattr(params, params_key), getattr(hparams, hparams_key))
            for params_key, hparams_key in self.key_map.items()
        ]
        loss = self.reg_strength * self._reduce(utils.flatcat(loss_tree))

        return loss, {"loss_reg": loss}


class LearnedL2(EnergyFunction):
    def __init__(self, key_map: Dict[str, str], reduction: str = "mean") -> None:
        """
        Args:
            key_map: specifies the corresponding hparam to each param in a dict
                {"param_name": "hparam_name"}
        """
        super().__init__(reduction)
        self.key_map = key_map

    @staticmethod
    def elem_fn(param: chex.Array, log_lambda: chex.Array) -> chex.Array:
        return jnp.exp(log_lambda) * (param) ** 2

    def __call__(
        self,
        rng: chex.PRNGKey,
        pred: chex.Array,
        target: chex.Array,
        params: Any,
        hparams: Any,
        state: Any,
        hstate: Any,
        info: Dict[str, Any] = dict(),
    ) -> Tuple[chex.Array, Dict[str, chex.Array]]:
        loss_tree = [
            jtu.tree_map(self.elem_fn, getattr(params, params_key), getattr(hparams, hparams_key))
            for params_key, hparams_key in self.key_map.items()
        ]

        loss = self._reduce(utils.flatcat(loss_tree))

        return loss, {"loss_reg": loss}


class LNorm(EnergyFunction):
    def __init__(
        self, reg_strength: float, param_keys: List[str], order: int, reduction: str = "mean"
    ) -> None:
        """
        Args:
            param_keys: `param_keys` to which regularization should be applied
            order: order of the norm to be applied
        """
        super().__init__(reduction)
        self.reg_strength = reg_strength
        self.param_keys = param_keys
        self.order = order

    def elem_fn(self, param: chex.Array) -> chex.Array:
        if self.order == 0:
            return param != 0
        elif self.order == 1:
            return jnp.abs(param)
        elif self.order == 2:
            return jnp.square(param)
        else:
            raise ValueError("Order {} not implemented".format(self.order))

    def __call__(
        self,
        rng: chex.PRNGKey,
        pred: chex.Array,
        target: chex.Array,
        params: Any,
        hparams: Any,
        state: Any,
        hstate: Any,
        info: Dict[str, Any] = dict(),
    ) -> Tuple[chex.Array, Dict[str, chex.Array]]:
        loss_tree = [jtu.tree_map(self.elem_fn, getattr(params, key)) for key in self.param_keys]

        loss = self.reg_strength * self._reduce(utils.flatcat(loss_tree))

        return loss, {"loss_reg": loss}
