"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import abc
from functools import partial
from typing import Dict, NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from metax import data, utils
from metax.data import Dataset, MetaDataset
from metax.module.base import Params, State, HParams, HState, MetaModule
from metax.utils import append_keys


class MetaLearnerState(NamedTuple):
    hparams: HParams
    hstate: HState
    optim: optax.OptState


class MetaLearner(abc.ABC):
    """
    Abstract base class for meta-learning algorithms wrapping a meta model.
    """

    def __init__(self, meta_model: MetaModule):
        self.meta_model = meta_model

    @abc.abstractmethod
    def adapt(
        self,
        rng: chex.PRNGKey,
        state: State,
        hstate: HState,
        params: Params,
        hparams: HParams,
        dataset: Dataset,
        steps: int,
    ) -> Tuple[Tuple[State, Params], Dict]:
        pass

    def eval(
        self,
        rng: chex.PRNGKey,
        meta_state: MetaLearnerState,
        metadataset: data.MetaDataset,
        steps: int,
    ) -> Tuple[Tuple[State, Params], Dict]:
        """
        Evaluate the meta_model on the given metadataset
        """
        outer_loss_batched = jax.vmap(
            partial(self.outer_loss, steps_adapt=steps), in_axes=(0, None, None, 0)
        )
        rngs = jax.random.split(rng, utils.tree_length(metadataset))
        _, (((state, params), _, _), metrics) = outer_loss_batched(
            rngs, meta_state.hstate, meta_state.hparams, metadataset
        )
        metrics = jtu.tree_map(partial(jnp.mean, axis=0), metrics)

        return (state, params), metrics

    def inner_loss(
        self,
        rng: chex.PRNGKey,
        state: State,
        hstate: HState,
        params: Params,
        hparams: HParams,
        batch: data.Dataset,
    ) -> Tuple[chex.Array, Tuple[Tuple[State, HState], Dict]]:
        rng_pred, rng_loss = jax.random.split(rng)

        pred, (state, hstate) = self.meta_model(
            rng_pred, state, hstate, params, hparams, batch.x, is_training=True
        )

        loss, metrics = self.meta_model.loss_fn_inner(
            rng=rng_loss,
            pred=pred,
            target=batch.y,
            params=params,
            hparams=hparams,
            state=state,
            hstate=hstate,
            info=batch.info,
        )

        return loss, ((state, hstate), metrics)

    def outer_loss(
        self,
        rng: chex.PRNGKey,
        hstate: HState,
        hparams: HParams,
        metadataset: MetaDataset,
        steps_adapt: int,
    ) -> Tuple[chex.Array, Tuple[Tuple[Tuple[State, HState], Tuple[State, Params], HState], Dict]]:
        rng_adapt, rng_loss, rng_pred, rng_reset = jax.random.split(rng, 4)
        params_init, state_init = self.meta_model.reset_params(
            rng_reset, hparams, hstate, metadataset.train.x
        )
        (state, params), metrics_inner = self.adapt(
            rng_adapt, state_init, hstate, params_init, hparams, metadataset.train, steps_adapt
        )
        # NOTE: is_training=True in combination with batch_norm would make this transductive,
        #       i.e. p(answer | query, all_other_points_in_query).
        # NOTE: If is_training=True, batch_norm stats accumulated during training
        #       will not be used regardless of the decay rate.
        pred, (state, hstate) = self.meta_model(
            rng_pred, state, hstate, params, hparams, metadataset.test.x, is_training=False
        )
        loss, metrics_outer = self.meta_model.loss_fn_outer(
            rng=rng_loss,
            pred=pred,
            target=metadataset.test.y,
            params=params,
            hparams=hparams,
            state=state,
            hstate=hstate,
            info=metadataset.test.info,
        )

        metrics = {
            **append_keys(metrics_outer, "outer"),
            **metrics_inner,
        }
        aux = ((state, params), (state_init, params_init), hstate)

        return loss, (aux, metrics)

    def reset(self, rng: chex.PRNGKey, sample_input: chex.Array) -> MetaLearnerState:
        (rng_init,) = jax.random.split(rng, 1)

        hparams_init, hstate_init = self.meta_model.reset_hparams(rng_init, sample_input)
        optim_state_init = self.optim_fn_outer.init(hparams_init)

        return MetaLearnerState(hparams=hparams_init, optim=optim_state_init, hstate=hstate_init)

    @abc.abstractmethod
    def update(
        self, rng: chex.PRNGKey, meta_state: MetaLearnerState, metadataset: data.MetaDataset
    ) -> Tuple[MetaLearnerState, Dict]:
        """
        Update the meta_state - primarily the meta-parameters - given the meta-dataset.
        """
        pass


class MetaLearnerInnerGradientDescent(MetaLearner):
    """
    Abstract base class for meta-learning algorithms that use gradient descent in the inner loop.
    """
    def __init__(
        self,
        meta_model: MetaModule,
        batch_size: int,
        steps_inner: int,
        optim_fn_inner: optax.GradientTransformation,
    ):
        super().__init__(meta_model)
        self.optim_fn_inner = optim_fn_inner
        self.batch_size = batch_size  # batch_size=None performs full GD (instead of SGD)
        self.steps_inner = steps_inner

    def adapt(self, rng, state, hstate, params, hparams, dataset, steps):
        """
        Adapts params and state using optim_fn_inner on dataset given hparams and hstate.
        """
        log_metric = dict()
        grads, (_, metrics_init) = jax.grad(self.inner_loss, argnums=3, has_aux=True)(
            rng, state, hstate, params, hparams, dataset
        )
        # Initial inner loop metric
        log_metric.update(
            append_keys({**metrics_init, "gradnorm": optax.global_norm(grads)}, "inner_init")
        )

        def inner_step(carry, step):
            # NOTE: inner_step closes over hstate and hparams
            rng, state, params, optim = carry
            rng_next, rng_loss, rng_batch = jax.random.split(rng, 3)

            if self.batch_size is None:
                batch = dataset
            else:
                batch = data.get_batch(rng_batch, dataset, self.batch_size)

            grads, ((state, _), _) = jax.grad(self.inner_loss, argnums=3, has_aux=True)(
                rng_loss, state, hstate, params, hparams, batch
            )

            params_update, optim = self.optim_fn_inner.update(grads, optim, params)
            params = optax.apply_updates(params, params_update)

            carry = [rng_next, state, params, optim]

            return carry, None

        optim_init = self.optim_fn_inner.init(params)
        (_, state, params, _), _ = jax.lax.scan(
            inner_step, [rng, state, params, optim_init], jnp.arange(steps)
        )
        # Final inner loop metric
        grads, (_, metrics_final) = jax.grad(self.inner_loss, argnums=3, has_aux=True)(
            rng, state, hstate, params, hparams, dataset
        )
        log_metric.update(
            append_keys({**metrics_final, "gradnorm": optax.global_norm(grads)}, "inner_final")
        )

        return (state, params), log_metric


class MetaGradLearner(MetaLearnerInnerGradientDescent):
    """
    Abstract base class for meta-learning algorithms that estimate the meta-gradient.
    """

    def __init__(
        self,
        meta_model: MetaModule,
        batch_size: int,
        steps_inner: int,
        optim_fn_inner: optax.GradientTransformation,
        optim_fn_outer: optax.GradientTransformation,
    ):
        super().__init__(meta_model, batch_size, steps_inner, optim_fn_inner)
        self.optim_fn_outer = optim_fn_outer

        self.batch_grad = jax.vmap(self.grad, in_axes=(0, None, None, 0))

    @abc.abstractmethod
    def grad(
        self, rng: chex.PRNGKey, hstate: HState, hparams: HParams, metadataset: data.MetaDataset
    ) -> Tuple[chex.Array, HState, Dict]:
        pass

    def update(self, rng, meta_state, metadataset: data.MetaDataset):
        rng_batch = jax.random.split(rng, len(metadataset.train.x))
        hgrads, hstate, metrics = self.batch_grad(
            rng_batch, meta_state.hstate, meta_state.hparams, metadataset
        )

        hgrads = jtu.tree_map(partial(jnp.mean, axis=0), hgrads)  # Average hgrads across tasks
        hparams_update, optim_state = self.optim_fn_outer.update(
            hgrads, meta_state.optim, meta_state.hparams
        )
        hparams = optax.apply_updates(meta_state.hparams, hparams_update)

        # HACK: Averaging over the model state might result in unexpected behaviour
        # HACK: Averaging might change dtype (e.g. int to float), this simply casts it back
        hstate_dtypes = jtu.tree_map(jnp.dtype, hstate)
        hstate = jtu.tree_map(partial(jnp.mean, axis=0), hstate)
        hstate = jtu.tree_map(jax.lax.convert_element_type, hstate, hstate_dtypes)
        metrics = jtu.tree_map(partial(jnp.mean, axis=0), metrics)

        return MetaLearnerState(hparams=hparams, optim=optim_state, hstate=hstate), metrics
