"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from metax import data, utils
from metax.utils import append_keys

from .base import MetaLearner, MetaLearnerState


class ModelAgnosticMetaLearning(MetaLearner):
    def __init__(
        self, meta_model, batch_size, steps_inner, optim_fn_inner, optim_fn_outer, first_order
    ):
        super().__init__(meta_model)
        self.batch_size = batch_size  # batch_size=None performs full GD (instead of SGD)
        self.steps_inner = steps_inner
        self.optim_fn_inner = optim_fn_inner
        self.optim_fn_outer = optim_fn_outer
        self.first_order = first_order

    def adapt(self, rng, state, hstate, params, hparams, dataset, steps):
        """
        Adapts params and state on dataset given hparams and hstate.
        """
        log_metric = dict()

        grads, (_, metrics_init) = jax.grad(self.inner_loss, argnums=3, has_aux=True)(
            rng, state, hstate, params, hparams, dataset
        )
        # Initial inner loop metric
        log_metric.update(append_keys({**metrics_init, "gradnorm": optax.global_norm(grads)}, "inner_init"))

        def inner_step(carry, step):
            rng, state, params, optim = carry
            rng_next, rng_loss, rng_batch = jax.random.split(rng, 3)

            # Sample the minibatch here to avoid creating huge tensors
            if self.batch_size is None:
                batch = dataset
            else:
                batch = data.get_batch(rng_batch, dataset, self.batch_size)

            grads, ((state, _), metrics) = jax.grad(self.inner_loss, argnums=3, has_aux=True)(
                rng_loss, state, hstate, params, hparams, batch
            )

            if self.first_order:
                grads = jax.lax.stop_gradient(grads)

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
        log_metric.update(append_keys({**metrics_final, "gradnorm": optax.global_norm(grads)}, "inner_final"))

        return (state, params), log_metric

    def update(self, rng, meta_state, metadataset: data.MetaDataset):
        def batch_outer_loss(rng, hstate, hparams, metadataset):
            rngs = jax.random.split(rng, utils.tree_length(metadataset))
            outer_loss_vmap = jax.vmap(self.outer_loss, in_axes=(0, None, None, 0, None))
            loss, ((_, _, hstate), metrics) = outer_loss_vmap(
                rngs, hstate, hparams, metadataset, self.steps_inner
            )
            return jnp.mean(loss), (hstate, metrics)

        grad_fn = jax.grad(batch_outer_loss, argnums=2, has_aux=True)
        hgrads, (hstate, metrics) = grad_fn(
            rng, meta_state.hstate, meta_state.hparams, metadataset
        )

        # HACK: Averaging over the model state might result in unexpected behaviour
        # HACK: Averaging might change dtype (e.g. int to float), this simply casts it back
        hstate_dtypes = jtu.tree_map(jnp.dtype, hstate)
        hstate = jtu.tree_map(partial(jnp.mean, axis=0), hstate)
        hstate = jtu.tree_map(jax.lax.convert_element_type, hstate, hstate_dtypes)

        hparams_update, optim_state = self.optim_fn_outer.update(
            hgrads, meta_state.optim, meta_state.hparams
        )
        hparams = optax.apply_updates(meta_state.hparams, hparams_update)

        metrics = {
            "gradnorm_outer": optax.global_norm(hgrads),
            **jtu.tree_map(partial(jnp.mean, axis=0), metrics),
        }

        return MetaLearnerState(hparams=hparams, optim=optim_state, hstate=hstate), metrics
