"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from metax.data import Dataset, batch_generator
from metax.module import LearnedInit
from metax.module.init import LearnedInitMetaParams
from metax.utils import append_keys

from .base import MetaGradLearner


class Reptile(MetaGradLearner):
    def __init__(
        self,
        meta_model,
        batch_size,
        steps_inner,
        optim_fn_inner,
        optim_fn_outer,
    ):
        assert isinstance(meta_model, LearnedInit)
        super().__init__(meta_model, batch_size, steps_inner, optim_fn_inner, optim_fn_outer)

    def grad(self, rng, hstate, hparams, metadataset):
        rng_adapt, rng_eval, rng_loss, rng_reset = jax.random.split(rng, 4)

        # Combine train and test data of metadataset
        dataset = jtu.tree_map(
            lambda x, y: jnp.concatenate((x, y), axis=0), metadataset.train, metadataset.test
        )

        # Adapt the parameters on the full data
        params, state = self.meta_model.reset_params(rng_reset, hparams, hstate, dataset.x)
        (state, params), metrics_inner = self.adapt(
            rng_adapt, state, hstate, params, hparams, dataset, self.steps_inner
        )

        grads = jtu.tree_map(jnp.subtract, hparams, LearnedInitMetaParams(params.base_learner))

        # Evaluate on outer loss for logging and updating hstate
        pred, (state, hstate) = self.meta_model(
            rng_eval, state, hstate, params, hparams, metadataset.test.x, is_training=True
        )
        _, metrics_outer = self.meta_model.loss_fn_outer(
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
            "gradnorm_outer": optax.global_norm(grads),
            **metrics_inner,
            **append_keys(metrics_outer, "outer"),
        }

        return grads, hstate, metrics


class BufferedReptile(MetaGradLearner):
    """
    Reptile-like meta-learner that accumulates partial hgrads during adaptation.
    """
    def adapt_buffered(self, rng, state, hstate, params, hparams, dataset, steps):
        def inner_step(carry, batch):
            rng, state, params, hgrads_buffer, optim = carry
            rng_next, rng_loss = jax.random.split(rng)
            (grads, hgrads), ((state, _), metrics) = jax.grad(self.inner_loss, argnums=(3, 4), has_aux=True)(
                rng_loss, state, hstate, params, hparams, batch
            )

            hgrads_buffer = jtu.tree_map(jnp.add, hgrads, hgrads_buffer)

            params_update, optim = self.optim_fn_inner.update(grads, optim, params)
            params = optax.apply_updates(params, params_update)

            return [rng_next, state, params, hgrads_buffer, optim], {**metrics, "gradnorm": optax.global_norm(grads)}

        rng_batch, rng_inner = jax.random.split(rng)
        optim_init = self.optim_fn_inner.init(params)
        batch_loader = batch_generator(
            rng_batch, dataset, steps, self.batch_size
        )
        hgrads_buffer = jtu.tree_map(jnp.zeros_like, hparams)
        carry, metrics = jax.lax.scan(
            inner_step, [rng_inner, state, params, hgrads_buffer, optim_init], batch_loader
        )
        _, state, params, hgrads_buffer, _ = carry

        return hgrads_buffer, ((state, params), append_keys(metrics, "inner"))

    def grad(self, rng, hstate, hparams, metadataset):
        rng_adapt, rng_eval, rng_loss, rng_reset = jax.random.split(rng, 4)

        # Combine train and test data of metadataset
        dataset = Dataset(
            x=jnp.concatenate((metadataset.train.x, metadataset.test.x), axis=0),
            y=jnp.concatenate((metadataset.train.y, metadataset.test.y), axis=0),
        )

        # Adapt the parameters on the full data
        params, state = self.meta_model.reset_params(rng_reset, hparams, hstate, dataset.x)
        grads, ((state, params), metrics_inner) = self.adapt_buffered(
            rng_adapt, state, hstate, params, hparams, dataset, self.steps_inner
        )

        # Evaluate on outer loss for logging and updating hstate
        pred, (state, hstate) = self.meta_model(
            rng_eval, state, hstate, params, hparams, metadataset.test.x, is_training=True
        )
        _, metrics_outer = self.meta_model.loss_fn_outer(
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
            "gradnorm_outer": optax.global_norm(grads),
            **metrics_inner,
            **append_keys(metrics_outer, "outer"),
        }

        return grads, hstate, metrics
