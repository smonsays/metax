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

from metax import data
from metax.utils import append_keys

from .base import MetaGradLearner, MetaLearnerState


class EquilibriumPropagation(MetaGradLearner):
    def __init__(
        self,
        meta_model,
        batch_size,
        steps_inner,
        steps_nudged,
        optim_fn_inner,
        optim_fn_nudged,
        optim_fn_outer,
        beta,
    ):
        super().__init__(
            meta_model, batch_size, steps_inner, optim_fn_inner, optim_fn_outer
        )
        self.optim_fn_nudged = optim_fn_nudged
        self.steps_nudged = steps_nudged
        self.beta = beta
        self.augmented_loss_grad = jax.grad(self.augmented_loss, argnums=3, has_aux=True)
        self.augmented_loss_hgrad = jax.grad(self.augmented_loss, 4, has_aux=True)

    def augmented_loss(self, rng, state, hstate, params, hparams, batch: data.MetaDataset, beta):
        rng_cost, rng_energy, rng_test, rng_train = jax.random.split(rng, 4)

        # Get prediction on both test and train in one meta_model call to get a single new state
        batch_train_test_x = jnp.concatenate((batch.train.x, batch.test.x))
        pred_train_test, (state, hstate) = self.meta_model(
            rng_train, state, hstate, params, hparams, batch_train_test_x, is_training=True
        )
        pred_train, pred_test = jnp.split(pred_train_test, [len(batch.train.x)])

        energy, metrics_energy = self.meta_model.loss_fn_inner(
            rng=rng_energy,
            pred=pred_train,
            target=batch.train.y,
            params=params,
            hparams=hparams,
            state=state,
            hstate=hstate
        )

        cost, metrics_cost = self.meta_model.loss_fn_outer(
            rng=rng_cost,
            pred=pred_test,
            target=batch.test.y,
            params=params,
            hparams=hparams,
            state=state,
            hstate=hstate,
        )

        loss = energy + beta * cost

        metrics = {
            "loss": loss,
            **append_keys(metrics_energy, "energy"),
            **append_keys(metrics_cost, "cost")
        }

        return loss, ((state, hstate), metrics)

    def adapt_augmented(self, rng, state, hstate, params, hparams, metadataset, beta):

        def augmented_step(carry, batch_train_test):
            rng, state, params, optim = carry
            rng_next, rng_grad = jax.random.split(rng)
            grads, ((state, _), metrics) = self.augmented_loss_grad(
                rng_grad, state, hstate, params, hparams, data.MetaDataset(*batch_train_test), beta
            )
            params_update, optim = self.optim_fn_nudged.update(grads, optim, params)
            params = optax.apply_updates(params, params_update)

            metrics = append_keys({**metrics, "gradnorm": optax.global_norm(grads)}, "nudged")

            return [rng_next, state, params, optim], metrics

        rng_batch_train, rng_batch_test, rng_scan = jax.random.split(rng, 3)
        optim_init = self.optim_fn_nudged.init(params)

        # NOTE: We cannot call batch_generator on the whole metadataset due to potentially
        #       differing number of samples in train and test set
        train_loader = data.batch_generator(
            rng_batch_train, metadataset.train, self.steps_nudged, self.batch_size
        )
        # NOTE: When batch_size > len(metadataset.x_test), this will dupilicate samples
        #       within a single batch.
        test_loader = data.batch_generator(
            rng_batch_test, metadataset.test, self.steps_nudged, self.batch_size
        )

        carry, metrics = jax.lax.scan(
            f=augmented_step,
            init=[rng_scan, state, params, optim_init],
            xs=(train_loader, test_loader)
        )
        _, state, params, _ = carry

        return (state, params), metrics

    def grad(self, rng, hstate, hparams, metadataset):
        (
            rng_eval,
            rng_free,
            rng_grad_f,
            rng_grad_n,
            rng_loss,
            rng_nudged,
            rng_reset,
        ) = jax.random.split(rng, 7)

        # Free phase
        params_init, state = self.meta_model.reset_params(
            rng_reset, hparams, hstate, metadataset.train.x
        )
        (state_free, params_free), metrics_free = self.adapt(
            rng_free, state, hstate, params_init, hparams, metadataset.train, self.steps_inner
        )
        # Nudged phase
        (state_nudged, params_nudged), metrics_nudged = self.adapt_augmented(
            rng_nudged, state, hstate, params_free, hparams, metadataset, self.beta
        )

        # Evaluate partial derivatives of augmented loss wrt hparams
        # NOTE: We evaluate on the full data, not a single batch
        hgrads_free, _ = self.augmented_loss_hgrad(
            rng_grad_f, state_free, hstate, params_free, hparams, metadataset, beta=0.0
        )
        hgrads_nudged, _ = self.augmented_loss_hgrad(
            rng_grad_n, state_nudged, hstate, params_nudged, hparams, metadataset, beta=self.beta
        )

        # Compute the meta-gradient
        def ep_first_order(hg_free, hg_nudged):
            return (1.0 / self.beta) * (hg_nudged - hg_free)

        grads = jtu.tree_map(ep_first_order, hgrads_free, hgrads_nudged)

        # Evaluate the pure outer loss to obtain the new model state
        # NOTE: hstate is updated here
        pred, (state, hstate) = self.meta_model(
            rng_eval, state, hstate, params_free, hparams, metadataset.test.x, is_training=True
        )
        loss_outer, metrics_outer = self.meta_model.loss_fn_outer(
            rng=rng_loss,
            pred=pred,
            target=metadataset.test.y,
            params=params_free,
            hparams=hparams,
            state=state,
            hstate=hstate,
            info=metadataset.test.info,
        )

        metrics = {
            **append_keys({**metrics_outer, "gradnorm": optax.global_norm(grads)}, "outer"),
            **metrics_free,
            **metrics_nudged
        }

        return grads, hstate, metrics


class SymmetricEquilibriumPropagation(EquilibriumPropagation):

    def grad(self, rng, hstate, hparams, metadataset):
        (
            rng_eval,
            rng_free,
            rng_grad_n,
            rng_grad_p,
            rng_loss,
            rng_nudged_neg,
            rng_nudged_pos,
            rng_reset,
        ) = jax.random.split(rng, 8)

        # Free phase
        params_init, state = self.meta_model.reset_params(
            rng_reset, hparams, hstate, metadataset.train.x
        )
        (state_free, params_free), metrics_free = self.adapt(
            rng_free, state, hstate, params_init, hparams, metadataset.train, self.steps_inner
        )

        # Nudged phases
        (state_nudged_pos, params_nudged_pos), metrics_nudged_pos = self.adapt_augmented(
            rng_nudged_pos, state_free, hstate, params_free, hparams, metadataset, self.beta
        )
        (state_nudged_neg, params_nudged_neg), metrics_nudged_neg = self.adapt_augmented(
            rng_nudged_neg, state_free, hstate, params_free, hparams, metadataset, -self.beta
        )

        # Evaluate partial derivatives of augmented loss wrt hparams
        # NOTE: We evaluate on the full data, not a single batch
        grads_nudged_pos, _ = self.augmented_loss_hgrad(
            rng_grad_p, state_nudged_pos, hstate, params_nudged_pos, hparams, metadataset, beta=self.beta
        )

        grads_nudged_neg, _ = self.augmented_loss_hgrad(
            rng_grad_n, state_nudged_neg, hstate, params_nudged_neg, hparams, metadataset, beta=-self.beta
        )

        # Compute the meta-gradient
        def ep_symmetric(g_nudged_pos, g_nudged_neg):
            return (1 / (2.0 * self.beta)) * (g_nudged_pos - g_nudged_neg)

        grads = jtu.tree_map(ep_symmetric, grads_nudged_pos, grads_nudged_neg)

        # Evaluate the pure outer loss to obtain the new model state
        # NOTE: hstate is updated here
        pred, (state, hstate) = self.meta_model(
            rng_eval, state_free, hstate, params_free, hparams, metadataset.test.x, is_training=True
        )
        loss_outer, metrics_outer = self.meta_model.loss_fn_outer(
            rng=rng_loss,
            pred=pred,
            target=metadataset.test.y,
            params=params_free,
            hparams=hparams,
            state=state,
            hstate=hstate,
            info=metadataset.test.info,
        )

        metrics = {
            **append_keys({**metrics_outer, "gradnorm": optax.global_norm(grads)}, "outer"),
            **metrics_free,
            **metrics_nudged_pos,
            **append_keys(metrics_nudged_neg, "neg"),
        }

        return grads, hstate, metrics
