"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import abc

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from metax.learner.base import MetaGradLearner
from metax.utils import append_keys


class ImplicitDifferentiation(MetaGradLearner):
    """
    Abstract base class for meta-learning methods based on implicit differentiation.
    NOTE: Implicit methods cannot meta-learn meta-parameters that only affect the
          optimization process (e.g. learning rates, initializations)
    """
    @abc.abstractmethod
    def inverse_hvp(self, inner_grad_fn, grads_outer_param, params):
        """
        Approximate the inverse parameter Hessian (combined with vector-product).
        """
        pass

    def grad(self, rng, hstate, hparams, metadataset):
        rng_reset, rng_adapt, rng_hvp, rng_jvp, rng_outer = jax.random.split(rng, 5)

        # Adapt the parameters on the inner loss
        params, state = self.meta_model.reset_params(
            rng_reset, hparams, hstate, metadataset.train.x
        )
        (state, params_adapted), metrics_inner = self.adapt(
            rng_adapt, state, hstate, params, hparams, metadataset.train, self.steps_inner
        )

        def loss_fn_outer(rng, params, hparams):
            """
            Outer loss fun closed over (h)state and data.
            """
            rng_pred, rng_loss = jax.random.split(rng)
            pred, (state_new, hstate_new) = self.meta_model(
                rng_pred, state, hstate, params, hparams, metadataset.test.x, is_training=True
            )
            loss, metrics = self.meta_model.loss_fn_outer(
                rng=rng_loss,
                pred=pred,
                target=metadataset.test.y,
                params=params,
                hparams=hparams,
                state=state_new,
                hstate=hstate_new
            )
            return loss, (hstate, metrics)

        def loss_fn_inner(rng, params, hparams):
            """
            Inner loss fun closed over (h)state and data.
            """
            rng_pred, rng_loss = jax.random.split(rng)
            pred, (state_new, hstate_new) = self.meta_model(
                rng_pred, state, hstate, params, hparams, metadataset.train.x, is_training=True
            )

            loss, _ = self.meta_model.loss_fn_inner(
                rng=rng_loss,
                pred=pred,
                target=metadataset.train.y,
                params=params,
                hparams=hparams,
                state=state_new,
                hstate=hstate_new,
            )

            return loss

        # Compute the outer gradients and update hstate
        (grads_outer_param, grads_outer_hparam_direct), (hstate, metrics_outer) = jax.grad(
            loss_fn_outer, argnums=(1, 2), has_aux=True)(rng_outer, params_adapted, hparams)

        # Compute the indirect gradient w.r.t. hyperparameters
        inner_grad_fn = jax.grad(lambda p: loss_fn_inner(rng_hvp, p, hparams))
        inv_hvp = self.inverse_hvp(inner_grad_fn, grads_outer_param, params_adapted)
        _, grads_outer_hparam_indirect = jax.jvp(
            fun=lambda p: jax.grad(lambda hp: loss_fn_inner(rng_jvp, p, hp))(hparams),
            primals=[params_adapted],
            tangents=[inv_hvp]
        )

        grads = jtu.tree_map(jnp.subtract, grads_outer_hparam_direct, grads_outer_hparam_indirect)

        metrics = {
            **append_keys(metrics_outer, "outer"),
            "gradnorm_outer": optax.global_norm(grads),
            **metrics_inner
        }

        return grads, hstate, metrics


class ConjugateGradient(ImplicitDifferentiation):

    def __init__(
        self,
        meta_model,
        batch_size,
        steps_inner,
        steps_cg,
        optim_fn_inner,
        optim_fn_outer,
        cg_damping=1.0,
        cg_reg=0.0,
        linsolver="cg",
    ):
        super().__init__(meta_model, batch_size, steps_inner, optim_fn_inner, optim_fn_outer)
        self.steps_cg = steps_cg
        self.cg_damping = cg_damping
        self.cg_reg = cg_reg
        self.linsolver = getattr(jax.scipy.sparse.linalg, linsolver)  # bicgstab, cg, gmres

    def inverse_hvp(self, inner_grad_fn, grads_outer_param, params):
        """
        Approximate the inverse solving Hx = g by minimizing 1/2 x^T H x - x^T g
        with H the parameter Hessian and g the gradient of the outer loss w.r.t. the parameters.

        NOTE: In principle x could be initialised differently, but both the official iMAML [0]
              and RBP [1] implementations also initialise with zero as we do here
              [0] https://github.com/aravindr93/imaml_dev/blob/master/examples/omniglot_implicit_maml.py#L125
              [1] https://github.com/lrjconan/RBP/blob/master/utils/model_helper.py#L26
        """
        _, hvp_fn = jax.linearize(inner_grad_fn, params)  # curried jvp

        def hvp_fn_dampened(tangents):
            def damping_fn(h, t):
                return self.cg_reg * t + h / self.cg_damping

            return jtu.tree_map(damping_fn, hvp_fn(tangents), tangents)

        inv_hvp, _ = self.linsolver(hvp_fn_dampened, grads_outer_param, maxiter=self.steps_cg)

        return inv_hvp


class RecurrentBackpropagation(ImplicitDifferentiation):
    def __init__(
        self,
        meta_model,
        batch_size,
        steps_inner,
        steps_rbp,
        optim_fn_inner,
        optim_fn_outer,
        alpha,
    ):
        super().__init__(meta_model, batch_size, steps_inner, optim_fn_inner, optim_fn_outer)
        self.steps_rbp = steps_rbp
        self.alpha = alpha

    def inverse_hvp(self, inner_grad_fn, grads_outer_param, params):
        def rbp_step(carry, _):
            vec, inv = carry
            _, hvp = jax.jvp(
                fun=inner_grad_fn,
                primals=[params],
                tangents=[vec]
            )
            vec = jtu.tree_map(lambda v, h: v - self.alpha * h, vec, hvp)
            inv = jtu.tree_map(jnp.add, inv, vec)

            return [vec, inv], ()

        carry, _ = jax.lax.scan(
            rbp_step, [grads_outer_param, grads_outer_param], jnp.arange(self.steps_rbp)
        )
        _, inv = carry

        return jtu.tree_map(lambda n: self.alpha * n, inv)


class T1T2(ImplicitDifferentiation):
    def inverse_hvp(self, inner_grad_fn, grads_outer_param, params):
        return grads_outer_param
