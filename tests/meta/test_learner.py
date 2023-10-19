"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import unittest

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

import metax
from metax import data


class MetaLearnerTestCase(unittest.TestCase):
    def setUp(self):
        self.steps_outer = 3
        self.meta_batch_size = 5

        def loss_fn(rng, pred, target, params, hparams, state, hstate, info=dict()):
            loss = (pred - target) ** 2
            loss = jnp.mean(loss)

            return loss, {"loss": loss}

        @hk.transform_with_state
        def body(inputs, is_training):
            return hk.nets.MLP([64, 64], activate_final=True)(inputs)

        self.meta_model = metax.module.AlmostNoInnerLoop(
            loss_fn_inner=loss_fn, loss_fn_outer=loss_fn, body=body, output_dim=1
        )

        self.metatrainset, self.metatestset, _, _, _ = data.create_synthetic_metadataset(
            meta_batch_size=self.meta_batch_size,
            train_test_split=True,
            name="family",
            shots_train=10,
            shots_test=10,
            num_tasks_train=100,
            num_tasks_test=100,
            num_tasks_valid=10,
        )
        self.dummy_input = self.metatrainset.sample_input
        self.rng = jax.random.PRNGKey(0)

    def train(self, meta_learner):
        rng_reset, rng_train = jax.random.split(self.rng, 2)
        meta_state = meta_learner.reset(rng_reset, self.dummy_input)

        metrics_list = []
        for r, meta_batch in zip(jax.random.split(rng_train, self.steps_outer), self.metatrainset):
            meta_state, m = meta_learner.update(r, meta_state, meta_batch)
            metrics_list.append(m)

        metrics = jtu.tree_map(lambda *args: jnp.stack((args)), *metrics_list)

        return meta_state, metrics

    def eval(self, meta_learner, meta_state):
        metrics_list = []
        for r, meta_batch in zip(jax.random.split(self.rng, len(self.metatestset)), self.metatrainset):
            _, m = meta_learner.eval(r, meta_state, meta_batch, steps=10)
            metrics_list.append(m)

        metrics = jtu.tree_map(lambda *args: jnp.stack((args)), *metrics_list)

        return metrics

    def train_eval_shapes(self, meta_learner):
        meta_state, metrics_train = self.train(meta_learner)
        metrics_eval = self.eval(meta_learner, meta_state)

        chex.assert_tree_shape_prefix(metrics_train, (self.steps_outer, ))
        chex.assert_tree_shape_prefix(metrics_eval, (1, ))

    def test_conjugate_gradient(self):
        meta_learner = metax.learner.ConjugateGradient(
            meta_model=self.meta_model,
            batch_size=3,
            steps_inner=10,
            steps_cg=7,
            optim_fn_inner=optax.adam(0.1),
            optim_fn_outer=optax.adam(0.01),
        )
        self.train_eval_shapes(meta_learner)

    def test_eqprop(self):
        meta_learner = metax.learner.EquilibriumPropagation(
            meta_model=self.meta_model,
            batch_size=3,
            steps_inner=10,
            steps_nudged=7,
            optim_fn_inner=optax.adam(0.1),
            optim_fn_nudged=optax.adam(0.1),
            optim_fn_outer=optax.adam(0.01),
            beta=1.0,
        )
        self.train_eval_shapes(meta_learner)

    def test_eqprop_symmetric(self):
        meta_learner = metax.learner.SymmetricEquilibriumPropagation(
            meta_model=self.meta_model,
            batch_size=3,
            steps_inner=10,
            steps_nudged=7,
            optim_fn_inner=optax.adam(0.1),
            optim_fn_nudged=optax.adam(0.1),
            optim_fn_outer=optax.adam(0.01),
            beta=1.0,
        )
        self.train_eval_shapes(meta_learner)

    def test_evolution(self):
        meta_learner = metax.learner.Evosax(
            meta_model=self.meta_model,
            batch_size=3,
            steps_inner=10,
            optim_fn_inner=optax.adam(0.1),
            input_shape=self.dummy_input.shape[-1:],
            algorithm="CMA_ES",
            evosax_kwargs=dict(popsize=8, elite_ratio=0.1)
        )
        self.train_eval_shapes(meta_learner)

    def test_maml(self):
        meta_learner = metax.learner.ModelAgnosticMetaLearning(
            meta_model=self.meta_model,
            batch_size=None,
            steps_inner=10,
            optim_fn_inner=optax.sgd(0.1),
            optim_fn_outer=optax.adam(0.01),
            first_order=True,
        )
        self.train_eval_shapes(meta_learner)

    def test_recurrent_backpropagation(self):
        meta_learner = metax.learner.RecurrentBackpropagation(
            meta_model=self.meta_model,
            batch_size=3,
            steps_inner=10,
            steps_rbp=7,
            optim_fn_inner=optax.adam(0.1),
            optim_fn_outer=optax.adam(0.01),
            alpha=0.0001,
        )
        self.train_eval_shapes(meta_learner)

    def test_reptile(self):
        base_learner = hk.transform_with_state(lambda x, is_training: hk.nets.MLP([64, 64, 1])(x))
        meta_model = metax.module.LearnedInit(
            self.meta_model.loss_fn_inner, self.meta_model.loss_fn_outer, base_learner, None
        )
        meta_learner = metax.learner.BufferedReptile(
            meta_model=meta_model,
            batch_size=3,
            steps_inner=10,
            optim_fn_inner=optax.adam(0.1),
            optim_fn_outer=optax.adam(0.01),
        )
        self.train_eval_shapes(meta_learner)

    def test_reptile_buffered(self):
        meta_learner = metax.learner.BufferedReptile(
            meta_model=self.meta_model,
            batch_size=3,
            steps_inner=10,
            optim_fn_inner=optax.adam(0.1),
            optim_fn_outer=optax.adam(0.01),
        )
        self.train_eval_shapes(meta_learner)

    def test_t1t2(self):
        meta_learner = metax.learner.T1T2(
            meta_model=self.meta_model,
            batch_size=3,
            steps_inner=10,
            optim_fn_inner=optax.adam(0.1),
            optim_fn_outer=optax.adam(0.01),
        )
        self.train_eval_shapes(meta_learner)
