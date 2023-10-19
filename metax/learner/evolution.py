"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from metax import utils
from metax.module import MetaModule

from .base import MetaLearnerInnerGradientDescent, MetaLearnerState


class Evosax(MetaLearnerInnerGradientDescent):
    def __init__(
        self,
        meta_model: MetaModule,
        batch_size: int,
        steps_inner: int,
        optim_fn_inner: optax.GradientTransformation,
        input_shape: Tuple[int],
        algorithm: str,
        evosax_kwargs: dict,
    ):
        super().__init__(meta_model, batch_size, steps_inner, optim_fn_inner)

        # Instantiate reshaper for hparams
        hparams_shape_dtype, _ = jax.eval_shape(
            self.meta_model.reset_hparams, jax.random.PRNGKey(0), jnp.empty((1, *input_shape))
        )
        self.hparam_reshaper = utils.PytreeReshaper(jtu.tree_map(jnp.shape, hparams_shape_dtype))
        self.population_reshape = jax.vmap(self.hparam_reshaper)
        self.population_eval = jax.vmap(  # NOTE: We share the state across the population and tasks
            fun=jax.vmap(
                fun=self.outer_loss,
                in_axes=(0, None, None, 0, None)
            ),
            in_axes=(None, None, 0, None, None)
        )

        # Instantiate the search strategy
        import evosax
        self.strategy = getattr(evosax, algorithm)(
            num_dims=self.hparam_reshaper.num_elements,
            **evosax_kwargs,
        )
        self.es_params = self.strategy.default_params

    def reset(self, rng, sample_input):
        rng_optim, rng_state = jax.random.split(rng)

        hparams_init, hstate = self.meta_model.reset_hparams(rng_state, sample_input)
        optim_state = self.strategy.initialize(rng_optim, self.es_params)

        return MetaLearnerState(hparams=hparams_init, optim=optim_state, hstate=hstate)

    def update(self, rng, meta_state, metadataset):

        rng_gen, rng_eval = jax.random.split(rng, 2)

        # Generate a new population of hparams
        hparams_population_flat, optim_state = self.strategy.ask(
            rng_gen, meta_state.optim, self.es_params
        )
        hparams_population = self.population_reshape(hparams_population_flat)

        # Evaluate every member of the population on each task
        rng_eval = jax.random.split(rng_eval, len(metadataset.train.x))
        loss, ((_, _, hstate_population), metrics_inner) = self.population_eval(
            rng_eval, meta_state.hstate, hparams_population, metadataset, self.steps_inner
        )

        # Report the batch fitness of each member
        batch_loss = jnp.mean(loss, axis=1)
        optim_state = self.strategy.tell(hparams_population_flat, batch_loss, optim_state, self.es_params)

        # Get best population member and its associated metrics
        best_idx = jnp.argmin(batch_loss)
        hparams = jtu.tree_map(lambda x: x[best_idx], hparams_population)
        hstate_all_tasks = jtu.tree_map(lambda x: x[best_idx], hstate_population)

        # HACK: Average hstate across tasks
        hstate_dtypes = jtu.tree_map(jnp.dtype, hstate_all_tasks)
        hstate = jtu.tree_map(partial(jnp.mean, axis=0), hstate_all_tasks)
        hstate = jtu.tree_map(jax.lax.convert_element_type, hstate, hstate_dtypes)

        metrics = {
            "loss_outer": jnp.mean(loss[best_idx], axis=0),
            **jtu.tree_map(lambda x: jnp.mean(x[best_idx], axis=0), metrics_inner)
        }

        return MetaLearnerState(hparams=hparams, optim=optim_state, hstate=hstate), metrics
