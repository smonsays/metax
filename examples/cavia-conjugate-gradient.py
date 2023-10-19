"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import argparse

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from tqdm import tqdm

import metax

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--embedding_dim", type=int, default=16)
parser.add_argument("--first_order", type=bool, default=False)
parser.add_argument("--lr_inner", type=float, default=0.01)
parser.add_argument("--lr_outer", type=float, default=0.001)
parser.add_argument("--meta_batch_size", type=int, default=100)
parser.add_argument("--num_tasks_test", type=int, default=100)
parser.add_argument("--num_tasks_train", type=int, default=10000)
parser.add_argument("--num_tasks_valid", type=int, default=10)
parser.add_argument("--shots_test", type=int, default=10)
parser.add_argument("--shots_train", type=int, default=10)
parser.add_argument("--steps_inner", type=int, default=10)
parser.add_argument("--steps_outer", type=int, default=100)
parser.add_argument("--seed", type=int, default=2022)
args = parser.parse_args()

# Create the data
metatrainset, metatestset, _, _, _ = metax.data.create_synthetic_metadataset(
    meta_batch_size=args.meta_batch_size,
    train_test_split=True,
    name="family",
    shots_train=args.shots_train,
    shots_test=args.shots_test,
    num_tasks_train=args.num_tasks_train,
    num_tasks_test=args.num_tasks_test,
    num_tasks_valid=args.num_tasks_valid,
)

# Define the loss, meta-model and meta-learning algorithm
meta_model = metax.module.CAVIA(
    loss_fn_inner=metax.energy.SquaredError(),
    loss_fn_outer=metax.energy.SquaredError(),
    base_model=hk.transform_with_state(lambda x, is_training: hk.nets.MLP([64, 64, metatrainset.output_dim])(x)),
    embedding_dim=args.embedding_dim,
)
meta_learner = metax.learner.ConjugateGradient(
    meta_model=meta_model,
    batch_size=args.batch_size,
    steps_inner=args.steps_inner,
    steps_cg=args.steps_inner,
    optim_fn_inner=optax.adamw(args.lr_inner),
    optim_fn_outer=optax.adamw(args.lr_outer),
)

# Initialize
rng = jax.random.PRNGKey(args.seed)
rng_reset, rng_train, rng_test = jax.random.split(rng, 3)
meta_state = meta_learner.reset(rng_reset, metatrainset.sample_input)
meta_update = jax.jit(meta_learner.update)
meta_eval = jax.jit(meta_learner.eval, static_argnames="steps")

# Train
with tqdm(metatrainset, desc="Train") as pbar:
    for idx, batch in enumerate(pbar):
        rng_update, rng_train = jax.random.split(rng_train)
        meta_state, metric_train = meta_update(rng_update, meta_state, batch)

        if (idx % 10) == 0:
            pbar.set_postfix(
                loss=f'{metric_train["loss_outer"]:.4f}',
            )

# Evaluate
metrics_test = []
rngs_eval = jax.random.split(rng_test, len(metatestset))
for (batch, rng_eval) in zip(metatestset, rngs_eval):
    _, metric_test = meta_eval(rng_eval, meta_state, batch, args.steps_inner)
    metrics_test.append(metric_test)

metrics_test = jtu.tree_map(lambda *args: jnp.stack((args)), *metrics_test)
print("test_loss_outer: ", jnp.mean(jnp.stack(metrics_test["loss_outer"])))
