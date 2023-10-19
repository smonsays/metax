"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import argparse

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jax_meta.datasets import Omniglot

import metax
from metax.data.base import DATAPATH, Dataset, MetaDataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--bn_decay", type=float, default=0.9)
parser.add_argument("--channels", type=int, default=64)
parser.add_argument("--num_tasks_test", type=int, default=100)
parser.add_argument("--num_tasks_train", type=int, default=10000)
parser.add_argument("--num_tasks_valid", type=int, default=10)
parser.add_argument("--ways", type=int, default=5)
parser.add_argument("--shots_test", type=int, default=10)
parser.add_argument("--shots_train", type=int, default=10)
parser.add_argument("--first_order", type=bool, default=False)
parser.add_argument("--lr_inner", type=float, default=0.4)
parser.add_argument("--lr_outer", type=float, default=0.001)
parser.add_argument("--meta_batch_size", type=int, default=16)
parser.add_argument("--steps_inner", type=int, default=1)
parser.add_argument("--steps_outer", type=int, default=100)
parser.add_argument("--seed", type=int, default=2022)
args = parser.parse_args()


# Load data from [jax_meta](https://github.com/tristandeleu/jax-meta-learning)
metaloader = Omniglot(
    DATAPATH,
    batch_size=args.meta_batch_size,
    shots=args.shots_train,
    ways=args.ways,
)

metaloader.input_shape = metaloader.shape
metaloader.output_dim = metaloader.ways
metaloader.sample_input = jnp.array(metaloader.dummy_input)

# Define the loss, meta-model and meta-learning algorithm
base_model = metax.models.Conv4(args.channels, args.bn_decay, readout=args.ways)
meta_model = metax.module.LearnedInit(
    loss_fn_inner=metax.energy.CrossEntropy(),
    loss_fn_outer=metax.energy.CrossEntropy(),
    base_learner=base_model,
    reg_strength=None
)
meta_learner = metax.learner.ModelAgnosticMetaLearning(
    meta_model=meta_model,
    batch_size=args.batch_size,
    steps_inner=args.steps_inner,
    optim_fn_inner=optax.sgd(args.lr_inner),
    optim_fn_outer=optax.adam(args.lr_outer),
    first_order=args.first_order,
)

# Initialize
rng = jax.random.PRNGKey(args.seed)
rng_reset, rng_train, rng_test = jax.random.split(rng, 3)
meta_state = meta_learner.reset(rng_reset, metaloader.sample_input)
meta_update = jax.jit(meta_learner.update)
meta_eval = jax.jit(meta_learner.eval, static_argnames="steps")

# Train
for idx, batch in zip(range(args.steps_outer), metaloader):
    # Mangle data into the format expected by metax
    batch = MetaDataset(
        train=Dataset(x=batch["train"].inputs, y=batch["train"].targets),
        test=Dataset(x=batch["test"].inputs, y=batch["test"].targets),
    )
    batch = jtu.tree_map(lambda x: jnp.array(x), batch)

    # Meta-update
    rng_update, rng_train = jax.random.split(rng_train)
    meta_state, metric_train = meta_update(rng_update, meta_state, batch)

    # Print step and loss
    print(f"step: {idx} \t loss: {metric_train['loss_outer']:.4f} \t acc: {metric_train['acc_outer']:.4f}")
