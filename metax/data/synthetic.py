"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import logging
from typing import List, Optional

import chex
import jax

from metax.data.dataset.base import DatasetGenerator

from .base import Dataloader, MetaDataset
from .dataset import family, sinusoid
from .utils import create_metadataset


class SyntheticMetaDataloader(Dataloader):
    def __init__(
        self,
        data_generator: DatasetGenerator,
        num_tasks: int,
        shots_train: int,
        shots_test: int,
        meta_batch_size: int,
        mode: str,
        train_test_split: bool,
        rng: chex.PRNGKey,
    ):
        super().__init__(
            input_shape=data_generator.input_shape,
            output_dim=data_generator.output_dim
        )
        self.data_generator = data_generator
        self.num_tasks = num_tasks
        self.shots_train = shots_train
        self.shots_test = shots_test
        self.meta_batch_size = meta_batch_size
        self.mode = mode
        self.train_test_split = train_test_split
        self.fixed_rng = rng

        assert train_test_split or mode == "train", "mode must be train if train_test_split is False"
        assert num_tasks % meta_batch_size == 0, "num_tasks must be divisible by meta_batch_size"
        self.num_steps = num_tasks // meta_batch_size
        self.shots = shots_train + shots_test

        # Sample data to get placeholder_input
        self._sample_input = self.data_generator.sample(
            self.fixed_rng, 1, self.shots_train, mode="train"
        ).x[0]

    @property
    def sample_input(self):
        return self._sample_input

    def __len__(self):
        return self.num_steps

    def __iter__(self):
        for rng in jax.random.split(self.fixed_rng, self.num_steps):
            dataset = self.data_generator.sample(rng, self.meta_batch_size, self.shots, mode=self.mode)

            if self.train_test_split:
                # Split into train and test set
                yield create_metadataset(dataset, self.shots_train)
            else:
                # No train_test split means, meta.train == meta.test set
                yield MetaDataset(train=dataset, test=dataset)


def create_synthetic_metadataset(
    name,
    meta_batch_size,
    shots_train,
    shots_test,
    train_test_split,
    num_tasks_train,
    num_tasks_test,
    num_tasks_valid,
    num_tasks_ood: Optional[int] = None,
    ood_sets_hot: Optional[List[int]] = None,
    seed: int = 0,
    **kwargs,
):

    if name == "family":
        data_generator = family.Family()
    elif name == "harmonic":
        data_generator = family.Harmonic()
    elif name == "linear":
        data_generator = family.Linear()
    elif name == "polynomial":
        data_generator = family.Polynomial()
    elif name == "sawtooth":
        data_generator = family.Sawtooth()
    elif name == "sinusoid_family":
        data_generator = family.SinusoidFamily()
    elif name == "sinusoid":
        data_generator = sinusoid.Sinusoid()
    else:
        raise ValueError

    # Wrap data generator in SyntheticMetaDataloader
    rng_train, rng_test, rng_valid, rng_ood, rng_aux = jax.random.split(
        jax.random.PRNGKey(seed), 5
    )
    metatrainloader = SyntheticMetaDataloader(
        data_generator=data_generator,
        num_tasks=num_tasks_train,
        shots_train=shots_train,
        shots_test=shots_test if train_test_split else 0,
        meta_batch_size=meta_batch_size,
        mode="train",
        train_test_split=train_test_split,
        rng=rng_train,
    )

    metatestloader = SyntheticMetaDataloader(
        data_generator=data_generator,
        num_tasks=num_tasks_test,
        shots_train=shots_train,
        shots_test=shots_test,
        meta_batch_size=num_tasks_test,
        mode="test",
        train_test_split=True,
        rng=rng_test,
    )

    metavalidloader = SyntheticMetaDataloader(
        data_generator=data_generator,
        num_tasks=num_tasks_valid,
        shots_train=shots_train,
        shots_test=shots_test,
        meta_batch_size=num_tasks_valid,
        mode="test",
        train_test_split=True,
        rng=rng_valid,
    )

    if num_tasks_ood is not None:
        metaoodloader = SyntheticMetaDataloader(
            data_generator=data_generator,
            num_tasks=num_tasks_ood,
            shots_train=shots_train,
            shots_test=shots_test,
            meta_batch_size=num_tasks_ood,
            mode="ood",
            train_test_split=True,
            rng=rng_ood,
        )
    else:
        metaoodloader = None

    if ood_sets_hot is not None:
        metaauxloaders = {
            "ood_{}".format(ood_set): SyntheticMetaDataloader(
                data_generator=data_generator,
                num_tasks=num_tasks_ood,
                shots_train=shots_train,
                shots_test=shots_test,
                meta_batch_size=num_tasks_ood,
                mode="ood_{}".format(ood_set),
                train_test_split=True,
                rng=r,
            )
            for ood_set, r in zip(ood_sets_hot, jax.random.split(rng_aux, len(ood_sets_hot)))
        }
    else:
        metaauxloaders = None

    return metatrainloader, metatestloader, metavalidloader, metaoodloader, metaauxloaders
