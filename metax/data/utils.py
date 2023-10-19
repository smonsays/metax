from typing import Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from metax.utils import tree_length

from .base import Dataset, MetaDataset, MultitaskDataset


def batch_generator(rng, datastruct, steps, batch_size):
    """
    Add leading dims to datastruct resulting in (steps, batch_size, *data.shape).
    If batch_size is None, repeat each data leaf, otherwise sample random batches.
    """
    if batch_size is None or batch_size < 1:
        # Repeat whole data on new leading dim for number of steps
        def repeat(x):
            return jnp.repeat(jnp.expand_dims(x, axis=0), steps, axis=0)

        return jtu.tree_map(repeat, datastruct)

    else:
        rng_batch = jax.random.split(rng, steps)
        batch_get_batch = jax.vmap(get_batch, in_axes=(0, None, None))

        return batch_get_batch(rng_batch, datastruct, batch_size)


def batch_idx_generator(rng, batch_size, steps, num_data):
    """
    Creates indeces of shape (steps, batch_size) to iterate over data in random batches.
    """
    if steps * batch_size <= num_data:
        # If there are sufficiently many data points, sample without replacement
        idx = jax.random.choice(rng, num_data, (steps, batch_size), replace=False)
    else:
        # Otherwise some samples are repeated
        idx = jax.random.choice(rng, num_data, (steps, batch_size), replace=True)

    return idx


def create_metadataset(dataset: Union[Dataset, MultitaskDataset], shots):
    """
    Split data into train and test set and create batches of tasks on leading axis.

    Args:
        shots: Number of samples used for train (support) set
    """
    # Split all leafs into train and test shots
    dataset_train = jtu.tree_map(
        lambda x: jnp.split(x, indices_or_sections=(shots, ), axis=1)[0], dataset
    )
    dataset_test = jtu.tree_map(
        lambda x: jnp.split(x, indices_or_sections=(shots, ), axis=1)[1], dataset
    )

    return MetaDataset(train=dataset_train, test=dataset_test)


def merge_metadataset(metaset: MetaDataset):
    """
    Undo `create_metadataset`, i.e. re-join train and test sets.

    Args:
        metaset: MetaDataset containing MultitaskDatasets
    """
    return jtu.tree_map(lambda x, y: jnp.concatenate((x, y), axis=1), metaset.train, metaset.test)


def get_batch(rng, datastruct, batch_size):
    """
    Get single random batch from data.
    """
    # Draw random indeces with replacement
    num_entries = tree_length(datastruct)
    idx = jax.random.choice(rng, num_entries, (batch_size, ), replace=True)

    return jtu.tree_map(lambda x: x[idx], datastruct)
