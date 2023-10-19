"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import shutil
from typing import Tuple, Union

import flax
import jax.numpy as jnp
import jax.tree_util as jtu


def dict_combine(*ds):
    d_comb = {}
    for d in ds:
        d_comb.update(flax.traverse_util.flatten_dict(d))

    return flax.traverse_util.unflatten_dict(d_comb)


def dict_filter(d, key, all_but_key=False):
    """
    Filter a dictionary by key returning only entries that contain the key.
    Returns the complement if all_but_key=True.
    """

    def match_key_tuples(key1: Union[str, Tuple[str]], key2: Tuple[str]):
        """
        Check if key1 is contained in key2.
        """
        if isinstance(key1, str):
            return any(key1 in k for k in key2)
        else:
            return all(k1 in k2 for k1, k2 in zip(key1, key2))

    # Flatten, filter, unflatten
    d_flat = flax.traverse_util.flatten_dict(d)
    if not all_but_key:
        d_flat_filtered = {k: v for k, v in d_flat.items() if match_key_tuples(key, k)}
    else:
        d_flat_filtered = {k: v for k, v in d_flat.items() if not match_key_tuples(key, k)}

    d_filtered = flax.traverse_util.unflatten_dict(d_flat_filtered)

    return d_filtered


def flatcat(pytree):
    # return jax.flatten_util.ravel_pytree(pytree)[0]
    return jnp.concatenate([p.flatten() for p in jtu.tree_leaves(pytree)])


def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten nested dictionary/namedtuple.
    https://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
    """
    items = []

    if isinstance(d, tuple):
        d = d._asdict()

    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, tuple):
            items.extend(flatten_dict(v._asdict(), new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def append_keys(dictionary, suffix):
    return {key + "_" + suffix: value for key, value in dictionary.items()}


def prepend_keys(dictionary, prefix):
    return {prefix + "_" + key: value for key, value in dictionary.items()}


def zip_and_remove(path):
    """
    Zip and remove a folder to save disk space.
    """
    shutil.make_archive(path, 'zip', path)
    shutil.rmtree(path)
