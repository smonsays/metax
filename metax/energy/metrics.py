"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import Dict, Optional

import chex
import jax.numpy as jnp


def accuracy(pred: chex.Array, target: chex.Array, mask: Optional[chex.Array] = None) -> chex.Array:
    correct = jnp.argmax(pred, axis=-1) == target

    if mask is not None:
        return jnp.sum(mask * correct) / jnp.sum(mask)
    else:
        return jnp.mean(correct)


def accuracy_metrics(
    pred: chex.Array,
    target: chex.Array,
    mask: Optional[chex.Array] = None
) -> Dict:
    correct = jnp.argmax(pred, axis=-1) == target
    last_unmasked = jnp.max(jnp.arange(len(mask)) * mask)

    if mask is not None:
        acc = jnp.sum(mask * correct) / jnp.sum(mask)
    else:
        acc = jnp.mean(correct)

    return {
        "acc": acc,
        "acc_all": acc == 1.0,
        "acc_first": correct[0],
        "acc_last": correct[last_unmasked],
    }
