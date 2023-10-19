from functools import partial

import haiku as hk
from haiku._src.transform import Transformed

from .conv import Conv4
from .linear import LinearBlock
from .mlp import MultilayerPerceptron


def flaxify(m: hk.Module, transform_fn=hk.transform_with_state) -> Transformed:
    """
    Wraps a haiku.Module in a transformed forward function making it behave
    similar to a flax.nn.Module.

    Example:
        >>> # Instead of having to define a boiler plate lambda function
        >>> my_linear = hk.transform(lambda x: hk.Linear(10)(x))
        >>> # Flaxified modules are automatically converted to pure functions after module instantiation
        >>> FlaxifiedLinear = flaxify(hk.Linear)
        >>> my_linear = FlaxifiedLinear(10)
    """
    def fun(*haiku_module_args, **haiku_module_kwargs):
        def forward(*forward_args, **forward_kwargs):
            return m(*haiku_module_args, **haiku_module_kwargs)(*forward_args, **forward_kwargs)

        return transform_fn(forward)

    return fun


# Flaxify haiku modules giving them a similar interface
Conv4 = flaxify(Conv4)
Linear = flaxify(partial(LinearBlock, batch_norm=False))
LinearBlock = flaxify(LinearBlock)
MultilayerPerceptron = flaxify(MultilayerPerceptron)


__all__ = [
    "Conv4",
    "MultilayerPerceptron",
]
