"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import Callable, Iterable, List, Optional

import haiku as hk
import jax
import jax.numpy as jnp

from .linear import LinearBlock


class MultilayerPerceptron(hk.Module):
    def __init__(
        self,
        output_sizes: Iterable[int],
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        with_bias: bool = True,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
        activate_final: bool = False,
        batch_norm: bool = False,
        reparametrized_linear: bool = False,
        names_layers: Optional[List[str]] = None,
        name: Optional[str] = None,
    ):
        """Constructs an MLP.

        Args:
          output_sizes: Sequence of layer sizes.
          w_init: Initializer for :class:`~haiku.Linear` weights.
          b_init: Initializer for :class:`~haiku.Linear` bias. Must be ``None`` if
            ``with_bias=False``.
          with_bias: Whether or not to apply a bias in each layer.
          activation: Activation function to apply between :class:`~haiku.Linear`
            layers. Defaults to ReLU.
          activate_final: Whether or not to activate the final layer of the MLP.
          batch_norm: Whether or not to add batch_norm after each linear layer.
          name: Optional name for this module.

        Raises:
          ValueError: If ``with_bias`` is ``False`` and ``b_init`` is not ``None``.
        """
        if not with_bias and b_init is not None:
            raise ValueError("When with_bias=False b_init must not be set.")

        super().__init__(name=name)
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init
        self.activation = activation
        self.activate_final = activate_final
        layers = []
        output_sizes = tuple(output_sizes)
        for i, output_size in enumerate(output_sizes):
            if names_layers is not None:
                name = names_layers[i]
            else:
                name = "linear_{}".format(i)

            layers.append(
                LinearBlock(
                    output_size=output_size,
                    w_init=w_init,
                    b_init=b_init,
                    with_bias=with_bias,
                    batch_norm=batch_norm,
                    reparametrized_linear=reparametrized_linear,
                    name=name,
                )
            )
        self.layers = tuple(layers)
        self.output_size = output_sizes[-1] if output_sizes else None

    def __call__(
        self,
        inputs: jnp.ndarray,
        is_training: bool,
        gain: Optional[jnp.ndarray] = None,
        shift: Optional[jnp.ndarray] = None,
        skip_readout: Optional[bool] = False,
        dropout_rate: Optional[float] = None,
    ) -> jnp.ndarray:
        """
        Multilayer perceptron with optional gain and shift modulation and skipping of readout layer.
        Args:
            inputs: A Tensor of shape ``[batch_size, input_size]``.
            gain: An optional list of Tensors of length ``num_layers``
                    and of shapes ``hidden_dims + [output_dim]``
            shift: An optional list of Tensors of length ``num_layers``
                    and of shapes ``hidden_dims + [output_dim]``
            skip_readout: An optional bool indicating whether to skip last readout layer
            dropout_rate: Optional dropout rate.
            rng: Optional RNG key. Require when using dropout.

        Returns:
            The output of the model of size ``[batch_size, output_size]``.
        """
        num_layers = len(self.layers)

        out = hk.Flatten(preserve_dims=1)(inputs)

        for i, layer in enumerate(self.layers):
            if i < (num_layers - 1) or not skip_readout:
                out = layer(out, is_training)
                if gain is not None:
                    out = out * gain[i]
                if shift is not None:
                    out = out + shift[i]
                if i < (num_layers - 1) or self.activate_final:
                    # Only perform dropout if we are activating the output.
                    if dropout_rate is not None and is_training:
                        out = hk.dropout(hk.next_rng_key(), dropout_rate, out)
                    out = self.activation(out)

        return out
