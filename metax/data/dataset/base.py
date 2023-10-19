import abc
from typing import Tuple

import chex

from metax.data.base import Dataset


class DatasetGenerator(abc.ABC):
    """
    Abstract base class for generated datasets.

    Attributes:
        input_shape (tuple): The shape of the input data.
        output_dim (int): The dimensionality of the output data.
    """
    def __init__(self, input_shape: Tuple[int], output_dim: int) -> None:
        self.input_shape = input_shape
        self.output_dim = output_dim

    @abc.abstractmethod
    def sample(self, rng: chex.PRNGKey, num_tasks: int, num_samples: int, mode: str) -> Dataset:
        """
        Generate a batch of tasks.

        Args:
            rng (jax.random.PRNGKey): The random number generator to use.
            num_tasks (int): The number of tasks to generate.
            num_samples (int): The number of samples per task.
            mode (str): The mode of the generated data (e.g. 'train', 'test', 'ood').

        Returns:
            A namedtuple `Dataset` (x, y) containing the input and output data for the generated tasks.
            x has shape (num_tasks, num_samples) + input_shape.
            y has shape (num_tasks, num_samples, output_dim).
        """
        pass
