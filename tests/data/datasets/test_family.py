"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import unittest

import jax
import matplotlib.pyplot as plt

from metax.data.dataset.family import Family


class FamilyTestCase(unittest.TestCase):
    def test_visual(self):
        # Plot all families
        data = Family().sample(jax.random.PRNGKey(0), num_tasks := 10, num_samples=100)
        for i in range(num_tasks):
            plt.scatter(data.x[i], data.y[i], label=data.task_id[i])

        plt.legend()
        plt.show()

        # Plot each family individually
        for fun_type in ["harm", "lin", "poly", "saw", "sine"]:
            data = Family(fun_types=[fun_type]).sample(jax.random.PRNGKey(0), num_tasks := 10, num_samples=500)
            for i in range(num_tasks):
                plt.scatter(data.x[i], data.y[i], label=fun_type)
            plt.legend()
            plt.show()
