# `metax`

`metax` is a meta-learning research library in jax. 
It bundles various meta-learning algorithms and architectures that can be flexibly combined and is simple to extend.
It includes the following components

- `metax/learner`
  - maml.py: Backpropagation through the optimization as in [MAML](http://proceedings.mlr.press/v70/finn17a.html)
  - eqprop.py: Equilibrium propagation as in [CML](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a6d7226db2ff3643d8624624e3859c19-Abstract-Conference.html)
  - evolution.py: Evolutionary algorithms interfacing with [`evosax`](https://github.com/RobertTLange/evosax) 
  - implicit.py: [Conjugate Gradient](https://papers.nips.cc/paper/2019/hash/072b030ba126b2f4b2374f342be9ed44-Abstract.html), [Recurrent Backpropagation](https://arxiv.org/abs/1803.06396), [T1T2](https://proceedings.mlr.press/v48/luketina16.html)
  - reptile.py: [Reptile](https://arxiv.org/abs/1803.02999)
- `metax/module`
  - anil.py: [ANIL](https://arxiv.org/abs/1909.09157)
  - cavia.py: [CAVIA](https://arxiv.org/abs/1810.03642)
  - compsyn.py: Complex synapse model from [CML](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a6d7226db2ff3643d8624624e3859c19-Abstract-Conference.html)
  - gainmod.py: Gain modulation model from [CML](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a6d7226db2ff3643d8624624e3859c19-Abstract-Conference.html)
  - init.py: Meta-learned initialization as in [MAML](http://proceedings.mlr.press/v70/finn17a.html)


## Installation
Install metax using pip:
```
pip install git+https://github.com/smonsays/metax
```

## Examples

The classic MAML model meta-learns the initialization of the model parameters by backpropagating through the optimizer. In `metax` the code would look as follows:
```python
meta_model = metax.module.LearnedInit(
    loss_fn_inner=metax.energy.SquaredError(),
    loss_fn_outer=metax.energy.SquaredError(),
    base_learner=hk.transform_with_state(
        lambda x, is_training: hk.nets.MLP([64, 64, 1])(x)
    ),
    output_dim=1,
)
meta_learner = metax.learner.ModelAgnosticMetaLearning(
    meta_model=meta_model,
    batch_size=None,  # full batch GD
    steps_inner=10,
    optim_fn_inner=optax.adamw(0.1),
    optim_fn_outer=optax.adamw(0.001),
    first_order=False,
)
```
[`examples/`](https://github.com/smonsays/metax/examples) contains a number of educational examples that demonstrate various combinations of meta-algorithms with meta-architectures on a simple regression task.


## Citation

If you use `metax` in your research, please cite it as:

```
@software{metax2023,
  title={metax: a jax meta-learning library},
  author={Schug, Simon},
  url = {http://github.com/smonsays/metax},
  year={2023}
}
```

