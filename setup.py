from setuptools import find_namespace_packages, setup

setup(
    name="metax",
    version="0.0.1",
    description="Meta-learning research library in jax.",
    author="smonsays",
    url="https://github.com/smonsays/metax",
    license='MIT',
    install_requires=["chex", "dm_haiku", "flax", "jax", "numpy", "optax", "tqdm"],
    extras_require={
        "dev": ["black", "flake8", "matplotlib"],
    },
    packages=find_namespace_packages(),
)
