<div align="center">
  <a href="https://github.com/gerlero/caj"><img src="https://raw.githubusercontent.com/gerlero/caj/main/logo.png" alt="caj" width="200"/></a>

  **Automatic persistent caching for JAX function invocations**

  [![CI](https://github.com/gerlero/caj/actions/workflows/ci.yml/badge.svg)](https://github.com/gerlero/caj/actions/workflows/ci.yml)
  [![Codecov](https://codecov.io/gh/gerlero/caj/branch/main/graph/badge.svg)](https://codecov.io/gh/gerlero/caj)
  [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
  [![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
  [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
  [![Publish](https://github.com/gerlero/caj/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/gerlero/caj/actions/workflows/pypi-publish.yml)
  [![PyPI](https://img.shields.io/pypi/v/caj)](https://pypi.org/project/caj/)
  [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/caj)](https://pypi.org/project/caj/)
</div>

**caj** is a lightweight persistent cache for [JAX](https://github.com/jax-ml/jax)-based functions. It stores computation results on disk and reuses them even across Python sessions, avoiding repeated expensive computations.

Unlike caches based only on function arguments, **caj** also uses JAX's traced computation to identify cached results. Changes to a function's computation—including changes in any other functions it calls—automatically invalidate the corresponding cached results.

```python
import jax
import jax.numpy as jnp
from caj import cache


@cache
@jax.jit
def compute(x):
    return jnp.linalg.eigvalsh(x)


x = jnp.eye(1000)

y = compute(x)  # First call computes and caches the result
y = compute(x)  # Following calls just load the cached result
```

## Installation

Install **caj** from [PyPI](https://pypi.org/project/caj/):

```console
pip install caj
```

## Usage

Simply decorate a JAX-based function with `@caj.cache`:

```python
import caj


@caj.cache
@jax.jit
def f(x): ...
```

**Note:** adding [`@jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit) is optional. If `@jax.jit` is used, it should be placed below `@caj.cache` (as the latter is not itself JIT-compatible). It is recommended to use `@jax.jit` for performance, since functions that are not JIT-compatible will generally not work with `@caj.cache` either.

By default, **caj** stores results in a per-user cache directory, with a shared size limit of **1 GB**.

To use a custom directory:

```python
@caj.cache(dir=".cache")
def f(x): ...
```

The maximum cache size can also be configured:

```python
@caj.cache(dir=".cache", max_bytes=2_000_000_000)
def f(x): ...
```

Set `max_bytes=None` for an unlimited cache:

```python
@caj.cache(dir=".cache", max_bytes=None)
def f(x): ...
```

When a size limit is enabled, least-recently-used entries are removed as necessary to make room for new results.

## How it works

**caj** identifies cached results using a hash of the traced JAX computation, its input values, and the JAX and jaxlib versions. This allows it to reuse results even across Python sessions while automatically invalidating cached results when the computation changes.

Unlike JAX's persistent compilation cache, which stores compiled executables, **caj** stores the results of computations themselves.
