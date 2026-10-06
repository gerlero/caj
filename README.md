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

**caj** is a simple persistent cache for JAX-based code.

Decorate a function with `@cache`, and **caj** will store the return on disk. Later calls with the same inputs will load the result directly from the cache instead of performing the computation again.

```python
import jax.numpy as jnp
from caj import cache


@cache
@jax.jit
def compute(x):
    return jnp.linalg.eigvalsh(x)


x = jnp.eye(1000)

y = compute(x)
```

Running the above code twice will only compute the eigenvalues once: the second time, **caj** will just load the result from disk.

**caj** takes advantage of the computation tracing provided by JAX so it can detect changes in the decorated function or any other functions it calls and not return cached results that were computed with different code.

## Installation

Install **caj** from PyPI:

```console
pip install caj
```

## Usage

For most uses, the default cache is enough:

```python
from caj import cache


@cache
def f(x): ...
```

By default, **caj** stores entries in a per-user cache directory and limits the cache to **1 GB**.

A custom cache directory can be specified with `dir`:

```python
@cache(dir=".cache")
def f(x): ...
```

The maximum size of a custom cache can also be configured via `max_bytes`:

```python
@cache(dir=".cache", max_bytes=2_000_000_000)
def f(x): ...
```

Or, set `max_bytes=None` for no size limit:

```python
@cache(dir=".cache", max_bytes=None)
def f(x): ...
```

When a size limit is enabled, **caj** will remove the least recently used entries as necessary to keep the cache within the target size.
