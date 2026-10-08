from pathlib import Path

import jax.numpy as jnp

from caj import cache


def test_culling(tmp_path: Path) -> None:
    @cache(dir=tmp_path, max_bytes=25)
    def f(x):
        return x + 1

    for i in range(100):
        assert f(jnp.array(i)) == i + 1

    assert 5 < len(list(tmp_path.glob("*.caj"))) < 10
