from pathlib import Path

from caj import cache


def test_culling(tmp_path: Path) -> None:
    @cache(dir=tmp_path, max_bytes=25)
    def f(x):
        return x + 1

    assert f(0) == 1
    [first] = tmp_path.glob("*.caj")
    size = first.stat().st_size

    for i in range(1, 25 // size + 1):
        assert len(list(tmp_path.glob("*.caj"))) == i
        assert f(i) == i + 1

    assert len(list(tmp_path.glob("*.caj"))) == 25 // size
    assert not first.exists()
