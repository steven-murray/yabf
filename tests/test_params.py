import pytest

from yabf.core.parameters import Param, Params


@pytest.fixture
def params():
    return Params((Param(name="a"), Param("b"), Param("c")))


def test_params_functionality(params):
    assert "a" in params
    assert params.a.name == "a"
    assert len(params) == 3
    assert params[0] == params.a

    for i, param in enumerate(params):
        # Deliberately re-index rather than comparing `param` to itself: this
        # checks that __getitem__ ordering agrees with __iter__ ordering.
        assert params[i] == param  # noqa: PLR1736
