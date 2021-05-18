import pytest

from yabf.core.parameters import Param, Params


@pytest.fixture(scope="function")
def params():
    return Params((Param(name="a"), Param("b"), Param("c")))


def test_params_functionality(params):
    assert "a" in params
    assert params.a.name == "a"
    assert len(params) == 3
    assert params[0] == params.a

    for i, param in enumerate(params):
        assert params[i] == param
