import numpy as np
import pytest
from yabf import Component, Param

from .shared_resources import SimpleComponent, SimpleLikelihood, SuperComponent


def test_component_plugin():
    assert SimpleComponent.__name__ in Component._plugins


def test_super_component():
    s = SuperComponent(
        components=(SimpleComponent(params=("x",)),),
    )
    a = SimpleLikelihood(components=(s,))

    ctx = a.get_ctx(params={"x": 3})
    assert ctx["x2"] == 9
    assert ctx["x4"] == 81


def test_simple_component_properties():
    a = SimpleComponent()
    assert not a.in_active_mode
    assert "x" in a.base_parameter_dct
    assert len(a.child_base_parameters) == 1
    assert len(a.active_params) == 0
    assert a.fiducial_params == {"x": 0}

    a = SimpleComponent(params=[Param("x", fiducial=1)])
    assert a.in_active_mode
    assert len(a.active_params) == 1
    assert a.fiducial_params == {"x": 1}

    b = a._fill_params({"x": 2})
    assert b["x"] == 2
    assert len(b) == 1

    b = a._fill_params({})
    assert len(b) == 1
    assert b["x"] == 1


def test_bad_derived():
    with pytest.raises(AssertionError):
        SimpleComponent(derived=["non-existent-quantity"])


def test_genref():
    a = SimpleComponent()

    with pytest.raises(AttributeError):
        a.generate_refs(params=["x"])

    b = SimpleComponent(params=[Param("x", min=-10, max=10)])
    refs = b.generate_refs()
    assert len(refs) == len(refs[0]) == 1
    assert -10 < refs[0][0] < 10


def test_transform_param():
    a = SimpleComponent(params=[Param("logx", determines=("x",), transforms=(np.exp,))])

    # this can't work because the min/max for logx are by default -inf/inf
    with pytest.raises(ValueError, match="The defined support for 'logx'"):
        a.active_params

    a = SimpleComponent(
        params=[Param("logx", min=-1, max=1, determines=("x",), transforms=(np.exp,))]
    )
    assert a.active_params["logx"].min == -1
    assert a.active_params["logx"].max == 1
    assert list(a.active_params["logx"].transform(0))[0] == 1
