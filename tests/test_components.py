import pytest

from yabf import Component, Param

from .shared_resources import SimpleComponent


def test_component_plugin():
    assert SimpleComponent.__name__ in Component._plugins


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
