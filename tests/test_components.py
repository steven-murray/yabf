import pytest

from yabf import Component, Parameter, Param


@pytest.fixture(scope="module")
def simple_component():
    class SimpleComponent(Component):
        provides = ('x2',)

        base_parameters = [
            Parameter("x", 0)
        ]

        def calculate(self, **param):
            return param['x'] ** 2

    return SimpleComponent


def test_importing_bad_component():
    with pytest.raises(TypeError):
        # no "provides" set!

        class BadComponent(Component):
            def calculate(self, **params):
                return 1


def test_component_plugin(simple_component):
    assert simple_component.__name__ in Component._plugins


def test_simple_component_properties(simple_component):
    a = simple_component()
    assert not a.in_active_mode
    assert "x" in a.base_parameter_dct
    assert len(a.child_base_parameters) == 1
    assert len(a.active_params) == 0
    assert a.fiducial_params == {"x": 0}

    a = simple_component(params=[Param('x', fiducial=1)])
    assert a.in_active_mode
    assert len(a.active_params) == 1
    assert a.fiducial_params == {"x": 1}

    b = a._fill_params({"x": 2})
    assert b['x'] == 2
    assert len(b) == 1

    b = a._fill_params({})
    assert len(b) == 1
    assert b['x'] == 1


def test_bad_derived(simple_component):
    with pytest.raises(AssertionError):
        a = simple_component(derived=['non-existent-quantity'])


def test_genref(simple_component):
    a = simple_component()

    refs = a.generate_refs(params=['x'])
    assert len(refs) == 1
    assert len(refs[0]) == 1

    b = simple_component(params=[Param('x', min=-10, max=10)])
    refs = b.generate_refs()
    assert len(refs) == len(refs[0]) == 1
    assert -10 < refs[0][0] < 10
