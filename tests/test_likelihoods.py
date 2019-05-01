import pytest

from yabf import Component, Parameter, Param, Likelihood


@pytest.fixture(scope="module")
def simple_component():
    class SimpleComponent(Component):
        provides = ("x2",)
        base_parameters = [
            Parameter("x", 0, min=-10, max=10)
        ]

        def calculate(self, **param):
            return param['x'] ** 2

    return SimpleComponent


@pytest.fixture(scope="module")
def simple_likelihood():
    class SimpleLikelihood(Likelihood):
        base_parameters = [
            Parameter("y", 0, min=-100, max=100)
        ]

        def _reduce(self, ctx, **params):
            return ctx['x2'] * params['y']

        def lnl(self, model, **params):
            return -model

    return SimpleLikelihood


@pytest.fixture(scope="module")
def inactive_lk(simple_likelihood, simple_component):
    return simple_likelihood(components=[simple_component(name='cmp')])


@pytest.fixture(scope="module")
def global_lk(simple_likelihood, simple_component):
    return simple_likelihood(components=[simple_component(name='cmp')],
                             params=(Param('x', fiducial=1.5),))


@pytest.fixture(scope="module")
def sub_lk(simple_likelihood, simple_component):
    return simple_likelihood(
        components=[
            simple_component(
                name='cmp',
                params=(Param('x', fiducial=1.5),)
            )
        ],
        params=(Param('y', fiducial=2),)
    )


def test_likelihood_properties(inactive_lk, global_lk, sub_lk):
    lk = inactive_lk

    assert not lk.in_active_mode
    assert "y" in lk.base_parameter_dct
    assert len(lk.child_base_parameters) == len(lk.child_base_parameter_names) == 2
    assert len(lk.active_params_dct) == len(lk.active_params) == 0
    assert lk.total_active_params == 0
    assert lk.fiducial_params['y'] == 0
    assert lk.logprior() == 0

    assert global_lk.in_active_mode
    assert global_lk.total_active_params == 1
    assert global_lk.logprior() == 0
    assert len(global_lk.fiducial_params) == 1  # fidicual only cares about top-level

    assert sub_lk.in_active_mode
    assert sub_lk.total_active_params == 2
    assert sub_lk.logprior() == 0
    assert len(sub_lk.fiducial_params) == 1  # fidicual only cares about top-level


def test_generate_refs(inactive_lk, global_lk):
    lk = inactive_lk

    refs = lk.generate_refs()
    assert len(refs) == 0
    refs = lk.generate_refs(params=['x', 'y'])
    assert len(refs) == 2
    refs = lk.generate_refs(n=10, params=['x', 'y'])
    assert len(refs[0]) == len(refs[1]) == 10
    refs = lk.generate_refs(params=['x'], squeeze=True)
    assert refs == [0]

    lk = global_lk

    refs = lk.generate_refs()
    assert len(refs) == 1
    refs = lk.generate_refs(squeeze=True)
    assert -10 < refs[0] < 10
    refs = lk.generate_refs(n=10, squeeze=True, params=['y'])
    assert len(refs) == 1
    assert refs[0] == [0] * 10


def test_fill_params(inactive_lk, global_lk, sub_lk):
    params = inactive_lk._fill_params()
    assert inactive_lk._is_params_full(params)
    assert params['y'] == 0
    assert params['cmp']['x'] == 0

    params = global_lk._fill_params()
    assert global_lk._is_params_full(params)
    assert params['y'] == 0

    assert params['cmp']['x'] == 1.5

    params = sub_lk._fill_params()
    assert sub_lk._is_params_full(params)
    assert params['y'] == 2
    assert params['cmp']['x'] == 1.5


def test_parameter_list_to_dict(inactive_lk, global_lk, sub_lk):
    with pytest.raises(AttributeError):
        inactive_lk._parameter_list_to_dict([])

    with pytest.raises(ValueError):
        p = global_lk._parameter_list_to_dict([])

    p = global_lk._parameter_list_to_dict([7])
    assert p['x'] == 7

    p = sub_lk._parameter_list_to_dict([7, 10])
    assert p['y'] == 7
    assert p['cmp']['x'] == 10
