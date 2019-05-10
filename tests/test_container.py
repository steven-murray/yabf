import pytest

from yabf import Component, Parameter, Likelihood, LikelihoodContainer, Param
from .shared_resources import SimpleComponent, SimpleLikelihood


def test_two_lk_single_external_cmp():
    lk = LikelihoodContainer(
        likelihoods=(
            SimpleLikelihood(
                name="use_external"
            ),
            SimpleLikelihood(
                name="use_internal",
                components=(
                    SimpleComponent(name="internal"),
                )
            )
        ),
        components=(
            SimpleComponent(name='external'),
        )
    )

    params = lk._fill_params()

    assert 'use_external' in params
    assert 'use_internal' in params
    assert params['use_external']['y'] == 0
    assert params['use_internal']['y'] == 0
    assert params['use_internal']['internal']['x'] == 0

    ctx = lk.get_ctx(**params)
    assert ctx['x2'] == 0
    assert len(ctx) == 1  # Though there are two components, they overwrite each other.
    assert lk.logl(**params) == 0
    print(lk.fiducial_params)
    assert lk.logl(**{"external.x": 2, "use_external.y": 1}) == -4
    assert lk.logl(**{"external": {"x": 2}, "use_external": {"y": 1}}) == -4

    with pytest.raises(ValueError):
        lk.logl(x=2, y=2)  # x can't be found.


def test_two_lk_sharing_a_param():
    class ThisComponent(Component):
        provides = ("x2",)
        base_parameters = [
            Parameter("x", 0, min=-10, max=10),
            Parameter("y", 0, min=-10, max=10),
            Parameter("z", 0, min=-10, max=10),
        ]

        def calculate(self, ctx, **param):
            return param['x'] ** 2 + param['y'] ** 2 + param['z'] ** 2

    class ThisLikelihood(Likelihood):
        base_parameters = [
            Parameter("w", 0, min=-100, max=100)
        ]

        def _reduce(self, ctx, **params):
            return ctx['x2'] * params['w']

        def lnl(self, model, **params):
            return -model

    lk = LikelihoodContainer(
        params=[Param('z', fiducial=2), Param("w", fiducial=1)],
        likelihoods=(
            ThisLikelihood(
                name='small',
                components=[
                    ThisComponent(
                        "small_component",
                        params=(
                            Param('x', fiducial=0),
                            Param('y', fiducial=0)
                        )
                    )
                ]
            ),
            ThisLikelihood(
                name='big',
                components=[
                    ThisComponent("large_component",
                                  params=(
                                      Param('xx', fiducial=5, alias_for='x'),
                                      Param('yy', fiducial=5, alias_for='y')
                                  )
                                  )
                ]
            )
        )
    )

    print(lk.child_active_params)
    assert lk.total_active_params == 6
    assert lk.logl() == lk.logp() == -58
