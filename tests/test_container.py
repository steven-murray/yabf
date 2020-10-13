import pytest

from yabf import Component, Likelihood, LikelihoodContainer, Param, Parameter

from .shared_resources import SimpleComponent, SimpleLikelihood


def test_two_lk_shared_cmp():
    shared = SimpleComponent(name="shared")

    lk = LikelihoodContainer(
        likelihoods=(
            SimpleLikelihood(name="only_shared", components=(shared,)),
            SimpleLikelihood(
                name="shared_and", components=(SimpleComponent(name="unshared"), shared)
            ),
        )
    )

    params = lk._fill_params()

    assert "only_shared" in params
    assert "shared_and" in params
    assert params["only_shared"]["y"] == 0
    assert params["shared_and"]["y"] == 0
    assert params["shared_and"]["unshared"]["x"] == 0

    ctx = lk.get_ctx(params=params)

    assert ctx["only_shared"]["x2"] == 0
    assert len(ctx) == 2  # each likelihood gets an entry
    assert lk.logl(params=params) == 0
    assert lk.logl(params={"only_shared.shared.x": 2, "only_shared.y": 1}) == -4

    params = lk._fill_params({"only_shared": {"shared": {"x": 2}, "y": 1}})
    print("PARAMS: ", params)
    assert lk.logl(params={"only_shared": {"shared": {"x": 2}, "y": 1}}) == -4

    with pytest.raises(TypeError):
        lk.logl(x=2, y=2)  # x can't be found.


def test_two_lk_sharing_a_param():
    class ThisComponent(Component):
        provides = ("x2",)
        base_parameters = [
            Parameter("x", 0, min=-10, max=10),
            Parameter("y", 0, min=-10, max=10),
            Parameter("z", 0, min=-10, max=10),
        ]

        def calculate(self, ctx, x, y, z):
            return x ** 2 + y ** 2 + z ** 2

    class ThisLikelihood(Likelihood):
        base_parameters = [Parameter("w", 0, min=-100, max=100)]

        def _reduce(self, ctx, **params):
            return ctx["x2"] * params["w"]

        def lnl(self, model, **params):
            return -model

    small_cmp = ThisComponent(
        name="small_component",
        params=(
            Param("x", fiducial=0),
            Param("y", fiducial=0),
            Param("z", fiducial=2),
        ),
    )

    lk = LikelihoodContainer(
        likelihoods=(
            ThisLikelihood(
                name="small",
                params=(Param("w", fiducial=1),),
                components=[small_cmp],
            ),
            ThisLikelihood(
                name="big",
                params=(Param("w", fiducial=1),),
                components=[
                    ThisComponent(
                        name="large_component",
                        params=(
                            Param("xx", fiducial=5, determines=["x"]),
                            Param("yy", fiducial=5, determines=["y"]),
                            Param("z", fiducial=2),
                        ),
                    )
                ],
            ),
        )
    )

    print(lk.child_active_params)
    assert lk.total_active_params == 6
    assert lk.logl() == lk.logp() == -58
