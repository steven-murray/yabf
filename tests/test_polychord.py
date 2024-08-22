"""
Analytic tests of Bayesian Evidence.
"""
import pytest

import attr
import numpy as np
from scipy import stats

from yabf import Likelihood, Param, Parameter
from yabf.samplers.polychord import polychord


@attr.s(frozen=True)
class Gaussian(Likelihood):
    base_parameters = (Parameter("mu", fiducial=0),)
    sigma: float = attr.ib(1.0)

    def lnl(self, model, **params):
        return np.sum(stats.norm(loc=params["mu"], scale=self.sigma).logpdf(self.data))


def derived_function(model, ctx, **params):
    return params["mu"] * 4


def derived_container(model, ctx, **params):
    return [params["mu"], params["mu"] * 4]


@pytest.fixture(scope="module")
def simple_gaussian():
    np.random.seed(5151)
    return Gaussian(
        data=np.random.normal(0, 1, size=100),
        params=(Param("mu", prior=stats.norm(0, 1)),),
        derived=(derived_function, derived_container),
    )


def test_derived(simple_gaussian, tmp_path):
    out = tmp_path / "polychord_derived"
    out.mkdir()

    poly = polychord(
        likelihood=simple_gaussian,
        save_full_config=False,
        output_dir=str(out),
        output_prefix="test_gaussian",
        sampler_kwargs={"nlive": 100, "feedback": 2, "precision_criterion": 0.1},
    )

    assert poly.derived_shapes[0] == 0
    assert poly.derived_shapes[1] == (2,)
    pnames = poly.get_derived_paramnames()
    assert "derived_function" in pnames[0]
    assert "derived_container_0" in pnames[1]
    assert "derived_container_1" in pnames[2]

    poly.sample()
