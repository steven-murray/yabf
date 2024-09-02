"""
Analytic tests of Bayesian Evidence.
"""

import attr
import numpy as np
import pytest
from pypolychord.output import PolyChordOutput
from pytest_lazy_fixtures import lf
from scipy import stats
from scipy.integrate import quad

from yabf import Likelihood, Param, Parameter
from yabf.samplers import polychord


@attr.s(frozen=True)
class Gaussian(Likelihood):
    base_parameters = (Parameter("mu", fiducial=0),)
    sigma: float = attr.ib(1.0)

    def lnl(self, model, **params):
        return np.sum(stats.norm(loc=params["mu"], scale=self.sigma).logpdf(self.data))


@pytest.fixture(scope="module")
def simple_gaussian():
    np.random.seed(5151)
    return Gaussian(
        data=np.random.normal(0, 1, size=100),
        params=(Param("mu", prior=stats.norm(0, 1)),),
    )


@pytest.fixture(scope="module")
def gaussian_proper_unif_prior():
    np.random.seed(1616)
    return Gaussian(
        data=np.random.normal(0, 1, size=100), params=(Param("mu", min=-1, max=1),)
    )


def get_numerical_evidence(lk):
    def fnc(m):
        return np.exp(lk.logp(params=[m]))

    assert lk.total_active_params == 1

    a, b = lk.child_active_params[0].prior.support()
    return np.log(quad(fnc, a, b)[0])


@pytest.mark.parametrize(
    "lk", [lf("simple_gaussian"), lf("gaussian_proper_unif_prior")]
)
def test_simple_gaussian_evidence(lk, tmp_path):
    out = tmp_path / "polychord"
    out.mkdir()

    poly = polychord(
        likelihood=lk,
        save_full_config=False,
        output_dir=str(out),
        output_prefix="test_gaussian",
        sampler_kwargs={"nlive": 100, "feedback": 2, "precision_criterion": 0.1},
    )

    poly.sample()
    out = PolyChordOutput(str(out), "test_gaussian")

    num_evidence = get_numerical_evidence(lk)
    assert np.isclose(num_evidence, out.logZ, atol=1.0, rtol=0)
