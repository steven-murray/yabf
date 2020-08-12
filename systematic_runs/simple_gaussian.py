"""
Define a simple Gaussian likelihood to make sure things fit well.
"""

import numpy as np
from chi2_components import run
from scipy import stats
import attr
from yabf import Likelihood, Parameter


@attr.s(frozen=True)
class SimpleGaussian(Likelihood):
    base_parameters = [
        Parameter("mu", 0, latex=r"\mu"),
        Parameter("sigma", 1, min=0, latex=r"\sigma"),
    ]

    ndata = attr.ib(
        100, converter=int, validator=attr.validators.instance_of(int), kw_only=True
    )

    def _mock(self, model, **params):
        print(params)
        return np.random.normal(
            loc=params["mu"], scale=params["sigma"], size=self.ndata
        )

    def lnl(self, model, **params):
        nm = stats.norm(loc=params["mu"], scale=params["sigma"])

        lnl = np.sum(nm.logpdf(self.data))
        if np.isnan(lnl):
            lnl = -np.inf
        return lnl


if __name__ == "__main__":
    import sys
    from yabf import Param

    sampler = sys.argv[1]

    lk = SimpleGaussian(
        name="gaussian",
        ndata=10000,
        data_seed=1234,
        params=(
            Param("mu", min=-10, max=10, ref=stats.norm(0, 1)),
            Param("sigma", max=10, ref=stats.norm(1, 0.1)),
        ),
    )

    run(sampler, lk, name_sub="_simple_gaussian")
