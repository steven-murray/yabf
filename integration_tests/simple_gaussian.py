"""
Define a simple Gaussian likelihood to make sure things fit well.
"""

from yabf import Likelihood, Parameter
import numpy as np
from scipy import stats
from getdist import plots
import matplotlib.pyplot as plt


class SimpleGaussian(Likelihood):
    base_parameters = [
        Parameter("mu", 0, latex=r"\mu"),
        Parameter("sigma", 1, min=0, latex=r"\sigma")
    ]

    def __init__(self, ndata=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ndata = ndata

    def _mock(self, model, **params):
        print(params)
        return np.random.normal(loc=params['mu'], scale=params['sigma'], size=self.ndata)

    def lnl(self, model, **params):
        nm = stats.norm(loc=params['mu'], scale=params['sigma'])

        lnl = np.sum(nm.logpdf(self.data))
        if np.isnan(lnl):
            lnl = -np.inf
        return lnl


if __name__ == "__main__":
    import sys
    from yabf.samplers import emcee, polychord
    from yabf import Param
    sampler = sys.argv[1]

    lk = SimpleGaussian(
        name="gaussian",
        ndata=10000,
        params=(Param('mu', min=-10, max=10, ref=stats.norm(0, 1)), Param("sigma", max=10, ref=stats.norm(1,0.1)))
    )

    if sampler == "emcee":
        smp = emcee(
            likelihood=lk, output_dir="EmceeChains", output_prefix="gaussian",
        )
        mcsamples = smp.sample(progress=True, nsteps=2000)

    elif sampler == "polychord":
        smp = polychord(
            likelihood=lk, output_dir="PolyChordChains", output_prefix="gaussian",
            sampler_kwargs={"nlive":256}
        )
        mcsamples = smp.sample()

    print(mcsamples.getMeans())
    print(mcsamples.getVars())

    g = plots.getSubplotPlotter()
    g.triangle_plot(mcsamples, params=list(lk.child_active_params), shaded=True)
    plt.savefig(f"simple_gaussian_{sampler}_corner.png")
