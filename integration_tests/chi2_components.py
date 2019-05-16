import matplotlib.pyplot as plt
import numpy as np
from getdist import plots
from scipy import stats

from yabf import Parameter, Likelihood, Component, mpi
from yabf.samplers import emcee, polychord


class Chi2(Likelihood):
    base_parameters = [
        Parameter("sigma", 1, min=0, latex=r"\sigma")
    ]

    def __init__(self, x, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x

    def _reduce(self, ctx, **dct):
        model = np.array([v for k, v in ctx.items() if k.endswith("model")])
        return np.sum(model, axis=0)

    def _mock(self, model, **params):
        return model + np.random.normal(loc=0, scale=params['sigma'], size=len(model))

    def lnl(self, model, **params):
        sigma = params['sigma']
        nm = stats.norm(loc=model, scale=sigma)

        lnl = np.sum(nm.logpdf(self.data))
        if np.isnan(lnl):
            lnl = -np.inf
        return lnl


class LinearComponent(Component):
    base_parameters = [
        Parameter("p0", 1),
        Parameter("p1", 0)
    ]

    def __init__(self, x, **kwargs):
        self.x = x
        super().__init__(**kwargs)

        self.provides = [f"{self.name}_model"]

    def calculate(self, ctx, **params):
        return params['p0'] * self.x + params['p1']


class Poly(Component):
    def __new__(cls, n=5, *args, **kwargs):
        # First create the parameters.
        p = []
        for i in range(n):
            p.append(Parameter(f"p{i}", 0, latex=r"p_{}".format(i)))
        cls.base_parameters = tuple(p)

        obj = super(Poly, cls).__new__(cls)

        return obj

    def __init__(self, x, n=5, *args, **kwargs):
        # Need to add n to signature to take it out of the call to __init__
        self.poly_order = n
        self.x = x

        super().__init__(*args, **kwargs)
        self.provides = [f"{self.name}_model"]

    def calculate(self, ctx, **params):
        p = [params[f'p{i}'] for i in range(self.poly_order)]
        return sum([pp * self.x ** i for i, pp in enumerate(p)])


def run(sampler, lk, name_sub=""):
    if sampler == "emcee":
        smp = emcee(
            likelihood=lk, output_dir="EmceeChains", output_prefix=f"chi2{name_sub}",
        )
        mcsamples = smp.sample(progress=True, nsteps=2000)

    elif sampler == "polychord":
        smp = polychord(
            likelihood=lk, output_dir="PolyChordChains", output_prefix=f"chi2{name_sub}",
            sampler_kwargs={"nlive": 256, "read_resume": False, "precision_criterion":0.1}
        )
        mcsamples = smp.sample()

    means = mcsamples.getMeans()
    vars = mcsamples.getVars()

    if mpi.am_single_or_primary_process:

        for param, mean, var in zip(lk.child_active_params, means, vars):
            print(f"{param.name}: {mean:1.3e} +/- {np.sqrt(var):1.3e}\t[{param.fiducial}]")

        g = plots.getSubplotPlotter()
        g.triangle_plot(mcsamples, params=list(lk.child_active_params), shaded=True)
        plt.savefig(f"chi2_{sampler}{name_sub}_corner.png")

    return means

