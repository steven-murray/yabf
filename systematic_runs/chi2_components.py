import matplotlib.pyplot as plt
import numpy as np
from getdist import plots
from scipy import stats
import attr

from yabf import Parameter, Likelihood, Component, mpi
from yabf import samplers
from cached_property import cached_property

@attr.s(frozen=True)
class Chi2(Likelihood):
    base_parameters = [
        Parameter("sigma", 1, min=0, latex=r"\sigma")
    ]

    x = attr.ib(kw_only=True)

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


@attr.s(frozen=True)
class LinearComponent(Component):
    base_parameters = [
        Parameter("p0", 1),
        Parameter("p1", 0)
    ]

    x = attr.ib(kw_only=True)

    @cached_property
    def provides(self):
        return [f"{self.name}_model"]

    def calculate(self, ctx, **params):
        return params['p0'] * self.x + params['p1']

@attr.s(frozen=True)
class Poly(Component):
    def __new__(cls, poly_order=5, *args, **kwargs):
        # First create the parameters.
        p = []
        for i in range(poly_order):
            p.append(Parameter(f"p{i}", 0, latex=r"p_{}".format(i)))
        cls.base_parameters = tuple(p)

        obj = super(Poly, cls).__new__(cls)

        return obj

    x = attr.ib(kw_only=True)
    poly_order = attr.ib(5, kw_only=True, converter=int)

    @cached_property
    def provides(self):
        return [f"{self.name}_model"]

    def calculate(self, ctx, **params):
        p = [params[f'p{i}'] for i in range(self.poly_order)]
        return sum([pp * self.x ** i for i, pp in enumerate(p)])


def run(sampler, lk, name_sub=""):
    if sampler == "emcee":
        smp = samplers.emcee(
            likelihood=lk, output_dir="EmceeChains", output_prefix=f"chi2{name_sub}",
        )
        mcsamples = smp.sample(progress=True, nsteps=2000)

    elif sampler == "polychord":
        smp = samplers.polychord(
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

