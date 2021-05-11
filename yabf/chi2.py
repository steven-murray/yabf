"""Defines a Chi2 Likelihood."""
import attr
import numpy as np
from scipy import stats

from .core.likelihood import Likelihood
from .core.parameters import Parameter


def is_diagonal(m):
    """Fast check for if an array is diagonal.

    Gotten from https://stackoverflow.com/a/43885215/1467820.
    """
    assert m.ndim == 2
    i, j = m.shape
    assert i == j
    test = m.reshape(-1)[:-1].reshape(i - 1, j + 1)
    return ~np.any(test[:, 1:])


@attr.s(frozen=True)
class Chi2:
    base_parameters = [Parameter("sigma", 0.013, min=0, latex=r"\sigma")]

    sigma = attr.ib(None, kw_only=True)

    def get_sigma(self, model, **params):
        if self.sigma is not None:
            if "sigma" in self.active_params:
                # Act as if sigma is important
                return params["sigma"] * self.sigma
            else:
                return self.sigma
        else:
            return params["sigma"]

    def _mock(self, model, **params):
        sigma = self.get_sigma(model, **params)
        return model + np.random.normal(loc=0, scale=sigma, size=len(model))

    def lnl(self, model, **params):
        sigma = self.get_sigma(model, **params)

        if isinstance(sigma, (float, int)):
            sigma = sigma * np.ones_like(self.data)

        # Ensure we don't use flagged channels
        mask = ~np.isnan(self.data)
        d = self.data[mask]
        m = model[mask]

        s = sigma[mask][:, mask] if sigma.ndim == 2 else sigma[mask]

        if s.ndim <= 2 or is_diagonal(s):
            if s.ndim == 2:
                s = np.diag(s)
            nm = stats.norm(loc=m, scale=s)
        else:
            nm = stats.multivariate_normal(mean=m, cov=s, allow_singular=True)

        lnl = np.sum(nm.logpdf(d))
        if np.isnan(lnl):
            lnl = -np.inf
        return lnl

    # ===== Potential Derived Quantities
    def residual(self, model, ctx, **params):
        return self.data - model

    def rms(self, model, ctx, **params):
        return np.sqrt(np.mean((model - self.data) ** 2))


@attr.s(frozen=True)
class MultiComponentChi2(Chi2, Likelihood):
    """A Chi2 likelihood that can add several models together to form a full model.

    Parameters
    ----------
    kind
        Any data in the context whose key ends with this string will be added together
        to form the final model.
    positive
        Whether to force the likelihood to be zero if any of the model points go
        below zero.
    """

    kind = attr.ib(validator=attr.validators.instance_of(str), kw_only=True)
    positive = attr.ib(True, converter=bool, kw_only=True)

    def _reduce(self, ctx, **params):
        models = np.array([v for k, v in ctx.items() if k.endswith(self.kind)])
        scalars = sum(v for k, v in ctx.items() if k.endswith("scalar"))
        return np.sum(models, axis=0) + scalars

    def lnl(self, model, **params):
        # return -inf if any bit of the spectrum is negative
        if self.positive and np.any(model[~np.isnan(model)] <= 0):
            return -np.inf

        return super().lnl(model, **params)
