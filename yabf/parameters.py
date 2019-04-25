import attr
import numpy as np


@attr.s
class Parameter:
    """
    A fiducial parameter of a model. Used to *define* the model
    and its defaults. A Parameter of a model does not mean it *will* be
    constrained, but rather that it *can* be constrained. To be constrained,
    a :class:`Param` must be set on the instance at run-time.

    min/max in this class specify the total physically/logically allowable domain
    for the parameter. This can be reduced via the specification of :class:`Param`
    at run-time.
    """
    name = attr.ib()
    default = attr.ib()
    min = attr.ib(-np.inf)
    max = attr.ib(np.inf)
    latex = attr.ib()

    @latex.default
    def _ltx_default(self):
        return self.name


@attr.s
class Param:
    """
    Specification of a parameter that is to be constrained.
    """
    name = attr.ib()
    ref = attr.ib(None)
    min = attr.ib(-np.inf, type=float)
    max = attr.ib(np.inf, type=float)
    prior = attr.ib(None)
    fiducial = attr.ib(None)


@attr.s
class _ActiveParameter:
    """A parameter that is actively being constrained. """
    parameter = attr.ib()
    param = attr.ib()

    @parameter.validator
    def _parameter_validator(self, att, val):
        assert type(val) is Parameter

    @param.validator
    def _param_validator(self, att, val):
        assert type(val) is Param
        assert val.name == self.parameter.name

    @property
    def name(self):
        return self.param.name

    @property
    def ref(self):
        return self.param.ref

    @property
    def min(self):
        return max(self.param.min, self.parameter.min)

    @property
    def min(self):
        return min(self.param.max, self.parameter.max)

    @property
    def prior(self):
        return self.param.prior

    @property
    def fiducial_value(self):
        return self.param.fiducial if self.param.fiducial is not None else self.parameter.default

    def generate_ref(self):
        if self.ref is None:
            # Use prior
            if self.prior is None:
                ref = np.random.uniform(self.min, self.max)
            else:
                try:
                    ref = self.prior.rvs()
                except AttributeError:
                    raise NotImplementedError("callable priors not yet implemented")
        else:
            try:
                ref = self.ref.rvs()
            except AttributeError:
                try:
                    ref = self.ref()
                except TypeError:
                    ref = self.ref

        if not self.min <= ref <= self.max:
            raise ValueError(f"param {self.name} produced a reference value outside its domain.")

        return ref

    def logprior(self, val):
        if not (self.min <= val <= self.max):
            return -np.inf

        if self.prior is None:
            return 0
        else:
            try:
                return self.prior.logpdf(val)
            except AttributeError:
                return self.prior(val)
