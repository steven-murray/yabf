import attr
import numpy as np
import yaml

@attr.s(frozen=True)
class Parameter:
    """
    A fiducial parameter of a reduce. Used to *define* the reduce
    and its defaults. A Parameter of a reduce does not mean it *will* be
    constrained, but rather that it *can* be constrained. To be constrained,
    a :class:`Param` must be set on the instance at run-time.

    min/max in this class specify the total physically/logically allowable domain
    for the parameter. This can be reduced via the specification of :class:`Param`
    at run-time.
    """
    name = attr.ib()
    fiducial = attr.ib()
    min = attr.ib(-np.inf, type=float)
    max = attr.ib(np.inf, type=float)
    latex = attr.ib()

    @latex.default
    def _ltx_default(self):
        return self.name


@attr.s(frozen=True)
class Param(Parameter):
    """
    Specification of a parameter that is to be constrained.
    """
    fiducial = attr.ib()
    ref = attr.ib(None)
    prior = attr.ib(None)
    alias_for = attr.ib()

    @fiducial.default
    def _fid_default(self):
        return None

    @alias_for.default
    def _alias_default(self):
        return self.name

    @alias_for.validator
    def _alias_validator(self, attribute, value):
        assert isinstance(value, str) or hasattr(value, "__iter__")

    def generate_ref(self, n=1):
        if self.ref is None:
            # Use prior
            if self.prior is None:
                if np.isinf(self.min) or np.isinf(self.max):
                    raise ValueError("Cannot generate reference values for active parameter with infinite bounds")

                ref = np.random.uniform(self.min, self.max, size=n)
            else:
                try:
                    ref = self.prior.rvs(size=n)
                except AttributeError:
                    raise NotImplementedError("callable priors not yet implemented")
        else:
            try:
                ref = self.ref.rvs(size=n)
            except AttributeError:
                try:
                    ref = self.ref(size=n)
                except TypeError:
                    raise TypeError("parameter '{}' does not have a valid value for ref".format(self.name))

        if not np.all(np.logical_and(self.min <= ref, ref <= self.max)):
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


    def new(self, p, aliases=None):
        """
        Create a new Param instance with any missing info from this
        instance filled in by the given instance.
        """
        assert isinstance(p, Parameter)
        assert p.name == self.alias_for

        return Param(
            name=self.name,
            fiducial=self.fiducial if self.fiducial is not None else p.fiducial,
            min=max(self.min, p.min),
            max=min(self.max, p.max),
            latex=self.latex if (self.latex != self.name or self.name != p.name) else p.latex,
            ref=self.ref,
            prior=self.prior,
            alias_for=aliases or self.alias_for
        )
