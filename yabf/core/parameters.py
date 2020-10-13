"""Module defining parameter objects."""
import attr
import numpy as np
from attr import converters as cnv
from attr import validators as vld
from cached_property import cached_property


def tuplify(x):
    if isinstance(x, str):
        return (x,)
    else:
        try:
            return tuple(x)
        except TypeError:
            return (x,)


def texify(name):
    if name.count("_") == 1:
        sub = name.split("_")[1]
        sub = r"{\rm %s}" % sub
        name = name.split("_")[0] + "_" + sub
    return name


@attr.s(frozen=True)
class Parameter:
    """
    A fiducial parameter of a model.

    Used to *define* the model and its defaults.
    A Parameter of a model does not mean it *will* be
    constrained, but rather that it *can* be constrained. To be constrained,
    a :class:`Param` must be set on the instance at run-time.

    min/max in this class specify the total physically/logically allowable domain
    for the parameter. This can be reduced via the specification of :class:`Param`
    at run-time.
    """

    name = attr.ib(validator=vld.instance_of(str))
    fiducial = attr.ib(converter=float, validator=vld.instance_of(float))
    min = attr.ib(
        -np.inf,
        type=float,
        converter=float,
        validator=vld.instance_of(float),
        kw_only=True,
    )
    max = attr.ib(
        np.inf,
        type=float,
        converter=float,
        validator=vld.instance_of(float),
        kw_only=True,
    )
    latex = attr.ib(validator=vld.instance_of(str), kw_only=True)

    @latex.default
    def _ltx_default(self):
        return texify(self.name)


@attr.s(frozen=True)
class Param:
    """Specification of a parameter that is to be constrained."""

    name = attr.ib()
    fiducial = attr.ib(
        None,
        type=float,
        converter=cnv.optional(float),
        validator=vld.optional(vld.instance_of(float)),
    )
    min = attr.ib(
        -np.inf,
        converter=float,
        type=float,
        validator=vld.instance_of(float),
        kw_only=True,
    )
    max = attr.ib(
        np.inf,
        type=float,
        converter=float,
        validator=vld.instance_of(float),
        kw_only=True,
    )
    latex = attr.ib(kw_only=True)
    ref = attr.ib(None, kw_only=True)
    prior = attr.ib(None, kw_only=True)
    determines = attr.ib(
        converter=tuplify,
        kw_only=True,
        validator=vld.deep_iterable(vld.instance_of(str)),
    )
    transforms = attr.ib(converter=tuplify, kw_only=True)

    @latex.default
    def _ltx_default(self):
        return texify(self.name)

    @determines.default
    def _determines_default(self):
        return (self.name,)

    @transforms.default
    def _transforms_default(self):
        return (None,) * len(self.determines)

    @transforms.validator
    def _transforms_validator(self, attribute, value):
        for val in value:
            if val is not None and not callable(val):
                raise TypeError("transforms must be a list of callables")

    @cached_property
    def is_alias(self):
        return all(pm is None for pm in self.transforms)

    @cached_property
    def is_pure_alias(self):
        return self.is_alias and len(self.determines) == 1

    def transform(self, val):
        for pm in self.transforms:
            if pm is None:
                yield val
            else:
                yield pm(val)

    def generate_ref(self, n=1):
        if self.ref is None:
            # Use prior
            if self.prior is None:
                if np.isinf(self.min) or np.isinf(self.max):
                    raise ValueError(
                        "Cannot generate reference values for active "
                        "parameter with infinite bounds: {}".format(self.name)
                    )

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
                    raise TypeError(
                        "parameter '{}' does not have a valid value for "
                        "ref".format(self.name)
                    )

        if not np.all(np.logical_and(self.min <= ref, ref <= self.max)):
            raise ValueError(
                f"param {self.name} produced a reference value outside " f"its domain."
            )

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

    def clone(self, **kwargs):
        return attr.evolve(self, **kwargs)

    def new(self, p: Parameter):
        """Create a new :class:`Param`.

        Any missing info from this instance filled in by the given instance.
        """
        assert isinstance(p, Parameter)
        assert self.determines == (p.name,)

        if len(self.determines) > 1:
            raise ValueError("Cannot create new Param if it is not just an alias")

        default_range = (list(self.transform(p.min))[0], list(self.transform(p.max))[0])

        return Param(
            name=self.name,
            fiducial=self.fiducial if self.fiducial is not None else p.fiducial,
            min=max(self.min, min(default_range)),
            max=min(self.max, max(default_range)),
            latex=self.latex
            if (self.latex != self.name or self.name != p.name)
            else p.latex,
            ref=self.ref,
            prior=self.prior,
            determines=self.determines,
            transforms=self.transforms,
        )
