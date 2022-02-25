"""Module defining parameter objects."""
from __future__ import annotations

import attr
import numpy as np
import yaml
from attr import NOTHING
from attr import converters as cnv
from attr import validators as vld
from cached_property import cached_property
from collections import OrderedDict
from scipy import stats
from typing import Callable, List, Sequence, Tuple, Union

from .typing import numeric


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


def positive(inst, att, val):
    if val <= 0:
        raise ValueError(f"Must be positive! Got {val}")


@attr.s(frozen=True)
class Parameter:
    """
    A potential parameter of a model.

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


def _tuple_or_float(inst, att, val):
    if val is not None:
        try:
            float(val)
        except TypeError:
            val = tuple(float(v) if v is not None else None for v in val)
            assert len(val) == inst.length


@attr.s(frozen=True)
class ParameterVector:
    """
    A fiducial vector of parameters in a model.

    This is useful for defining a number of similar parameters
    all at once (eg. coefficients of a polynomial).
    """

    name = attr.ib(validator=vld.instance_of(str))
    length = attr.ib(validator=[vld.instance_of(int), positive])
    _fiducial = attr.ib(validator=_tuple_or_float)
    _min = attr.ib(-np.inf, validator=_tuple_or_float, kw_only=True)
    _max = attr.ib(np.inf, validator=_tuple_or_float, kw_only=True)
    latex = attr.ib(validator=vld.instance_of(str), kw_only=True)

    def _tuplify(self, val):
        if not hasattr(val, "__len__"):
            return (val,) * self.length
        else:
            return tuple(float(v) for v in val)

    @property
    def fiducial(self) -> tuple[float]:
        return self._tuplify(self._fiducial)

    @property
    def min(self) -> tuple[float]:
        return self._tuplify(self._min)

    @property
    def max(self) -> tuple[float]:
        return self._tuplify(self._max)

    @latex.default
    def _ltx_default(self):
        return texify(self.name) + "_%s"

    def get_params(self) -> tuple[Parameter]:
        """Return all individual parameters from this vector."""
        return tuple(
            Parameter(
                name=f"{self.name}_{i}",
                fiducial=self.fiducial[i],
                min=self.min[i],
                max=self.max[i],
                latex=self.latex % i,
            )
            for i in range(self.length)
        )


@attr.s(frozen=True)
class Param:
    """Specification of a parameter that is to be constrained."""

    name = attr.ib()
    _min: numeric = attr.ib(-np.inf)
    _max: numeric = attr.ib(np.inf)

    prior = attr.ib(
        kw_only=True,
        validator=vld.optional(vld.instance_of(stats.distributions.rv_frozen)),
    )

    fiducial = attr.ib(
        None,
        type=float,
        converter=cnv.optional(float),
        validator=vld.optional(vld.instance_of(float)),
        kw_only=True,
    )
    latex = attr.ib(kw_only=True)
    ref = attr.ib(kw_only=True)
    determines = attr.ib(
        converter=tuplify,
        kw_only=True,
        validator=vld.deep_iterable(vld.instance_of(str)),
    )
    transforms = attr.ib(converter=tuplify, kw_only=True)

    @latex.default
    def _ltx_default(self):
        return texify(self.name)

    @ref.default
    def _ref_default(self):
        return self.prior

    @prior.default
    def _prior_default(self) -> stats.distributions.rv_frozen | None:
        if np.isinf(self._min) or np.isinf(self._max):
            return None

        return stats.uniform(self._min, self._max - self._min)

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

    @property
    def min(self) -> float:
        """The minimum boundary of the prior, helpful for constraints."""
        if self.prior is None:
            return self._min
        elif isinstance(self.prior, type(stats.uniform(0, 1))):
            return self.prior.support()[0]
        else:
            return -np.inf

    @property
    def max(self) -> float:
        """The maximum boundary of the prior, helpful for constraints."""
        if self.prior is None:
            return self._max
        elif isinstance(self.prior, type(stats.uniform(0, 1))):
            return self.prior.support()[1]
        else:
            return np.inf

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
            raise ValueError("Must specify a valid function for ref to generate refs.")

        try:
            ref = self.ref.rvs(size=n)
        except AttributeError:
            try:
                ref = self.ref(size=n)
            except TypeError:
                raise TypeError(
                    f"parameter '{self.name}' does not have a valid value for ref"
                )

        if np.any(self.prior.pdf(ref) == 0):
            raise ValueError(
                f"param {self.name} produced a reference value outside its domain."
            )

        return ref

    def logprior(self, val):
        if self.prior is None:
            if self._min > val or self._max < val:
                return -np.inf
            else:
                return 0

        return self.prior.logpdf(val)

    def clone(self, **kwargs):
        return attr.evolve(self, **kwargs)

    def new(self, p: Parameter) -> Param:
        """Create a new :class:`Param`.

        Any missing info from this instance filled in by the given instance.
        """
        assert isinstance(p, Parameter)
        assert self.determines == (p.name,)

        if len(self.determines) > 1:
            raise ValueError("Cannot create new Param if it is not just an alias")

        if len(self.transforms) == 1 and self.transforms[0] is None:
            # don't know how to do this if transforms are given
            pmin = max(self._min, p.min)
            pmax = min(self._max, p.max)
        else:
            tr = (
                list(self.transform(self._min))[0],
                list(self.transform(self._max))[0],
            )
            if not (p.min <= tr[0] <= p.max and p.min <= tr[1] <= p.max):
                raise ValueError(
                    f"The defined support for '{self.name}' ({self._min}-{self._max}) transforms to {min(tr)-max(tr)}, which is outside the support of its determined parameter '{p.name}', which has range {p.min}-{p.max}"
                )
            pmin, pmax = self._min, self._max

        return Param(
            name=self.name,
            min=pmin,
            max=pmax,
            fiducial=self.fiducial if self.fiducial is not None else p.fiducial,
            latex=self.latex
            if (self.latex != self.name or self.name != p.name)
            else p.latex,
            ref=self.ref or attr.NOTHING,
            prior=self.prior or attr.NOTHING,
            determines=self.determines,
            transforms=self.transforms,
        )

    def __getstate__(self):
        """Obtain a simple input state of the class that can initialize it."""
        out = attr.asdict(self)

        if self.transforms == (None,):
            del out["transforms"]
        if self.ref is None:
            del out["ref"]
        if self.determines == (self.name,):
            del out["determines"]
        if self.latex == self.name:
            del out["latex"]

        return out

    def as_dict(self):
        """Simple representation of the class as a dict.

        No "name" is included in the dict.
        """
        out = self.__getstate__()
        del out["name"]
        return out


def iterable_or_scalar(tp):
    def validator(self, att, val):
        if hasattr(val, "__len__"):
            assert all(isinstance(v, tp) for v in val)
        else:
            assert isinstance(val, tp), f"required '{tp}', got '{type(val)}'."

    return validator


@attr.s(frozen=True)
class ParamVec:
    name = attr.ib(validator=vld.instance_of(str))
    min: Sequence[numeric] | numeric = attr.ib(
        -np.inf, validator=iterable_or_scalar((float, int))
    )
    max: Sequence[numeric] | numeric = attr.ib(
        np.inf, validator=iterable_or_scalar((float, int))
    )
    prior = attr.ib(
        None,
        kw_only=True,
        validator=vld.optional(iterable_or_scalar(stats.distributions.rv_frozen)),
    )
    fiducial = attr.ib(None, validator=vld.optional(iterable_or_scalar((float, int))))
    latex = attr.ib(validator=vld.instance_of(str), kw_only=True)
    ref: Sequence[Callable] | Callable | None = attr.ib(
        None,
        kw_only=True,
        validator=vld.optional(iterable_or_scalar(stats.distributions.rv_frozen)),
    )
    length: int = attr.ib(validator=vld.instance_of(int))

    def _tuplify(self, val):
        if not hasattr(val, "__len__"):
            return (val,) * self.length
        else:
            return tuple(float(v) if v is not None else None for v in val)

    @latex.default
    def _ltx_default(self):
        return self.name

    @length.default
    def _length_default(self):
        for param in ["min", "max", "prior", "fiducial", "ref"]:
            if hasattr(getattr(self, param), "__len__"):
                return len(getattr(self, param))

        raise ValueError("You must provide length if none of the inputs are sequences.")

    def __attrs_post_init__(self):
        """Perform post-init validation."""
        # Ensure all the inputs are the same length, if they are sequences
        for param in ["min", "max", "prior", "fiducial", "ref"]:
            p = getattr(self, param)
            if hasattr(p, "__len__") and len(p) != self.length:
                raise ValueError(
                    f"ParamVec '{self.name}' has incompatible length for '{param}'"
                )

    def get_params(self) -> tuple[Param]:
        """Return a tuple of active Params for this vector."""

        def get(name, i):
            p = getattr(self, name)
            return p[i] if hasattr(p, "__len__") else p

        return tuple(
            Param(
                name=self.name % i if "%s" in self.name else f"{self.name}_{i}",
                fiducial=get("fiducial", i),
                min=get("min", i),
                max=get("max", i),
                ref=get("ref", i),
                prior=get("prior", i) or attr.NOTHING,
                latex=self.latex % i if (self.latex != self.name) else attr.NOTHING,
            )
            for i in range(self.length)
        )


@attr.s
class Params:
    _param_list = attr.ib(converter=tuple)

    @_param_list.validator
    def _param_list_vld(self, att, val):
        for v in val:
            if not isinstance(v, Param):
                raise ValueError("params must be a sequence of Param objects")

    def __attrs_post_init__(self):
        """Save the parameters in ordered dictionary form."""
        self._prm_dict = OrderedDict((p.name, p) for p in self._param_list)

    def __getitem__(self, item: str | int):
        """Make the params like a dictionary AND a list.

        Parameters
        ----------
        item
            Either a string specifying a parameter name, or
            an integer specifying its place in the parameter list.

        Returns
        -------
        Param
            The parameter instance.
        """
        if isinstance(item, int):
            return self._param_list[item]

        return self._prm_dict[item]

    def __getattr__(self, item: str) -> Param:
        """Get an attribute.

        Parameters
        ----------
        item
            The parameter name to retrieve.

        Returns
        -------
        Param
            The parameter instance.

        Raises
        ------
        AttributeError
            If the parameter name does not exist.
        """
        if item in super().__getattribute__("_prm_dict"):
            return self._prm_dict[item]
        else:
            raise AttributeError

    def items(self):
        """Equivalent to ``dict.items()``.

        Yields
        ------
        item
            A 2-tuple of (str, :class:`Param`) for each parameter.
        """
        yield from self._prm_dict.items()

    def keys(self):
        """Equivalnet to ``dict.keys()``.

        Yields
        ------
        key
            String parameter names.
        """
        yield from self._prm_dict.keys()

    def values(self):
        """Equivalent to ``dict.values()``.

        Yields
        ------
        :class:`Param`
            The parameter instances.
        """
        yield from self._prm_dict.values()

    def __len__(self) -> int:
        """The number of parameters."""
        return len(self._param_list)

    def __contains__(self, key: str | Param) -> bool:
        """Whether the given parameter exists in this group.

        Parameters
        ----------
        key
            Either a parameter name, or :class:`Param` instance to
            check for existence.

        Returns
        -------
        bool
            True if the parameter is in the instance, False otherwise.
        """
        return isinstance(key, (str, Param)) and key in self._prm_dict

    def __add__(self, x: tuple[Param] | Params) -> Params:
        """Magic method for adding two :class:`Param` instances.

        Parameters
        ----------
        x :
            The object to add to this instance. If a tuple,
            must be a tuple of parameters.

        Returns
        -------
        :class:`Param`
            A new instance with all the parameters in it.
        """
        x = Params(x)
        return Params(self._param_list + x._param_list)

    def as_dict(self) -> dict:
        """Represent the object as a plain dictionary.

        Useful for serializing.

        Returns
        -------
        dict
            The serialized dictionary.
        """
        return {name: param.as_dict() for name, param in self._prm_dict.items()}

    def to_yaml(self):
        """Convert the params to a YAML-style representation."""
        return "\n".join(
            yaml.dump(
                {name: attr.asdict(param, filter=lambda att, val: att.name != "name")}
            )
            for name, param in self._prm_dict.items()
        )
