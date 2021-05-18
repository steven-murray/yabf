"""Framework for MCMC Components."""
import attr
import collections
from abc import ABC
from cached_property import cached_property
from collections import OrderedDict
from copy import deepcopy
from frozendict import frozendict
from typing import Dict, List, Optional, Tuple

from . import utils
from .parameters import Param, Params


def _only_active(func):
    def func_wrapper(self, *args, **kwargs):
        if not self.in_active_mode:
            raise AttributeError(
                f"{func.__name__} is not available when not in active mode."
            )
        return func(self, *args, **kwargs)

    return func_wrapper


class DependencyError(ValueError):
    pass


class _ComponentTree(ABC):
    """A base class for all components and their containers."""

    @cached_property
    def _in_active_mode(self) -> bool:
        return bool(self.child_active_params)

    @cached_property
    def _subcomponents(self):
        pass

    @cached_property
    def child_components(self) -> OrderedDict:
        out = OrderedDict([(cmp.name, cmp) for cmp in self._subcomponents])
        for cmp in self._subcomponents:
            out.update(
                [(cmp.name + "." + k, v) for k, v in cmp.child_components.items()]
            )
        return out

    @cached_property
    def common_components(self) -> list:
        """All components that occur more than once."""
        common = []

        for loc, cmp in self.child_components.items():
            for i, (c, v) in enumerate(common):
                if cmp == c:
                    common[i] = (c, v + (loc,))
                    break
            else:
                common.append((cmp, (loc,)))

        # Remove non-common components
        common = [(k, v) for k, v in common if len(v) > 1]

        # Restrict to top-level
        common = [
            (k, v)
            for k, v in common
            if all(k not in kk.components for kk, vv in common)
        ]

        return common

    @cached_property
    def _subcomponent_names(self) -> Tuple[str]:
        return tuple(c.name for c in self._subcomponents)

    def __getitem__(self, item):
        return self._loc_to_component(item)

    @property
    def in_active_mode(self) -> bool:
        """Whether this class is in active mode (i.e. being actively constrained)."""
        return self._in_active_mode or any(
            cmp.in_active_mode for cmp in self._subcomponents
        )

    @cached_property
    def child_base_parameters(self) -> Tuple:
        """Tuple of all parameters in this and child components."""
        this = tuple(self.base_parameters) if hasattr(self, "base_parameters") else ()
        for cmp in self._subcomponents:
            this = this + cmp.child_base_parameters
        return this

    @cached_property
    def child_base_parameter_dct(self) -> OrderedDict:
        """Set of all parameter names in this component and its sub-components."""
        res = list(getattr(self, "base_parameter_dct", []))

        for cmp in self._subcomponents:
            this = [cmp.name + "." + child for child in cmp.child_base_parameter_dct]
            res.extend(this)

        return OrderedDict(
            [(loc, param) for loc, param in zip(res, self.child_base_parameters)]
        )

    def _loc_to_component(self, loc: str):
        """Take a string loc and return a sub-component based on the string.

        i.e. "foo.bar" would return the component named "bar" in the component
        named "foo" in self.
        """
        if loc == "":
            return self

        locs = loc.split(".")

        scs = {cmp.name: cmp for cmp in self._subcomponents}

        if len(locs) == 1:
            return scs[locs[0]]

        for cmp in self._subcomponents:
            try:
                return cmp._loc_to_component(".".join(locs[1:]))
            except KeyError:
                pass
        raise KeyError(f"loc '{loc}' does not exist in any subcomponents")

    def _loc_to_parameter(self, loc: str):
        """
        Take a string loc and return a base parameter based on the string.

        i.e. "foo.bar" would return the parameter named "bar" in the component
        named "foo" in self.
        """
        component, _, paramname = loc.rpartition(".")
        cmp = self._loc_to_component(component)

        try:
            return cmp.base_parameter_dct[paramname]
        except KeyError:
            raise KeyError(
                "param '{}' does not exist in component named '{}'".format(
                    paramname, cmp.name
                )
            )

    def get_determination_graph(self):
        p = getattr(self, "active_params", ())

        # Create tuple of active parameters that point to the right base param location
        # from _this_ component.
        for cmp in self._subcomponents:
            this = tuple(
                attr.evolve(
                    apar, determines=[cmp.name + "." + d for d in apar.determines]
                )
                for apar in cmp.child_active_params
            )

            p = p + this
        return p

    @cached_property
    def child_active_params(self):
        """Tuple of all active parameters in this component and its sub-components.

        Note that this will also check if child-components have params of the same
        name, and de-duplicate. The child active params will have correct dot-path
        names to their "determines" locations, and the correct number of mappings.
        """
        p = self.get_determination_graph()

        # Group params by name (and check that they're consistent)
        groups = OrderedDict()
        for param in p:
            if param.name in groups:
                # Ensure that this new param is functionally equivalent.
                that = groups[param.name][0]
                for item in ["fiducial", "min", "max", "latex"]:  # 'ref', 'prior'
                    # TODO: we don't check ref and prior here, because they're hard to
                    #       check properly. This will just use the first entry silently.
                    if getattr(param, item) != getattr(that, item):
                        raise ValueError(
                            f"two params with name '{param.name}' provided with "
                            f"different {item} values: ({getattr(param, item)}, "
                            f"{getattr(that, item)})"
                        )
                groups[param.name].append(param)
            else:
                groups[param.name] = [param]

        # Now we need to combine the grouped params, and in doing so, combine
        # their parameter mappings.
        out = []
        for params in groups.values():
            if len(params) == 1:
                out.append(params[0])
            else:
                out.append(
                    attr.evolve(
                        params[0],
                        determines=sum((list(p.determines) for p in params), []),
                        transforms=sum((list(p.transforms) for p in params), []),
                    )
                )

        return Params(tuple(out))

    # @cached_property
    # def child_active_param_dct(self):
    #     return OrderedDict([(p.name, p) for p in self.child_active_params])

    @cached_property
    def total_active_params(self):
        return len(self.child_active_params)

    def _active_param_locgen(self, param):
        for name in param.determines:
            paramname = name.rpartition(".")[-1]
            yield name, paramname

    def _fiducial_params(self, transform=True) -> Dict[str, float]:
        """Dictionary of fiducial parameters for all subcomponents."""
        dct = {}

        for aparam in self.child_active_params:
            if aparam.fiducial:
                if transform:
                    vals = aparam.transform(aparam.fiducial)
                else:
                    vals = [aparam.fiducial] * len(aparam.determines)

                for name, v in zip(aparam.determines, vals):
                    dct = utils.add_loc_to_dict(dct, name, v)

        for param in getattr(self, "base_parameters", []):
            if param.name in dct:
                continue
            elif param.name in self.fiducial:
                dct[param.name] = self.fiducial[param.name]
            else:
                dct[param.name] = param.fiducial

        for cmp in self._subcomponents:
            dct[cmp.name] = cmp._fiducial_params(transform=transform)

        return dct

    @cached_property
    def fiducial_params(self) -> Dict[str, float]:
        return self._fiducial_params()

    @cached_property
    def fiducial_params_untransformed(self) -> Dict[str, float]:
        return self._fiducial_params(transform=False)

    def _fill_params(self, params=None, transform=True):
        if isinstance(params, frozendict):
            return params

        if params is None:
            params = {}

        if transform:
            fiducial = deepcopy(self.fiducial_params)
        else:
            fiducial = deepcopy(self.fiducial_params_untransformed)

        using_list = False
        if not isinstance(params, dict):
            using_list = True
            try:
                params = self._parameter_list_to_dict(params, transform=transform)
            except ValueError:
                raise ValueError(
                    f"params must be a dict or list with same length as"
                    f"child_active_params (and in that order). \n"
                    f"Active params: {self.child_active_params.keys()}\n"
                    f"Received Params: {params}"
                )

        for k, v in list(params.items()):
            # Treat params first as if they are active
            if not using_list and k in self.child_active_params:
                aparam = self.child_active_params[k]
                vals = (
                    aparam.transform(v) if transform else [v] * len(aparam.determines)
                )
                for name, v in zip(aparam.determines, vals):
                    fiducial = utils.add_loc_to_dict(
                        fiducial, name, v, raise_if_not_exist=True
                    )

            else:
                # It might be dot-pathed loc to an actual parameter or sub-dict of such.
                fiducial = utils.add_loc_to_dict(
                    fiducial, k, v, raise_if_not_exist=True
                )

        # Now turn it into a frozendict, to signify that it's full
        return utils.recursive_frozendict(fiducial)

    @_only_active
    def _parameter_list_to_dict(self, p, transform=True):
        """Define the order in which the parameters should arrive.

        The same order as :func:`active_params`.

        It only makes sense to call in active mode, and p must be a list
        or array of the same length as :func:`child_active_params`.

        Typically, this will be internally called in the process of fitting
        parameters.

        Returns
        -------
        dict : each parameter is inserted at the various
        locations at which it is defined (via its aliases).
        """
        if len(p) != self.total_active_params:
            raise ValueError(
                "length of parameter list not equal to length of child_active_params"
            )

        dct = {}
        for pvalue, apar in zip(p, self.child_active_params):
            if transform:
                ptrans = apar.transform(pvalue)
            else:
                ptrans = [pvalue] * len(apar.determines)

            for loc, ptran in zip(apar.determines, ptrans):
                dct = utils.add_loc_to_dict(dct, loc, ptran)
        return dct

    @_only_active
    def generate_refs(self, n=1, squeeze=False, params=None):
        """
        Generate reference values for active parameters from their priors.

        Parameters
        ----------
        n : int, optional
            Number of reference values to choose for each parameter.
        squeeze : bool, optional
            If ``n=1``, ``squeeze`` will ensure that the returned results
            are one-dimensional.
        params : list, optional
            The parameters to include in the generated refs. Default is to
            include all (active) parameters. Only active parameters are allowed.

        Returns
        -------
        list, list-of-lists, dict or dict-of-lists :
            If `full` is False, returns a list. If `n=1` and `squeeze` is True,
            this is a list of parameter values (by default, active parameters).
            Otherwise, it is a list of lists, where each sub-list contains n
            values of the parameter. If `full` is True, returns a dict, where
            each key is the name of a parameter.
        """
        if params is None:
            params = self.child_active_params
        else:
            if any(p not in self.child_active_params for p in params):
                raise ValueError(
                    "Only currently active parameters may be specified in params"
                )

            params = [self.child_active_params[name] for name in params]

        refs = []

        for param in params:
            ref = param.generate_ref(n)

            if not squeeze and n == 1:
                ref = [ref]

            refs.append(ref)

        return refs

    def __getstate__(self):
        """
        Set the YAML/pickle state.

        Sets it such that only the initial values passed to the constructor are actually
        saved. In this sense, this class is "frozen",
        though there are some attributes which are lazy-loaded after instantiation.
        These don't need to be written.
        """
        return {k: v for k, v in self.__dict__.items() if k in attr.asdict(self)}


@attr.s(kw_only=True, frozen=True)
class ParameterComponent(_ComponentTree):
    """Base class for named components and likelihoods that take parameters.

    Parameters
    ----------
    name : str, optional
        A name for the component. Default is the class name.
    fiducial: dict, optional
        Fiducial values for parameters of this particular object (i.e. none of its
        children), and which aren't given as active parameters. Otherwise their
        fiducial values are inherited from the Component definition.
    params: tuple of :class:`~parameters.Params` instances or dict
        Definition of which parameters are to be treated as "active" (i.e. fitted for).
        These parameters receive special status in some methods.
        Can be a dictionary with parameter names as keys.
    derived: tuple of strings or callables
        A tuple defining which derived parameters may be obtained by calling the
        :func:`derived_quantities` method. If str, there must be a method within
        the object by that name, else if a callable, it must receive a number of
        arguments which depends on the kind of class this is. It will typically
        be ctx and kwargs for params, or perhaps a model, ctx and params.
    """

    def clone(self, **kwargs):
        return attr.evolve(self, **kwargs)

    @staticmethod
    def param_converter(val):
        if isinstance(val, collections.abc.Mapping):
            return Params(tuple(Param(name=k, **v) for k, v in val.items()))

        out = []
        for v in val:
            if isinstance(v, str):
                out.append(Param(name=v))
            elif isinstance(v, Param):
                out.append(v)
            else:
                raise ValueError(
                    f"Elements of params must be str or Param. Got {type(v)}"
                )
        return Params(tuple(out))

    _name = attr.ib(
        validator=attr.validators.optional(attr.validators.instance_of(str))
    )
    fiducial = attr.ib(factory=frozendict, converter=frozendict)
    params = attr.ib(factory=tuple, converter=param_converter.__func__)
    derived = attr.ib(factory=tuple, converter=tuple)

    base_parameters = ()

    def __init_subclass__(cls, **kwargs):
        """Enable plugin ability."""
        super().__init_subclass__()

    def __attrs_post_init__(self):
        """Perform checks on parameters."""
        if len(set(self.child_base_parameter_dct.keys())) != len(
            self.child_base_parameter_dct.keys()
        ):
            raise NameError(
                f"One or more of the parameter paths from {self.name} is not unique: "
                f"{self.child_base_parameter_dct.keys()}"
            )

        if len(self.base_parameter_dct) != len(self.base_parameters):
            raise ValueError(
                f"There are two parameters with the same name in {self.__class__.__name__}: "
                f"{self.base_parameter_dct.keys()}"
            )

    def _get_subcomponent_names(self):
        return [self.name] + sum(
            (cmp._get_subcomponent_names() for cmp in self._subcomponents), []
        )

    @_name.default
    def _name_default(self):
        return self.__class__.__name__

    @cached_property
    def name(self):
        """Name of the component."""
        return self._name or self.__class__.__name__

    @params.validator
    def _params_validator(self, attribute, value):
        assert all(isinstance(p, Param) for p in value)
        # Ensure unique names, and unique determines
        names = [p.name for p in value]
        if len(set(names)) != len(names):
            raise NameError(f"not all param names are unique in {self.name}: {names}")

        determines = sum((list(p.determines) for p in value), [])
        if len(set(determines)) != len(determines):
            raise ValueError(
                f"different parameters determine the same base parameter "
                f"in {self.name}:{determines}"
            )

        if any(d not in self.base_parameter_dct for d in determines):
            raise ValueError(
                f"One or more params do not map to any known Parameter in {self.name}: "
                f"{determines}. Known params: {list(self.base_parameter_dct.keys())}"
            )

    @fiducial.validator
    def _fiducial_validator(self, att, value):
        for name in value:
            if name in self.params:
                raise ValueError(
                    "Pass fiducial values to constrained parameters "
                    "inside the Param class"
                )
            if name not in self.base_parameter_dct:
                raise KeyError(
                    f"Fiducial parameter {name} does not match any parameters of "
                    f"{self.name}. Note that fiducial parameters must be passed at "
                    f"the level to which they belong."
                )

    @derived.validator
    def _derived_validator(self, att, val):
        self._validate_derived(val)

    def _validate_derived(self, val):
        for d in val:
            assert callable(d) or (
                isinstance(d, str) and hasattr(self, d)
            ), f"{d} is not a valid derived parameter"

    @cached_property
    def base_parameter_dct(self):
        """
        All possible parameters of this specific component or likelihood.

        Not just parameters that are being constrained.
        """
        return {p.name: p for p in self.base_parameters}

    @cached_property
    def active_params(self) -> Params:
        """
        Tuple of actively constrained parameters.

        Represents only those specified for this component or likelihood only.
        Note that this is just the parameters themselves.
        """
        out = []
        for v in self.params:
            if len(v.determines) == 1:
                out.append(v.new(self.base_parameter_dct[v.determines[0]]))
            else:
                out.append(v)

        return Params(tuple(out))

    # @cached_property
    # def active_params_dct(self) -> OrderedDict:
    #     """Actively constrained parameters in this component.

    #     Keys are string names, and values are the :class:`~Param` objects.
    #     """
    #     return OrderedDict([(p.name, p) for p in self.active_params])

    @cached_property
    def _base_to_param_mapping(self) -> dict:
        """Dict of base:active params in *this* component."""
        mapping = OrderedDict()
        for p in self.base_parameters:
            for prm in self.active_params:
                if p.name in prm.determines:
                    mapping[p] = prm
        return mapping


@attr.s(frozen=True, kw_only=True)
class Component(ParameterComponent):
    """A component of a likelihood.

    These are mainly for re-usability, so they can be mixed and matched inside
    likelihoods. Multiple components can be used in a single likelihood, providing the
    ability to compute different elements.
    """

    components = attr.ib(factory=tuple, converter=tuple, kw_only=True)

    provides = ()
    _plugins = {}

    def __init_subclass__(cls, is_abstract=False, **kwargs):
        """Provide plugin capablity and do some verification of a defined plugin."""
        super().__init_subclass__(**kwargs)

        # Plugin framework
        if not is_abstract:
            cls._plugins[cls.__name__] = cls

        if type(cls.provides) not in [list, set, tuple, cached_property]:
            raise TypeError(
                f"Component {cls.__name__} must define a list/set/tuple/cached_property "
                f"for 'provides'. Instead, it is {type(cls.provides)}"
            )

        if type(cls.provides) in [list, set, tuple]:
            for pr in cls.provides:
                if type(pr) is not str:
                    raise ValueError(
                        "provides should be an ordered iterable of strings"
                    )

    @components.validator
    def _cmp_valid(self, att, val):
        for cmp in val:
            assert isinstance(
                cmp, Component
            ), f"component {cmp.name} is not a valid Component"

    def derived_quantities(self, ctx=None, params=None):
        if ctx is None:
            ctx = self()

        dquants = []
        for d in self.derived:
            if isinstance(d, str):
                dquants.append(getattr(self, d)(ctx, **params))
            elif callable(d):
                dquants.append(d(ctx, **params))
            else:
                raise ValueError(f"{d} is not a valid entry for derived")

        return dquants

    @cached_property
    def _subcomponents(self):
        return self.components

    def calculate(self, ctx=None, **params):
        """Perform the main calculation of this component.

        Parameters
        ----------
        ctx
            The context into which to place the result of the calculation.

        Other Parameters
        ----------------
        Anything else must be a keyword corresponding to a named parameter of the
        component.
        """
        pass

    def __call__(
        self,
        params: [None, list, dict] = None,
        ctx: [None, dict] = None,
        ignore_components: [None, list] = None,
    ) -> dict:
        """Perform the calculation of this and all subcomponents.

        Results are returned as a dictionary, which can be pre-filled.

        Parameters
        ----------
        params
            The parameter values to use in the calculation. If a list (or tuple), must
            be of the same length as the active params of this component. If a dict,
            keys must be strings corresponding to the names of the active parameters,
            but not all parameters must be present (non-present keys are given default
            values specified by the component itself, or instance-level
            :attr:`fiducial_params`). By default, use all fiducial parameters.
        ctx
            An optional dictionary of calculated data into which will be inserted the
            results of this calculation.
        ignore_components
            A list of names of (sub-)components to *not* run.

        Returns
        -------
        ctx
            All the results under specific keys (defined by each sub-component).
        """
        params = self._fill_params(params)

        if ctx is None:
            ctx = {}

        # First get ignored components
        if ignore_components is None:
            ignore_components = [cmp.name for cmp, _ in self.common_components]

            for cmp, locs in self.common_components:
                ctx.update(
                    cmp(ctx=ctx, params=utils.get_loc_from_dict(params, locs[0]))
                )

        for cmp in self._subcomponents:
            if cmp.name in ignore_components:
                if any(name not in ctx for name in cmp.provides):
                    raise ValueError(
                        f"attempting to ignore component '{cmp.name}' in '{self.name}' "
                        f"without providing appropriate context"
                    )
                continue
            else:
                try:
                    ctx.update(
                        cmp(
                            params=params[cmp.name],
                            ctx=ctx,
                            ignore_components=ignore_components,
                        )
                    )
                except KeyError:
                    raise KeyError(
                        f"In component '{self.name}' params does not have key "
                        f"'{cmp.name}'. Available: {list(params.keys())}"
                    )

        res = self.calculate(ctx, **params)

        if res is None:
            if self.provides:
                raise ValueError(
                    "component {} says it provides {} but does not return anything "
                    "from calculate()".format(self.name, self.provides)
                )
        else:
            if type(res) != tuple:
                res = (res,)

            if len(self.provides) != len(res):
                raise ValueError(
                    f"{self.name} [{self.__class__.__name__}] does not return an "
                    f"ordered iterable of the same length as its 'provides' attribute "
                    f"from calculate(). Provides: {self.provides}. Returned elements: "
                    f"{len(res)}."
                )

            ctx.update({p: r for p, r in zip(self.provides, res)})
        return ctx
