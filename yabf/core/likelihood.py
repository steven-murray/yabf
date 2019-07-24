"""
Framework for MCMC likelihoods (and parameters).
"""
from abc import ABC
from collections import OrderedDict
from copy import deepcopy
import collections

import attr
import numpy as np
from attr import validators
from cached_property import cached_property

from . import mpi
from . import utils
from .parameters import Param
from frozendict import frozendict


def _only_active(func):
    def func_wrapper(self, *args, **kwargs):
        if not self.in_active_mode:
            raise AttributeError("{} is not available when not in active mode.".format(func.__name__))
        return func(self, *args, **kwargs)

    return func_wrapper


class DependencyError(ValueError):
    pass


class _ComponentTree(ABC):
    """
    A base class for all components and their containers.
    """

    @cached_property
    def _in_active_mode(self):
        return bool(self.child_active_params)

    @cached_property
    def _subcomponents(self):
        pass

    @cached_property
    def child_components(self):
        out = OrderedDict([(cmp.name, cmp) for cmp in self._subcomponents])
        for cmp in self._subcomponents:
            out.update([(cmp.name + "." + k, v) for k, v in cmp.child_components.items()])
        return out

    @cached_property
    def common_components(self):
        """
        All components that occur more than once.
        """
        common = []

        for loc, cmp in self.child_components.items():
            for i, (c,v) in enumerate(common):
                if cmp == c:
                    common[i] = (c, v + (loc,))
                    break
                elif i == len(common) - 1:
                    common.append((cmp, (loc,)))

        # Remove non-common components
        common = [(k, v) for k, v in common if len(v) > 1]

        # Restrict to top-level
        common = [(k, v) for k, v in common if not any([k in kk.components for kk, vv in common])]

        return common

    @cached_property
    def _subcomponent_names(self):
        return tuple([c.name for c in self._subcomponents])

    def __getitem__(self, item):
        return self._loc_to_component(item)

    @property
    def in_active_mode(self):
        """
        Bool representing whether this class is in active mode (i.e. whether
        it is being actively constrained).
        """
        return self._in_active_mode or any([cmp.in_active_mode for cmp in self._subcomponents])

    @cached_property
    def child_base_parameters(self):
        """
        Tuple of all parameters in this and child components.
        """
        this = tuple(self.base_parameters) if hasattr(self, "base_parameters") else tuple()
        for cmp in self._subcomponents:
            this = this + cmp.child_base_parameters
        return this

    @cached_property
    def child_base_parameter_dct(self):
        """Set of all available parameter names in this component and its sub-components"""
        res = list(getattr(self, "base_parameter_dct", []))

        for cmp in self._subcomponents:
            this = [cmp.name + "." + child for child in cmp.child_base_parameter_dct]
            res.extend(this)

        return OrderedDict([(loc, param) for loc, param in zip(res, self.child_base_parameters)])

    def _loc_to_component(self, loc: str):
        """
        Take a string loc and return a sub-component based on the string.

        i.e. "foo.bar" would return the component named "bar" in the component
        named "foo" in self.
        """
        if loc == "":
            return self
        else:
            locs = loc.split(".")

            scs = {cmp.name: cmp for cmp in self._subcomponents}

            if len(locs) == 1:
                return scs[locs[0]]

            for cmp in self._subcomponents:
                try:
                    return cmp._loc_to_component(".".join(locs[1:]))
                except KeyError:
                    pass
        raise KeyError("loc {} does not exist in any subcomponents".format(loc))

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
            raise KeyError("param '{}' does not exist in component named '{}'".format(paramname, cmp.name))

    def get_determination_graph(self):
        p = getattr(self, "active_params", tuple())

        # Create tuple of active parameters that point to the right base param location
        # from _this_ component. Note, any parameters that are given as referencing
        # something down-stream from them should also get the correct reference.
        for cmp in self._subcomponents:
            this = tuple(
                [
                    attr.evolve(
                        apar,
                        determines=[cmp.name + "." + d for d in apar.determines],
                    ) for apar in cmp.child_active_params
                ]
            )

            p = p + this
        return p

    @cached_property
    def child_active_params(self):
        """
        tuple of all active parameters in this likelihood/component and its
        sub-components

        Note that this will also check if child-components have params of the same
        name, and de-duplicate. The child active params will have correct dot-path
        names to their "determines" locations, and the correct number of mappings.
        """
        p = self.get_determination_graph()
        print("DETERMINATION GRAPH: ", [(a.name, a.determines) for a in p])

        # Group params by name (and check that they're consistent)
        groups = OrderedDict()
        for param in p:
            if param.name in groups:
                # Ensure that this new param is functionally equivalent.
                that = groups[param.name][0]
                for item in ['fiducial', 'min', 'max', 'latex']:  # 'ref', 'prior'
                    # TODO: we don't check ref and prior here, because they're hard to
                    #       check properly. This will just use the first entry silently.
                    if getattr(param, item) != getattr(that, item):
                        raise ValueError(
                            "two params with name '{}' provided with different {} "
                            "values: ({}, {})".format(
                                param.name, item, getattr(param, item),
                                getattr(that, item))
                        )
                groups[param.name].append(param)
            else:
                groups[param.name] = [param]

        # Now we need to combine the grouped params, and in doing so, combine
        # their parameter mappings.
        out = []
        for name, params in groups.items():
            if len(params) == 1:
                out.append(params[0])
            else:
                out.append(
                    attr.evolve(
                        params[0],
                        determines=sum([list(p.determines) for p in params], []),
                        parameter_mappings=sum([list(p._parameter_mappings) for p in params], [])
                    )
                )

        return tuple(out)

    @cached_property
    def child_active_param_dct(self):
        return OrderedDict([(p.name, p) for p in self.child_active_params])

    @cached_property
    def total_active_params(self):
        return len(self.child_active_params)

    def _active_param_locgen(self, param):
        for name in param.determines:
            loc, _, paramname = name.rpartition(".")
            yield name, paramname

    @cached_property
    def fiducial_params(self):
        """
        Dictionary of fiducial parameters for all subcomponents
        """
        dct = {}

        for aparam in self.child_active_params:
            if aparam.fiducial:
                vals = aparam.map_to_parameters(aparam.fiducial)

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
            dct[cmp.name] = cmp.fiducial_params

        return dct

    def _fill_params(self, params=None):
        if isinstance(params, frozendict):
            return params

        if params is None:
            params = {}

        if not isinstance(params, dict):
            try:
                params = self._parameter_list_to_dict(params)
            except ValueError:
                raise ValueError("params must be a dict or list with same length as"
                                 "child_active_params (and in that order)")
        fiducial = deepcopy(self.fiducial_params)

        for k, v in list(params.items()):
            # Treat params first as if they are active
            if k in self.child_active_param_dct:
                aparam = self.child_active_param_dct[k]
                vals = aparam.map_to_parameters(v)
                for name, v in zip(aparam.determines, vals):
                    fiducial = utils.add_loc_to_dict(fiducial, name, v,
                                                     raise_if_not_exist=True)

            else:
                # It might be dot-pathed loc to an actual parameter or sub-dict of such.
                fiducial = utils.add_loc_to_dict(
                    fiducial, k, v, raise_if_not_exist=True
                )

        # Now turn it into a frozendict, to signify that it's full
        return utils.recursive_frozendict(fiducial)

    @_only_active
    def _parameter_list_to_dict(self, p):
        """
        This defines the order in which the parameters should arrive, i.e.
        the same order as :func:`active_params`.

        It only makes sense to call in active mode, and p must be a list
        or array of the same length as the total active params of this
        component and its subcomponents.

        Typically, this will be internally called in the process of fitting
        parameters, and only by the top-level object in the likelihood
        hierarchy.

        This returns a dict, where each parameter is inserted at the various
        locations at which it is defined (via its aliases).
        """
        if len(p) != self.total_active_params:
            raise ValueError("length of parameter list not compatible with "
                             "active parameters. Please don't call "
                             "_parameter_list_to_dict yourself!")

        dct = {}
        for i, (pvalue, apar) in enumerate(zip(p, self.child_active_params)):
            for loc in apar.determines:
                dct = utils.add_loc_to_dict(dct, loc, pvalue)
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
            names = list(self.child_active_param_dct.keys())
            if any([p not in names for p in params]):
                raise ValueError("Only currently active parameters may be specified in params")

            params = [self.child_active_params[names.index(n)] for n in params]

        refs = []

        for param in params:
            ref = param.generate_ref(n)

            if not squeeze and n == 1:
                ref = [ref]

            refs.append(ref)

        return refs

    def __getstate__(self):
        """
        Set the YAML/pickle state such that only the initial values passed to
        the constructor are actually saved. In this sense, this class is "frozen",
        though there are some attributes which are lazy-loaded after instantiation.
        These don't need to be written.
        """
        return {k: v for k, v in self.__dict__.items() if k in attr.asdict(self)}


@attr.s(kw_only=True)
class ParameterComponent(_ComponentTree):
    """
    A base class for named components and likelihoods that take parameters.

    Parameters
    ----------
    name : str, optional
        A name for the component. Default is the class name.
    fiducial: dict, optional
        Fiducial values for parameters of this particular object (i.e. none of its
        children), and which aren't given as active parameters. Otherwise their
        fiducial values are inherited from the Component definition.
    params: tuple of :class:`~parameters.Params` instances
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
    @staticmethod
    def param_converter(val):
        if isinstance(val, collections.abc.Mapping):
            return tuple([Param(name=k, **v) for k,v in val.items()])
        else:
            return val

    _name = attr.ib(validator=attr.validators.optional(attr.validators.instance_of(str)))
    fiducial = attr.ib(factory=frozendict, converter=frozendict)
    params = attr.ib(factory=tuple, converter=param_converter.__func__)
    derived = attr.ib(factory=tuple, converter=tuple)

    base_parameters = tuple()


    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()

    def __attrs_post_init__(self):
        if len(set(self.child_base_parameter_dct.keys())) != len(self.child_base_parameter_dct.keys()):
            raise NameError("One or more of the parameter paths from {} is not unique: "
                            "{}".format(self.name, self.child_base_parameter_dct.keys()))

        if len(self.base_parameter_dct) != len(self.base_parameters):
            raise ValueError("There are two parameters with the same name in {}: "
                             "{}".format(self.__class__.__name__, self.base_parameter_dct.keys()))

    def _get_subcomponent_names(self):
        return [self.name] + sum(
            [cmp._get_subcomponent_names() for cmp in self._subcomponents], [])

    @_name.default
    def _name_default(self):
        return self.__class__.__name__

    @cached_property
    def name(self):
        """Name of the component"""
        return self._name or self.__class__.__name__

    @params.validator
    def _params_validator(self, attribute, value):
        assert all([isinstance(p, Param) for p in value])
        # Ensure unique names, and unique determines
        names = [p.name for p in value]
        if len(set(names)) != len(names):
            raise NameError("not all param names are unique in {}: "
                            "{}".format(self.name, names))

        determines = sum([list(p.determines) for p in value], [])
        if len(set(determines)) != len(determines):
            raise ValueError("different parameters determine the same base parameter "
                             "in {}:{}".format(self.name, determines))

        if any([d not in self.base_parameter_dct for d in determines]):
            raise ValueError(
                "One or more params do not map to any known Parameter in {}: {}. "
                "Known params: {}".format(self.name, determines,
                                          list(self.base_parameter_dct.keys())))

    @fiducial.validator
    def _fiducial_validator(self, att, value):
        for name, value in value.items():
            if name in self.params:
                raise ValueError("Pass fiducial values to constrained parameters "
                                 "inside the Param class")
            if name not in self.base_parameter_dct:
                raise KeyError(
                    "Fiducial parameter {} does not match any parameters of {}. "
                    "Note that fiducial parameters must be passed at the level "
                    "to which they belong.".format(name, self.name)
                )

    @derived.validator
    def _derived_validator(self, att, val):
        self._validate_derived(val)

    def _validate_derived(self, val):
        for d in val:
            assert (callable(d) or (type(d) is str and hasattr(self, d))), f"{d} is not a valid derived parameter"

    @cached_property
    def base_parameter_dct(self):
        """
        All possible parameters of this specific component or likelihood,
        not just those that are being constrained.
        """
        return {p.name: p for p in self.base_parameters}

    @cached_property
    def active_params(self):
        """
        Tuple of actively constrained parameters specified for this
        component or likelihood only.

        Note that this is just the parameters themselves.
        """

        out = []
        for v in self.params:
            if v.is_pure_alias:
                out.append(v.new(self.base_parameter_dct[v.determines[0]]))
            else:
                out.append(v)

        return tuple(out)

    @cached_property
    def active_params_dct(self):
        """OrderedDict of actively constrained parameter names in this component"""
        return OrderedDict([(p.name, p) for p in self.active_params])

    @cached_property
    def _base_to_param_mapping(self):
        """dict of base:active params in *this* component"""
        mapping = OrderedDict()
        for p in self.base_parameters:
            for prm in self.active_params:
                if p.name in prm.determines:
                    mapping[p] = prm
        return mapping


@attr.s(frozen=True, kw_only=True)
class Component(ParameterComponent):
    """
    A component of a likelihood. These are mainly for re-usability, so they
    can be mixed and matched inside likelihoods.
    """
    components = attr.ib(factory=tuple, converter=tuple, kw_only=True)

    provides = tuple()
    _plugins = {}

    def __init_subclass__(cls, is_abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)

        # Plugin framework
        if not is_abstract:
            cls._plugins[cls.__name__] = cls

        if type(cls.provides) not in [list, set, tuple, cached_property]:
            raise TypeError("Component {} must define a list/set/tuple/cached_property for 'provides'. Instead, it is {}".format(cls.__name__, type(cls.provides)))

        if type(cls.provides) in [list, set, tuple]:
            for pr in cls.provides:
                if type(pr) is not str:
                    raise ValueError("provides should be an ordered iterable of strings")

    @components.validator
    def _cmp_valid(self, att, val):
        for cmp in val:
            assert isinstance(cmp, Component), "component {} is not a valid Component".format(cmp.name)

    def derived_quantities(self, ctx=None, params=None):
        if ctx is None:
            ctx = self()

        dquants = []
        for d in self.derived:
            if type(d) == str:
                dquants.append(getattr(self, d)(ctx, **params))
            elif callable(d):
                dquants.append(d(ctx, **params))
            else:
                raise ValueError("{} is not a valid entry for derived".format(d))

        return dquants

    @cached_property
    def _subcomponents(self):
        return self.components

    def calculate(self, ctx=None, **params):
        """

        Parameters
        ----------
        params

        Returns
        -------

        """
        pass

    def __call__(self, params=None, ctx=None, ignore_components=None):
        """
        Every component should take a dct of parameter values and return
        a dict of values to be used. All Parameters of the component will
        be in params.
        """
        params = self._fill_params(params)

        if ctx is None:
            ctx = {}

        # First get ignored components
        if ignore_components is None:
            ignore_components = [cmp.name for cmp, _ in self.common_components]

            for cmp, locs in self.common_components:
                ctx.update(cmp(ctx=ctx, params=utils.get_loc_from_dict(params, locs[0])))

        for cmp in self._subcomponents:
            if cmp.name in ignore_components:
                if not all([name in ctx for name in cmp.provides]):
                    raise ValueError(
                        "attempting to ignore component '{}' in '{}' without providing "
                        "appropriate context".format(cmp.name, self.name))
                continue
            else:
                ctx.update(cmp(params=params[cmp.name], ctx=ctx, ignore_components=ignore_components))

        res = self.calculate(ctx, **params)

        if res is None:
            if self.provides:
                raise ValueError(
                    "component {} says it provides {} but does not return anything from calculate()".format(self.name,
                                                                                                            self.provides))
            return ctx
        else:
            if type(res) != tuple:
                res = (res,)

            if len(self.provides) != len(res):
                raise ValueError(
                    "{} does not return an ordered iterable of the same length "
                    "as its 'provides' attribute from calculate()".format(
                        self.__class__.__name__)
                )

            ctx.update({p: r for p, r in zip(self.provides, res)})
            return ctx


class LikelihoodInterface(ABC):
    """
    An abstract base class for likelihoods, defining the methods they must expose.
    """

    def mock(self, model=None, ctx=None, params=None):
        """Create a mock dataset given a set of parameters (and potentially a
        corresponding model and ctx)
        """
        pass

    def derived_quantities(self, model=None, ctx=None, params=None):
        """Generate specified derived quantities given a set of parameters (and potentially
        a corresponding model and ctx
        """
        pass

    def logprior(self, params=None):
        """Generate the total logprior for all active parameters."""
        pass

    def get_ctx(self, params=None):
        """Generate the context, by running all components."""
        pass

    def logl(self, model=None, ctx=None, params=None):
        """Return the log-likelihood at the given parameters"""
        pass

    def logp(self, model=None, params=None):
        """Return the log-posterior at the given parameters"""
        return self.logprior(**params) + self.logl(model, **params)

    def __call__(self, params=None, ctx=None):
        """Return a tuple of the log-posterior and derived quantities at given params"""
        pass


@attr.s(frozen=True, kw_only=True)
class Likelihood(ParameterComponent, LikelihoodInterface):
    _data = attr.ib(default=None)
    components = attr.ib(factory=tuple, converter=tuple)
    _data_seed = attr.ib(validator=validators.optional(validators.instance_of(int)))
    _store_data = attr.ib(False, converter=bool)

    _plugins = {}

    def __init_subclass__(cls, is_abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not is_abstract:
            cls._plugins[cls.__name__] = cls

    @components.validator
    def _cmp_valid(self, att, val):
        for cmp in val:
            assert isinstance(cmp, Component), "component {} is not a valid Component".format(cmp.name)

    @cached_property
    def using_mock_data(self):
        return self._data is None and hasattr(self, "_mock")

    @_data_seed.default
    def _data_seed_default(self):
        if self.using_mock_data:
            if mpi.more_than_one_process:
                raise TypeError("if using MPI and auto-generated mock data, data_seed must be set")
            return np.random.randint(0, 2 ** 32 - 1)
        else:
            return None

    @cached_property
    def _subcomponents(self):
        return self.components

    @cached_property
    def data(self):
        if self._data is None and not hasattr(self, "_mock"):
            raise AttributeError("You have not passed any data and no mock method found")

        if self.using_mock_data:
            # We only want to set the seed for data creation, and then
            # randomize afterwards.
            with utils.seed_as(self._data_seed):
                data = self.mock()
            return data

        return self._data

    @classmethod
    def check_component_requirements(cls, components):
        return True

    def get_ctx(self, ctx=None, ignore_components=None, params=None):
        params = self._fill_params(params)

        if ctx is None:
            ctx = {}
        else:
            assert isinstance(ctx, collections.abc.Mapping)
            ctx = deepcopy(ctx)

        if ignore_components is None:
            ignore_components = [cmp.name for cmp,_ in self.common_components]

            for cmp, locs in self.common_components:
                ctx.update(cmp(ctx=ctx, params=utils.get_loc_from_dict(params, locs[0])))

        for cmp in self._subcomponents:
            if cmp.name in ignore_components:
                continue

            res = cmp(params=params[cmp.name])
            ctx.update(res)

        return ctx

    def _get_model_ctx_params(self, params, model=None, ctx=None, ignore_components=None):
        params = self._fill_params(params)

        ctx = self.get_ctx(ctx=ctx, ignore_components=ignore_components, params=params)

        if model is None:
            model = self.reduce_model(ctx=ctx, params=params)

        return model, ctx, params

    def mock(self, model=None, ctx=None, ignore_components=None, params=None):
        """
        Create mock data at given parameters.
        """
        model, _, params = self._get_model_ctx_params(params, model, ctx, ignore_components)
        return self._mock(model, **params)

    def derived_quantities(self, model=None, ctx=None, ignore_components=None, params=None):
        model, ctx, params = self._get_model_ctx_params(params, model, ctx, ignore_components)

        dquants = []
        for d in self.derived:
            if type(d) == str:
                # Append local quantity
                dquants.append(getattr(self, d)(model, ctx, **params))
            elif callable(d):
                dquants.append(d(model, ctx, **params))
            else:
                raise ValueError("{} is not a valid entry for derived".format(d))

        for cmp in self._subcomponents:
            dquants += cmp.derived_quantities(ctx, params[cmp.name])

        return dquants

    def logprior(self, params=None):
        # This can be called in non-active mode, it will just return zero.
        params = self._fill_params(params)

        prior = 0

        for apar in self.child_active_params:
            value = utils.get_loc_from_dict(params, apar.determines[0])
            prior += apar.logprior(value)

        return prior

    def reduce_model(self, ctx=None, ignore_components=None, params=None):
        """
        Reduce the model data produced by the components, returning
        the most reduced model data required to calculate the likelihood.
        """
        params = self._fill_params(params)
        ctx = self.get_ctx(ctx=ctx, ignore_components=ignore_components, params=params)

        top_level_params = {p: v for p, v in params.items() if not isinstance(v, dict)}
        return self._reduce(ctx, **top_level_params)

    def _reduce(self, ctx, **params):
        """Basic reduction just returns the ctx as a whole"""
        return ctx

    def logl(self, model=None, ctx=None, ignore_components=None, params=None):
        model, _, params = self._get_model_ctx_params(params, model, ctx, ignore_components)
        top_level_params = {p: v for p, v in params.items() if not isinstance(v, collections.abc.Mapping)}
        return self.lnl(model, **top_level_params)

    def logp(self, model=None, ctx=None, ignore_components=None, params=None):
        params=self._fill_params(params)
        return self.logprior(params) + self.logl(
            model=model, ctx=ctx, ignore_components=ignore_components, params=params
        )

    def __call__(self, params=None, ctx=None, ignore_components=None):
        if params is None:
            params = {}
        params = self._fill_params(params)

        ctx = self.get_ctx(ctx, ignore_components, params)

        ignore = [cmp.name for cmp in self.components]  # ignore everything since ctx is full

        model = self.reduce_model(ctx, ignore_components=ignore, params=params)

        return (self.logp(model, ignore_components=ignore, params=params),
                self.derived_quantities(model, ctx, ignore_components=ignore, params=params))

    def __getstate__(self):
        dct = super().__getstate__()
        if self._store_data:
            dct.update(_data=self.data)
        return dct


@attr.s
class LikelihoodContainer(LikelihoodInterface, _ComponentTree):
    likelihoods = attr.ib()
    _name = attr.ib(None, kw_only=True, validator=validators.optional(validators.instance_of(str)))

    @likelihoods.validator
    def _lk_valid(self, att, val):
        if not hasattr(val, "__len__") or len(val) < 1:
            raise ValueError("likelihoods should be a tuple of at least one likelihood")

        for lk in val:
            assert isinstance(lk, Likelihood)

    @cached_property
    def _subcomponents(self):
        return self.likelihoods

    @cached_property
    def name(self):
        return self._name or "_".join([lk.name for lk in self.likelihoods])

    def get_ctx(self, params=None):
        """
        Generate the context by running all components in all likelihoods
        """
        params = self._fill_params(params)

        ctx = {}

        # Do the common components
        for cmp, locs in self.common_components:
            this = cmp(params=utils.get_loc_from_dict(params, locs[0]))

            for loc in locs:
                ctx[loc.split(".")[0]] = deepcopy(this)

        # Do the rest
        for lk in self.likelihoods:
            if lk.name not in ctx:
                ctx[lk.name] = {}

            ctx[lk.name].update(
                lk.get_ctx(
                    ctx=ctx[lk.name],
                    ignore_components=[cmp.name for cmp, _ in self.common_components],
                    params=params[lk.name]
                )
            )
        return ctx

    def reduce_model(self, ctx=None, params=None):
        """Get models from all likelihoods, as a dictionary of likelihood_name: reduce"""
        params = self._fill_params(params)

        if ctx is None:
            ctx = self.get_ctx(params)

        out = {}
        for lk in self.likelihoods:
            out[lk.name] = lk.reduce_model(
                ctx[lk.name],
                ignore_components=[cmp.name for cmp in lk.components],
                params=params[lk.name]
            )

        return out

    def get_logls(self, model=None, ctx=None, params=None):
        params = self._fill_params(params)

        if model is None:
            model = self.reduce_model(ctx, params)

        out = {}
        for lk in self.likelihoods:
            out[lk.name] = lk.lnl(model[lk.name], **params[lk.name])

        return out

    def logl(self, model=None, ctx=None, params=None):
        params = self._fill_params(params)
        logls = self.get_logls(model, ctx, params)
        return sum(logls.values())

    def logprior(self, params=None):
        params = self._fill_params(params)
        prior = 0
        for lk in self.likelihoods:
            prior += lk.logprior(params[lk.name])
        return prior

    def logp(self, model=None, ctx=None, params=None):
        params = self._fill_params(params)
        logl = self.logl(model, ctx, params)  # this fills the params
        return logl + self.logprior(params)

    @cached_property
    def derived(self):
        return sum([lk.derived for lk in self.likelihoods], tuple())

    def derived_quantities(self, model=None, ctx=None, params=None):
        params = self._fill_params(params)

        if ctx is None:
            ctx = self.get_ctx(params)

        if model is None:
            model = self.reduce_model(ctx, params)

        dquants = []
        for lk in self.likelihoods:
            dquants += lk.derived_quantities(model[lk.name], ctx[lk.name], params[lk.name])

        return dquants

    def __call__(self, params=None):
        params = self._fill_params(params)
        ctx = self.get_ctx(params)
        model = self.reduce_model(ctx=ctx, params=params)
        return self.logp(model=model, params=params), self.derived_quantities(model, ctx, params)
