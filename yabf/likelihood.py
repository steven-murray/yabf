"""
Framework for MCMC likelihoods (and parameters).
"""
import warnings
from collections import OrderedDict
from copy import copy

import attr
from cached_property import cached_property

from .plugin import plugin_mount_factory


def _only_active(func):
    def func_wrapper(self, *args, **kwargs):
        if not self.in_active_mode:
            raise AttributeError("{} is not available when not in active mode.".format(func.__name__))
        return func(self, *args, **kwargs)

    return func_wrapper


class DependencyError(ValueError):
    pass


@attr.s
class ParameterComponent:
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
    derived: tuple of strings or callables
        A tuple defining which derived parameters may be obtained by calling the
        :func:`derived_quantities` method. If str, there must be a method within
        the object by that name, else if a callable, it must receive a number of
        arguments which depends on the kind of class this is. It will typically
        be ctx and kwargs for params, or perhaps a model, ctx and params.
    """
    _name = attr.ib(validator=attr.validators.optional(attr.validators.instance_of(str)))
    fiducial = attr.ib(factory=dict, kw_only=True)
    params = attr.ib(factory=tuple, converter=tuple, kw_only=True)
    derived = attr.ib(factory=tuple, converter=tuple, kw_only=True)

    base_parameters = tuple()

    def __init_subclass__(cls, **kwargs):
        names = [p.name for p in cls.base_parameters]
        if len(set(names)) != len(cls.base_parameters):
            raise ValueError("There are two parameters with the same name in {}: {}".format(cls.__name__, names))
        super().__init_subclass__()

    def __attrs_post_init__(self):
        if len(set(self._child_parameter_locs)) != len(self._child_parameter_locs):
            raise NameError("One or more of the parameter paths from {} is not unique: "
                            "{}".format(self.name, self._child_parameter_locs))

    def _get_subcomponent_names(self):
        return [self.name] + sum([cmp._get_subcomponent_names() for cmp in self._subcomponents], [])

    @_name.default
    def _name_default(self):
        return self.__class__.__name__

    @cached_property
    def name(self):
        """Name of the component"""
        return self._name or self.__class__.__name__

    @fiducial.validator
    def _fiducial_validator(self, attribute, value):
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

    @cached_property
    def _in_active_mode(self):
        return bool(self.child_active_params)

    @cached_property
    def _subcomponents(self):
        return tuple()

    @cached_property
    def _subcomponent_names(self):
        return tuple([c.name for c in self._subcomponents])

    def _validate_derived(self, val):
        for d in val:
            assert (callable(d) or (type(d) is str and hasattr(self, d))), f"{d} is not a valid derived parameter"

    @property
    def in_active_mode(self):
        """
        Bool representing whether this class is in active mode (i.e. whether
        it is being actively constrained).
        """
        return self._in_active_mode or any([cmp.in_active_mode for cmp in self._subcomponents])

    @cached_property
    def base_parameter_dct(self):
        """
        All possible parameters of this specific component or likelihood,
        not just those that are being constrained.
        """
        return {p.name: p for p in self.base_parameters}

    @cached_property
    def child_base_parameters(self):
        """
        Tuple of all parameters in this and child components.
        """
        this = tuple(self.base_parameters)
        for cmp in self._subcomponents:
            this = this + cmp.child_base_parameters
        return this

    @cached_property
    def child_base_parameter_names(self):
        """Set of all available parameter names in this component and its sub-components"""
        return tuple([p.name for p in self.child_base_parameters])

    @staticmethod
    def _loc_to_dict_loc(loc, dict):
        if loc == "":
            return dict
        else:
            locs = loc.split(".")
            d = copy(dict)
            for l in locs:
                try:
                    d = d[l]
                except KeyError:
                    raise KeyError("loc {} does not exist in dict {}".format(loc, dict))

            return d

    def _loc_to_component(self, loc):
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

    @cached_property
    def _child_parameter_locs(self):
        """String locs to each child parameter"""
        res = []
        for p in self.base_parameter_dct:
            res.append(p)

        for cmp in self._subcomponents:
            this = [cmp.name + "." + child for child in cmp._child_parameter_locs]

            res.extend(this)

        return tuple(res)

    @cached_property
    def active_params(self):
        """
        Tuple of actively constrained parameters specified for this
        component or likelihood only.

        Note that this is just the parameters themselves.
        """
        out = []
        for v in self.params:
            if type(v.alias_for) is str:
                alias = [v.alias_for]
            else:
                alias = v.alias_for

            # If user has specified a path to a parameter, get the component
            # corresponding to that path.
            full_aliases = []

            for al in alias:
                loc, _, paramname = al.rpartition(".")

                # Now get a full alias, by matching the alias with the list of
                # child parameter locs
                matched = 0
                for lc in self._child_parameter_locs:
                    if lc.startswith(loc) and lc.endswith(paramname):
                        full_aliases.append(lc)
                        matched += 1

                if matched > 1:
                    warnings.warn(
                        "You have passed a parameter alias [{}] which matches "
                        "more than one parameter [{}]. Only information from the first "
                        "matched parameter will be used to define the active "
                        "parameter".format(al, full_aliases[-matched:])
                    )

            if not full_aliases:
                raise ValueError("No match found for active parameter {}".format(v))

            # We can only base the active parameter on
            # one actual parameter, and so we obviously do the first.
            param = self.child_base_parameters[self._child_parameter_locs.index(full_aliases[0])]
            out.append(v.new(param, aliases=full_aliases))

        out = tuple(out)
        if len(set([o.name for o in out])) != len(out):
            raise ValueError("You cannot specify the same name for two parameters!")

        return out

    @cached_property
    def active_params_dct(self):
        """OrderedDict of actively constrained parameter names in this component"""
        return OrderedDict([(p.name, p) for p in self.active_params])

    @cached_property
    def child_active_params(self):
        """tuple of all active parameters in this likelihood/component and its sub-components"""
        p = self.active_params
        for cmp in self._subcomponents:
            p = p + cmp.child_active_params

        # Ensure that no two active parameters below this level have the same name.
        names = [o.name for o in p]
        if len(set(names)) != len(p):
            raise ValueError("Two or more active parameters under {} have the same name."
                             "Consider using `alias_for`. {}".format(self.name, names))

        return p

    def _active_param_prefix(self, param):
        """
        Get a locotion prefix for an active param (i.e. dot-path of sub-components
        down to the location of the param).
        """
        if param in self.active_params:
            return ''
        else:
            for cmp in self._subcomponents:
                try:
                    return cmp.name + "." + cmp._active_param_prefix(param)
                except AttributeError:
                    pass

        raise AttributeError("param '{}' is not available under '{}'".format(param.name, self.name))

    def _get_subloc(self, name):
        if name in self.base_parameter_dct:
            return self.name

        for cmp in self._subcomponents:
            loc = cmp._get_subloc(name)
            if loc is not None:
                return self.name + '.' + loc

        return None

    @cached_property
    def total_active_params(self):
        return len(self.child_active_params)

    def _active_param_locgen(self, param):
        prefix = self._active_param_prefix(param)
        for alias in param.alias_for:
            loc, _, paramname = alias.rpartition(".")
            yield (prefix + loc).rstrip('.'), paramname

    @cached_property
    def fiducial_params(self):
        """
        Dictionary of fiducial parameters for all subcomponents
        """
        dct = {}

        for param in self.base_parameters:
            if param.name in self.fiducial:
                dct[param.name] = self.fiducial[param.name]
            else:
                dct[param.name] = param.fiducial

        for cmp in self._subcomponents:
            dct[cmp.name] = cmp.fiducial_params

        # Now overwrite everything with active params.
        for apar in self.active_params:
            for loc, paramname in self._active_param_locgen(apar):
                this = self._loc_to_dict_loc(loc, dct)
                this[paramname] = apar.fiducial

        return dct

    @staticmethod
    def _recursive_dict_update(ref, new):
        for key, val in new.items():
            if isinstance(val, dict):
                ParameterComponent._recursive_dict_update(ref[key], val)
            else:
                ref[key] = val
        return ref

    def _fill_params(self, params=None):
        params = params or {}
        if self._is_params_full(params):
            return params

        fiducial = copy(self.fiducial_params)

        # Update all levels of the params
        if self._in_active_mode:
            ParameterComponent._recursive_dict_update(fiducial, params)
        else:
            for key, val in params.items():
                if key in fiducial:
                    if isinstance(val, dict):
                        ParameterComponent._recursive_dict_update(fiducial[key], val)
                    else:
                        fiducial[key] = val
                elif '.' in key:
                    loc, _, name = key.rpartition('.')
                    this = self._loc_to_dict_loc(loc, fiducial)
                    if name not in this:
                        raise ValueError("{} was not found in params".format(key))
                    this[name] = val
                else:
                    # TODO: should also be able to find it if it is unique
                    raise ValueError("{} was not found in params".format(key))

        return fiducial

    def check_param_active_compat(self, params):
        """
        Check whether the given dictionary of parameters is compatible with active_params

        Checks whether each parameter referenced by active parameters is in the dict,
        and also whether those that are referenced by the same parameter are equal.
        """
        for apar in self.child_active_params:
            value = None

            for loc, paramname in self._active_param_locgen(apar):
                try:
                    this = self._loc_to_dict_loc(loc, params)
                except KeyError:
                    return False

                if paramname not in this:
                    return False
                else:
                    if value is None:
                        value = this[paramname]
                    elif value != this[paramname]:
                        return False
        return True

    def _is_params_full(self, params):
        if len(params) != len(self._subcomponents) + len(self.base_parameters):
            return False

        for cmp in self._subcomponents:
            if cmp.name not in params:
                return False
            if not cmp._is_params_full(params[cmp.name]):
                return False

        return True

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
        heirarchy.

        This returns a dict, where each parameter is inserted at the various
        locations at which it is defined (via its aliases).
        """
        if len(p) != self.total_active_params:
            raise ValueError("length of parameter list not compatible with "
                             "active parameters. Please don't call "
                             "_parameter_list_to_dict yourself!")

        dct = {}
        for i, (pvalue, apar) in enumerate(zip(p, self.child_active_params)):
            for loc, paramname in self._active_param_locgen(apar):

                this = dct
                for l in loc.split("."):
                    if l == "":
                        break

                    if l not in this:
                        this[l] = {}
                    this = this[l]

                this[paramname] = pvalue

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
            names = [a.name for a in self.child_active_params]
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
        return {k:v for k,v in self.__dict__.items() if k in attr.asdict(self)}


class Component(ParameterComponent, metaclass=plugin_mount_factory()):
    """
    A component of a likelihood. These are mainly for re-usability, so they
    can be mixed and matched inside likelihoods.
    """
    provides = tuple()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if type(cls.provides) not in [list, set, tuple]:
            raise TypeError("Component {} must define a list/set/tuple for provides".format(cls.__name__))

        for pr in cls.provides:
            if type(pr) is not str:
                raise ValueError("provides should be an ordered iterable of strings")

    def derived_quantities(self, ctx=None, **params):
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

    def calculate(self, **params):
        """

        Parameters
        ----------
        params

        Returns
        -------

        """
        pass

    def __call__(self, ctx=None, **params):
        """
        Every component should take a dct of parameter values and return
        a dict of values to be used. All Parameters of the component will
        be in params.
        """
        if len(params) != len(self.base_parameters):
            params = self._fill_params(params)

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

            if ctx is None:
                ctx = {}

            ctx.update({p: r for p, r in zip(self.provides, res)})
            return ctx


@attr.s
class Likelihood(ParameterComponent, metaclass=plugin_mount_factory()):
    _data = attr.ib(default=None, kw_only=True)
    components = attr.ib(factory=tuple, converter=tuple, kw_only=True)

    @components.validator
    def _cmp_valid(self, att, val):
        for cmp in val:
            assert isinstance(cmp, Component), "component {} is not a valid Component".format(cmp.name)

    @cached_property
    def _subcomponents(self):
        return self.components

    @cached_property
    def data(self):
        if self._data is None and not hasattr(self, "_mock"):
            raise AttributeError("You have not passed any data and no mock method found")

        if self._data is None and hasattr(self, "_mock"):
            self._data = self.mock()
        return self._data

    @classmethod
    def check_component_requirements(cls, components):
        return True

    def mock(self, model=None, ctx=None, **params):
        """
        Create mock data at given parameters.
        """
        model, params = self._get_model_and_params(params, model, ctx)
        return self._mock(model, **params)

    def derived_quantities(self, model=None, ctx=None, **params):
        params = self._fill_params(params)

        if ctx is None:
            ctx = self.get_ctx(**params)

        if model is None:
            model = self._model(ctx, **params)

        dquants = []
        for d in self.derived:
            if type(d) == str:
                # Append local quantity
                dquants.append(getattr(self, d)(model, ctx, **params))
            elif callable(d):
                dquants.append(d(model, ctx, **params))
            else:
                raise ValueError("{} is not a valid entry for derived".format(d))

        for cmp in self.components:
            dquants += cmp.derived_quantities(ctx, **params[cmp.name])

        return dquants

    def logprior(self, **params):
        # This can be called in non-active mode, it will just return zero.
        params = self._fill_params(params)

        if not self.check_param_active_compat(params):
            raise ValueError("params is inconsistent with active parameters. "
                             "logprior requires consistency.")
        prior = 0

        for apar in self.child_active_params:
            loc, paramname = list(self._active_param_locgen(apar))[0]
            this = self._loc_to_dict_loc(loc, params)
            prior += apar.logprior(this[paramname])

        return prior

    def get_ctx(self, **params):
        params = self._fill_params(params)

        ctx = {}

        for cmp in self._subcomponents:
            res = cmp(**params[cmp.name])
            ctx.update(res)

        return ctx

    def reduce_model(self, ctx=None, **params):
        """
        Reduce the model data produced by the components, returning
        the most reduced model data required to calculate the likelihood.
        """
        params = self._fill_params(params)

        if ctx is None:
            ctx = self.get_ctx(**params)

        top_level_params = {p: v for p, v in params.items() if not isinstance(v, dict)}
        return self._reduce(ctx, **top_level_params)

    def _reduce(self, ctx, **params):
        """Basic reduction just returns the ctx as a whole"""
        return ctx

    def _get_model_and_params(self, params, model=None, ctx=None):
        params = self._fill_params(params)

        if ctx is None:
            ctx = self.get_ctx(**params)

        if model is None:
            model = self.reduce_model(ctx, **params)

        return model, params

    def logl(self, model=None, ctx=None, **params):
        model, params = self._get_model_and_params(params, model, ctx)
        top_level_params = {p: v for p, v in params.items() if not isinstance(v, dict)}
        return self.lnl(model, **top_level_params)

    def logp(self, model=None, **params):
        return self.logprior(**params) + self.logl(model, **params)

    def __call__(self, params=None):
        if params is None:
            params = {}

        if type(params) is not dict:
            params = self._parameter_list_to_dict(params)

        params = self._fill_params(params)
        ctx = self.get_ctx(**params)
        model = self.reduce_model(ctx, **params)

        return self.logp(model, **params), self.derived_quantities(model, ctx, **params)


@attr.s
class LikelihoodContainer(Likelihood):
    likelihoods = attr.ib(converter=tuple, kw_only=True)

    @likelihoods.validator
    def _lk_valid(self, att, val):
        if not hasattr(val, "__len__") or len(val) < 1:
            raise ValueError("likelihoods should be a tuple of at least one likelihood")

        for lk in val:
            assert isinstance(lk, Likelihood)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.name == self.__class__.__name__:
            self.name = " ".join([lk.name for lk in self.likelihoods])

    @cached_property
    def _subcomponents(self):
        return self.components + self.likelihoods

    def _validate_derived(self, val):
        for d in val:
            assert callable(d), f"{d} is not a valid derived parameter"

    def get_ctx(self, **params):
        """
        Return the ctx for components *in the top-level*.

        Components in sub-likelihoods have their context evaluated only
        by the likelihood and only available therein.
        """
        params = self._fill_params(params)

        ctx = {}
        for cmp in self.components:
            ctx.update(cmp(**params[cmp.name]))

        return ctx

    def _reduce_all(self, ctx=None, **params):
        """Get models from all likelihoods, as a dictionary of likelihood_name: reduce"""
        params = self._fill_params(params)

        if ctx is None:
            ctx = self.get_ctx(**params)

        out = {}
        for lk in self.likelihoods:
            this_ctx = lk.get_ctx(**params[lk.name])
            this_ctx.update(ctx)

            out[lk.name] = lk.reduce_model(this_ctx, **params[lk.name])

        return out

    def get_logls(self, models=None, **params):
        params = self._fill_params(params)

        if models is None:
            models = self._reduce_all(**params)

        out = {}
        for lk in self.likelihoods:
            top_level_params = {p: v for p, v in params[lk.name].items() if not isinstance(v, dict)}
            out[lk.name] = lk.lnl(models[lk.name], **top_level_params)

        return out

    def logl(self, models=None, **params):
        logsls = self.get_logls(models, **params)
        return sum(logsls.values())

    def logp(self, models=None, **params):
        logl = self.logl(models, **params)  # this fills the params
        return logl + self.logprior(**params)

    def derived_quantities(self, models=None, ctx=None, **params):
        params = self._fill_params(params)

        if ctx is None:
            ctx = self.get_ctx(**params)

        if models is None:
            models = self._reduce_all(ctx, **params)

        dquants = []
        for d in self.derived:
            dquants.append(d(models, ctx, **params))

        for cmp in self.components:
            dquants += cmp.derived_quantities(ctx, **params[cmp.name])

        for lk in self.likelihoods:
            dquants += lk.derived_quantities(models[lk.name], ctx, **params[lk.name])

        return dquants

    def __call__(self, params=None):
        if params is None:
            params = {}

        if type(params) is not dict:
            params = self._parameter_list_to_dict(params)

        ctx = self.get_ctx(**params)
        models = self._reduce_all(ctx, **params)
        return self.logp(models, **params), self.derived_quantities(models, ctx, **params)
