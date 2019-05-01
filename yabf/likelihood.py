"""
Framework for MCMC likelihoods (and parameters).
"""
import warnings
from collections import OrderedDict
from copy import copy

import numpy as np
from cached_property import cached_property
from .plugin import plugin_mount_factory
from .parameters import Param

def _only_active(func):
    def func_wrapper(self, *args, **kwargs):
        if not self.in_active_mode:
            raise AttributeError("{} is not available when not in active mode.".format(func.__name__))
        return func(self, *args, **kwargs)

    return func_wrapper


class DependencyError(ValueError):
    pass


class ParameterComponent:
    """
    A base class for named components and likelihoods that take parameters.
    """
    base_parameters = tuple()

    def __init__(self, name=None, params=None, fiducial=None, derived=None):
        self.name = name or self.__class__.__name__

        params = tuple(params) if params is not None else tuple()
        self.derived = tuple(derived) if derived is not None else tuple()
        fiducial = fiducial or {}

        self.__constrained_params = OrderedDict([(p.name, p) for p in params])

        for name, value in fiducial.items():
            if name in self.__constrained_params:
                raise ValueError("Pass fiducial values to constrained parameters inside the Param class")
            if name not in self.base_parameter_dct:
                raise KeyError("parameter {} not found in set of all parameters".format(name))

        self._fixed_fiducial = fiducial
        self.validate_derived()

        self._in_active_mode = len(self.active_params) > 0

    @cached_property
    def _subcomponents(self):
        return tuple()

    @cached_property
    def _subcomponent_names(self):
        return tuple([c.name for c in self._subcomponents])

    def validate_derived(self):
        for d in self.derived:
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
        The *set* of all parameters in this and child components.

        Parameters with the same name in different sub-components can be
        represented twice if their other attributes are different.
        """
        this = set(self.base_parameters)
        for cmp in self._subcomponents:
            this.update(cmp.child_base_parameters)
        return this

    @cached_property
    def child_base_parameter_names(self):
        """Set of all available parameter names in this component and its sub-components"""
        p = set(self.base_parameter_dct.keys())
        for cmp in self._subcomponents:
            p.update(cmp.child_base_parameter_names)
        return p

    @cached_property
    def active_params(self):
        """
        Tuple of actively constrained parameters.

        Note that this is just the parameters themselves.
        """
        out = []
        for p, v in self.__constrained_params.items():
            # First check if there's two child params by this name
            if len([param.name for param in self.child_base_parameters if param.name == p]) > 1:
                raise ValueError(
                    "yabf does not currently support setting active parameters "
                    "for child components with different parameters of the same "
                    "name"
                )

            for param in self.child_base_parameters:
                if param.name == p:
                    out.append(v.new(param))
                    break
        out = tuple(out)

        return out

    @cached_property
    def active_params_dct(self):
        """OrderedDict of actively constrained parameter names"""
        return OrderedDict([( p.name, p) for p in self.active_params])

    @cached_property
    def child_active_params(self):
        """Set of all active parameters in this likelihood/component and its sub-components"""
        p = set(self.active_params)
        for cmp in self._subcomponents:
            p.update(cmp.child_active_params)
        return p

    def _get_subloc(self, name):
        if name in self.base_parameter_dct:
            return self.name

        for cmp in self._subcomponents:
            loc = cmp._get_subloc(name)
            if loc is not None:
                return self.name + '.' + loc

        return None

    @cached_property
    def flat_active_params(self):
        # TODO: possibly bad because names will overlap, which is why child_active_params is a set.
        result = OrderedDict()

        for p, v in self.active_params_dct.items():
            loc = self._get_subloc(p)
            if loc is None:
                raise TypeError("The parameter {} does not seem to exist in {} or any of its subcomponents".format(p, self.name))

            result[p] = dict(parent=self.name, param=v, loc=loc)

        for cmp in self._subcomponents:
            these = copy(cmp.flat_active_params)

            # Prepend locations with this class.
            for t in these.values():
                t['parent'] = self.name + "." + t['parent']
                t['loc'] = self.name + "." + t['loc']

            result.update(these)

        return result

    @cached_property
    def total_active_params(self):
        return len(self.child_active_params)

    @cached_property
    def fiducial_params(self):
        """
        Dictionary of fiducial parameters for this component (and no child components)
        """
        dct = {}
        for param in self.base_parameters:
            if param.name in self.active_params_dct:
                dct[param.name] = self.active_params_dct[param.name].fiducial
            elif param.name in self._fixed_fiducial:
                dct[param.name] = self._fixed_fiducial[param.name]
            else:
                dct[param.name] = param.fiducial
        return dct

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
            include all (active) parameters.

        Returns
        -------
        list, list-of-lists, dict or dict-of-lists :
            If `full` is False, returns a list. If `n=1` and `squeeze` is True,
            this is a list of parameter values (by default, active parameters).
            Otherwise, it is a list of lists, where each sub-list contains n
            values of the parameter. If `full` is True, returns a dict, where
            each key is the name of a parameter.
        """
        if params is not None:
            p = []
            for param in params:
                if isinstance(param, Param):
                    if param not in self.child_active_params:
                        raise ValueError("You have provided an active parameter which is not currently active")
                    else:
                        p.append(param.name)
                else:
                    p.append(param)
            params = p
        else:
            params = self.flat_active_params.keys()

        refs = []

        for param in params:
            if param in self.active_params_dct:
                ref = self.active_params_dct[param].generate_ref(n)
                if not squeeze and n == 1:
                    ref = [ref]

            elif param in self.fiducial_params:
                ref = [self.fiducial_params[param]] * n
                if squeeze and n == 1:
                    ref = ref[0]
            else:
                # search for the parameter in sub-components

                # First check if there's two by that name
                if len([p for p in self.child_base_parameters if p.name == param]) > 1:
                    raise ValueError(
                        "yabf does not currently support generating refs by string "
                        "for child components with different parameters of the same "
                        "name"
                    )

                # Otherwise, get the first such parameter and use it.
                for cmp in self._subcomponents:
                    ref = cmp.generate_refs(n=n, squeeze=squeeze, params=[param])
                    if ref:
                        ref = ref[0]
                        break

            refs.append(ref)

        return refs

    def _fill_params(self, params=None):
        # Note: this function does _not_ depend on the active parameters of
        #       either itself or its sub-components. As long as the params
        #       passed have the correct structure, it will fill them.

        params = params or {}
        if self._is_params_full(params):
            return params

        fiducial = copy(self.fiducial_params)
        fiducial.update(params)

        # we're gonna add to fiducial here from the active params
        # they are not necessarily in fiducial already, if they belong
        # to sub-components
        for p in self.active_params:
            if p.name not in fiducial:
                fiducial[p.name] = p.fiducial

        # Now do all the components
        for cmp in self._subcomponents:
            # Overwrite global parameters.
            # Have to do this *before* filling sub-component, so that its
            # sub-components can be filled appropriately.
            for param in list(fiducial.keys()):
                if type(fiducial[param]) is dict:
                    continue

                if param in cmp.child_base_parameter_names:
                    if cmp.name in fiducial:
                        fiducial[cmp.name][param] = fiducial[param]
                    else:
                        fiducial[cmp.name] = {param: fiducial[param]}

            fiducial[cmp.name] = cmp._fill_params(fiducial.get(cmp.name, {}))

        # Now remove global parameters
        for param in list(fiducial.keys()):
            if param not in self._subcomponent_names + tuple(self.base_parameter_dct.keys()):
                del fiducial[param]

        return fiducial

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
        the same order as :func:`flat_active_params`.

        It only makes sense to call in active mode, and p must be a list
        or array of the same length as the total active params of this
        component and its subcomponents.

        Typically, this will be internally called in the process of fitting
        parameters, and only by the top-level object in the likelihood
        heirarchy.

        This returns a dict, where each parameter is inserted at its active level,
        *not* necessarily the component for which it is a parameter. The
        active parameters get filtered down into their components in _fill_params.
        """
        if len(p) != self.total_active_params:
            raise ValueError("length of parameter list not compatible with "
                             "active parameters. Please don't call "
                             "_parameter_list_to_dict yourself!")

        if type(p) is np.ndarray:
            p = list(p)

        dct = {}
        for name, param in self.flat_active_params.items():
            parent = param['parent']

            if parent == self.name:
                dct[name] = p.pop(0)
            else:
                parent = parent.replace(self.name+".", "")
                parents = parent.split(".")

                this = dct
                for prnt in parents:
                    if prnt not in this:
                        this[prnt] = {}

                    this = this[prnt]
                this[name] = p.pop(0)

        return dct


class Component(ParameterComponent, metaclass=plugin_mount_factory()):
    """
    A component of a likelihood. These are mainly for re-usability, so they
    can be mixed and matched inside likelihoods.
    """
    provides = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if type(cls.provides) not in [list, set, tuple]:
            raise TypeError("Component {} must defined a list/set/tuple for provides".format(cls.__name__))

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

    def __call__(self, **params):
        """
        Every component should take a dct of parameter values and return
        a dict of values to be used. All Parameters of the component will
        be in params.
        """
        if len(params) != len(self.base_parameters):
            params = self._fill_params(params)

        res = self.calculate(**params)
        if type(res) != tuple:
            res = tuple([res])

        if len(self.provides) != len(res):
            raise ValueError(
                "{} does not return an ordered iterable of the same length as its 'provides' attribute from calculate()".format(
                    self.__class__.__name__))

        return {p: r for p, r in zip(self.provides, res)}


class Likelihood(ParameterComponent, metaclass=plugin_mount_factory()):

    def __init__(self, name=None, params=None, fiducial=None, data=None,
                 derived=None, components=None):
        """

        Parameters
        ----------
        name
        params
        fiducial
        data : whatever data is pertinent to this likelihood
        derived : a list of strings or functions
        components : components "owned" by this likelihood. These are processed specifically
                     inside this likelihood, *after* any components processed by
                     the container.
        """
        self.components = tuple(components) if components is not None else tuple()

        super().__init__(name, params, fiducial, derived)

        self._data = data

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
                dquants.append(getattr(self, d)(model, params, ctx))
            elif callable(d):
                dquants.append(d(model, params, ctx))
            else:
                raise ValueError("{} is not a valid entry for derived".format(d))

        for cmp in self.components:
            dquants += cmp.derived_quantities(ctx, **params[cmp.name])

        return dquants

    @staticmethod
    def _search_dict(dct, key, default=None):
        """
        Search through a dict for a given key, and return its value
        if that key is a direct child of parent

        So _search_dict({"a":1, "b":{"a":2, "c":3}}, "a") == 1
        _search_dict({"a":1, "b":{"a":2, "c":3}, "f":{"a":3, "c":4}}, "c", "f") == 4
        """
        for k, v in dct.items():
            if k == key:
                return v
            elif type(v) is dict:
                res = Likelihood._search_dict(v, key, default)
                if res != default:
                    return res
        return default

    def logprior(self, **params):
        # This can be called in non-active mode, it will just return zero.
        params = self._fill_params(params)

        prior = 0

        for name, p in self.flat_active_params.items():
            loc = p['loc']

            if loc == self.name:
                prior += p['param'].logprior(params[name])
            else:
                loc = loc.replace(self.name+".", "")

                locs = loc.split(".")
                dct = params
                for l in locs:
                    dct = dct[l]

                prior += p['param'].logprior(dct[name])

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

        return self._reduce(ctx, **params)

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
        return self.lnl(model, **params)

    def logp(self, model=None, **params):
        return self.logprior(**params) + self.logl(model, **params)

    def __call__(self, params=None):
        if params is None:
            params = {}

        if type(params) is not dict:
            params = self._parameter_list_to_dict(params)

        ctx = self.get_ctx(**params)
        model = self._reduce(ctx, **params)

        return self.logp(model, **params), self.derived_quantities(model, ctx, **params)


class LikelihoodContainer(Likelihood):
    def __init__(self, likelihoods, **kwargs):
        self.likelihoods = tuple(likelihoods)
        if not hasattr(likelihoods, "__len__") or len(likelihoods) < 1:
            raise ValueError("likelihoods should be a tuple of at least one likelihood")

        for lk in self.likelihoods:
            assert isinstance(lk, Likelihood)
            lk._in_active_mode = True  # ensure everything is in active mode.

        super().__init__(**kwargs)

        # Change the name
        if "name" not in kwargs:
            self.name = " ".join([lk.name for lk in likelihoods])

        for cmp in self.components:
            assert isinstance(cmp, Component)

            for lk in self.likelihoods:
                assert cmp.name not in lk.components, "Do not use the same component both globally and in a likelihood!"

        for lk in likelihoods:
            if not lk.check_component_requirements(self.components + lk.components):
                raise DependencyError(
                    "likelihood {} does not have all component requirements satisfied".format(lk.name))

        for d in self.derived:
            assert callable(d)

    @cached_property
    def _subcomponents(self):
        return self.components + self.likelihoods

    def validate_derived(self):
        for d in self.derived:
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

            out[lk.name] = lk._reduce(this_ctx, **params[lk.name])

        return out

    def get_logls(self, models=None, **params):
        params = self._fill_params(params)

        if models is None:
            models = self._reduce_all(**params)

        out = {}
        for lk in self.likelihoods:
            out[lk.name] = lk.lnl(models[lk.name], **params[lk.name])

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
