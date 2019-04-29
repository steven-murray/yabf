"""
Framework for MCMC likelihoods (and parameters).
"""
import warnings
from collections import OrderedDict
from copy import copy

import numpy as np
from cached_property import cached_property
from .plugin import plugin_mount_factory

from .parameters import Param, _ActiveParameter


def _only_active(func):
    def func_wrapper(self, *args, **kwargs):
        if not self.in_active_mode:
            raise AttributeError("{} is not available when not in active mode.")
        return func(self, *args, **kwargs)

    return func_wrapper


class ParameterComponent:
    """
    A base class for named components and likelihoods that take parameters.
    """
    parameters = []

    def __init__(self, name=None, params=None, fiducial=None, derived=None):
        self.name = name or self.__class__.__name__
        self._constrained_params = OrderedDict([(p.name, p) for p in params])
        self.derived = derived or []

        for name, value in fiducial.items():
            if name in self._constrained_params:
                raise ValueError("Pass fiducial values to constrained parameters inside the Param class")
            if name not in self.all_parameter_dct:
                raise KeyError("parameter {} not found in set of all parameters".format(name))

        self._fixed_fiducial = fiducial
        self.validate_derived()

        self._in_active_mode = len(self.active_params) > 0

    def validate_derived(self):
        for d in self.derived:
            assert (callable(d) or (type(d) is str and hasattr(self, d))), f"{d} is not a valid derived parameter"

    @property
    def in_active_mode(self):
        """
        Bool representing whether this class is in active mode (i.e. whether
        it is being actively constrained).
        """
        return self._in_active_mode

    @cached_property
    def all_parameter_dct(self):
        """
        All possible parameters of this specific component or likelihood,
        not just those that are being constrained.
        """
        return {p.name: p for p in self.parameters}

    @cached_property
    def active_params(self):
        """
        Actively constrained parameters
        """
        return OrderedDict(
            [(p, _ActiveParameter(self.all_parameter_dct[p], v)) for p, v in self._constrained_params.items()])

    @cached_property
    def fiducial_params(self):
        dct = {}
        for name in self.all_parameter_dct:
            if name in self.active_params:
                dct[name] = self.active_params[name].fiducial_value
            elif name in self._fixed_fiducial:
                dct[name] = self._fixed_fiducial[name]
            else:
                dct[name] = self.all_parameter_dct[name].default

        return dct

    @_only_active
    def generate_refs(self, n=1, squeeze=False, full=False, params=None):
        """
        Generate reference values for active parameters from their priors.

        Parameters
        ----------
        n : int, optional
            Number of reference values to choose for each parameter.
        squeeze : bool, optional
            If ``n=1``, ``squeeze`` will ensure that the returned results
            are one-dimensional.
        full : bool, optional
            Whether to return values for _all_ parameters, even those not
            active. If so, the returned object is a dict (as the parameters
            are not ordered).
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
        if params is None:
            params = self.active_params.keys()

        if full:
            refs = {}
        else:
            refs = []

        for param in params:
            if param in self.active_params:
                ref = self.active_params[param].generate_ref(n)
            else:
                ref = self.fiducial_params[param]

            if not squeeze and n == 1:
                ref = [ref]

            if full:
                refs[param] = ref
            else:
                refs.append(ref)

        if full:
            for param in self.all_parameter_dct:
                if param not in refs:
                    refs[param] = [self.fiducial_params[param]] * n
                    if squeeze and n == 1:
                        refs[param] = refs[param][0]

        return refs

    def _fill_params(self, params):
        fiducial = copy(self.fiducial_params)
        fiducial.update(params)
        return fiducial


class Component(ParameterComponent, metaclass=plugin_mount_factory()):
    """
    A component of a likelihood. These are mainly for re-usability, so they
    can be mixed and matched inside likelihoods.
    """
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
        if len(params) != len(self.all_parameter_dct):
            params = self._fill_params(params)

        return self.calculate(**params)


class Likelihood(ParameterComponent, metaclass=plugin_mount_factory(), ):

    def __init__(self, name=None, params=None, fiducial=None, data=None, derived=None, components=None):
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
        super().__init__(name, params, fiducial, derived)

        self.components = components or []

        if data is None:
            if hasattr(self, "mock"):
                self.data = self.mock()

            else:
                warnings.warn("You have not passed any data... logp will not work!")
                self.data = None
        else:
            self.data = data

    def mock(self, model=None, ctx=None, **params):
        """
        Create mock data at given parameters.
        """
        model, params = self._get_model_and_params(params, model, ctx)
        return self._mock(model, **params)

    @property
    def in_active_mode(self):
        return any([super().in_active_mode] + [cmp.in_active_mode for cmp in self.components])

    @cached_property
    def all_active_params(self):
        """Set of all active parameter names in this likelihood and its sub-components"""
        p = set(self.active_params.keys())
        for cmp in self.components:
            p.update(set(cmp.active_params.keys()))

        return p

    @cached_property
    def all_available_params(self):
        """Set of all available parameter names in this likelihood and its sub-components"""
        p = set(self.all_parameter_dct.keys())
        for cmp in self.components:
            p.update(set(cmp.all_parameter_dct.keys()))
        return p

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

    def logprior(self, **params):
        # This can be called in non-active mode, it will just return zero.
        params = self._fill_params(params)
        prior = np.sum([p.logprior(params[k]) for k, p in self.active_params.items()])

        for cmp in self.components:
            prior += np.sum([p.logprior(params[cmp.name][k]) for k, p in cmp.active_params.items()])

        return prior

    def get_ctx(self, **params):
        params = self._fill_params(params)

        ctx = {}

        for cmp in self.components:
            ctx.update(cmp(**params[cmp.name]))

        return ctx

    def _is_params_full(self, params):
        if len(params) != len(self.components) + len(self.all_parameter_dct):
            return False

        for cmp in self.components:
            if len(params.get(cmp.name, {})) != len(cmp.all_parameter_dct):
                return False

        return True

    @cached_property
    def total_active_params(self):
        return len(self.flat_active_params)

    def _fill_params(self, params):
        # Note: this function does _not_ depend on the active parameters of
        #       either itself or its sub-components. As long as the params
        #       passed have the correct structure, it will fill them.

        if self._is_params_full(params):
            return params

        # This fills up the dict with this likelihood's parameters
        params = super()._fill_params(params)

        # Now do all the components
        for cmp in self.components:
            params[cmp.name] = cmp._fill_params(params.get(cmp.name, {}))

            # Overwrite global parameters
            for param in params[cmp.name]:
                if param in params:
                    params[cmp.name][param] = params[param]

        # Now remove global parameters
        for param in list(params.keys()):
            if type(params[param]) is not dict and param not in self.all_parameter_dct:
                del params[param]

        return params

    def model(self, ctx=None, **params):
        params = self._fill_params(params)

        if ctx is None:
            ctx = self.get_ctx(**params)

        return self._model(ctx, **params)

    @_only_active
    def _parameter_list_to_dict(self, p):
        """
        This defines the order in which the parameters should arrive
        """
        if len(p) != self.total_active_params:
            raise ValueError("length of parameter list not compatible with active"
                             " parameters. Please don't call _parameter_list_to_dict yourself!")

        if type(p) is np.ndarray:
            p = list(p)

        dct = {param: p.pop(0) for param in self.active_params}

        for cmp in self.components:
            dct[cmp.name] = {param: p.pop(0) for param in cmp.active_params}

        return dct

    @cached_property
    def flat_active_params(self):
        result = OrderedDict([(p, dict(parent=self.name, param=v)) for p, v in self.active_params.items()])

        for cmp in self.components:
            result.update([(p, dict(parent=cmp.name, param=v)) for p, v in cmp.active_params.items()])

        return result

    def _get_model_and_params(self, params, model=None, ctx=None):
        params = self._fill_params(params)

        if ctx is None:
            ctx = self.get_ctx(**params)

        if model is None:
            model = self.model(ctx, **params)

        return model, params

    def logl(self, model=None, ctx = None, **params):
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
        model = self.model(ctx, **params)

        return self.logp(model, **params), self.derived_quantities(model, ctx, **params)

    @_only_active
    def generate_refs(self, n=1, squeeze=False, full=False, params=None):
        refs = super().generate_refs()

        for cmp in self.components:
            if params is not None:
                thisparams = [p for p in params if p in cmp.all_parameter_dct]
            else:
                thisparams = None

            ref = cmp.generate_refs(n=n, squeeze=squeeze, full=full, params=thisparams)

            if full:
                refs.update(ref)
            else:
                refs += ref

        return refs


class LikelihoodContainer(Likelihood):
    def __init__(self, likelihoods, **kwargs):
        super().__init__(**kwargs)

        # Change the name
        if "name" not in kwargs:
            self.name = " ".join([lk.name for lk in likelihoods])

        self.likelihoods = likelihoods
        if not hasattr(likelihoods, "__len__") or len(likelihoods) < 1:
            raise ValueError("likelihoods should be a list of at least one likelihood")

        for lk in likelihoods:
            assert isinstance(lk, Likelihood)
            lk._in_active_mode = True  # ensure everything is in active mode.

        for cmp in self.components:
            assert isinstance(cmp, Component)

            for lk in self.likelihoods:
                assert cmp.name not in lk.components, "Do not use the same component both globally and in a likelihood!"

        for d in self.derived:
            assert callable(d)

    def validate_derived(self):
        for d in self.derived:
            assert callable(d), f"{d} is not a valid derived parameter"

    @cached_property
    def all_parameter_dct(self):
        raise AttributeError("LikelihoodContainer has no parameters of its own.")

    def _parameter_list_to_dict(self, p):
        """
        This defines the order in which the parameters should arrive
        """
        if type(p) is np.ndarray:
            p = list(p)

        dct = super()._parameter_list_to_dict(p)

        for lk in self.likelihoods:
            dct[lk.name] = lk._parameter_list_to_dct(p)

        return dct

    @cached_property
    def all_active_params(self):
        """
        Set of all active parameter names in this likelihood and its sub-components

        NOTE: does not distinguish between components, and whether parameters
              are treated separately. To get that information, one must
              traverse the tree of likelihoods and components' active_params.
        """
        p = super().all_active_params

        for lk in self.likelihoods:
            p.update(lk.all_active_params)

        return p

    @cached_property
    def all_available_params(self):
        p = set()
        for cmp in self.components:
            p.update(set(cmp.all_parameter_dict.keys()))
        for lk in self.likelihoods:
            p.update(lk.all_active_params)
        return p

    @cached_property
    def flat_active_params(self):
        result = super().flat_active_params

        for lk in self.likelihoods:
            result.update(lk.flat_active_params)

        return result

    def _fill_params(self, params):
        # Note: this function does _not_ depend on the active parameters of
        #       either itself or its sub-components. As long as the params
        #       passed have the correct structure, it will fill them.

        if self._is_params_full(params):
            return params

        # This fills up the dict with this likelihood's parameters
        params = super()._fill_params(params)

        # Now do all the likelihoods
        for lk in self.likelihoods:
            # First overwrite globals
            for param in params:
                if param in lk.all_parameters_dict:
                    params[lk.name][param] = params[param]

            params[lk.name] = lk._fill_params(params.get(lk.name, {}))

        # Now remove global parameters
        for param in params:
            if type(param) is not dict:
                del params[param]

        return params

    def _is_params_full(self, params):
        if len(params) != len(self.components) + len(self.likelihoods):
            return False

        for cmp in self.components:
            if len(params.get(cmp.name, {})) != len(cmp.all_parameter_dct):
                return False

        for lk in self.likelihoods:
            if not lk._is_params_full(params.get(lk.name, {})):
                return False

        return True

    def get_ctx(self, **params):
        params = self._fill_params(params)

        ctx = {}
        for cmp in self.components:
            ctx.update(cmp(**params[cmp.name]))

        return ctx

    def get_models(self, ctx=None, **params):
        """Get models from all likelihoods, as a dictionary of likelihood_name: model"""
        params = self._fill_params(params)

        if ctx is None:
            ctx = self.get_ctx(**params)

        out = {}
        for lk in self.likelihoods:
            this_ctx = lk.get_ctx(params[lk.name])
            this_ctx.update(ctx)

            out[lk.name] = lk._model(this_ctx, **params[lk.name])

        return out

    def get_logls(self, models=None, **params):
        params = self._fill_params(params)

        if models is None:
            models = self.get_models(**params)

        out = {}
        for lk in self.likelihoods:
            out[lk.name] = lk.lnl(models[lk.name], **params[lk.name])

        return out

    def logl(self, models=None, **params):
        logsls = self.get_logls(models, **params)
        return sum(logsls.values())

    def logprior(self, **params):
        # This can be called in non-active mode, it will just return zero.
        prior = super().logprior(**params)

        for lk in self.likelihoods:
            prior += lk.logprior

        return prior

    def logp(self, models=None, **params):
        logl = self.logl(models, **params)  # this fills the params
        return logl + self.logprior(**params)

    def derived_quantities(self, models=None, ctx=None, **params):
        params = self._fill_params(params)

        if ctx is None:
            ctx = self.get_ctx(**params)

        if models is None:
            models = self.get_models(ctx, **params)

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
        models = self.get_models(ctx, **params)

        return self.logp(models, **params), self.derived_quantities(models, ctx, **params)

    @_only_active
    def generate_refs(self, n=1, squeeze=False, full=False, params=None):
        refs = super().generate_refs()

        for lk in self.likelihoods:

            if params is not None:
                thisparams = [p for p in params if p in lk.all_available_params]
            else:
                thisparams = None

            ref = lk.generate_refs(n=n, squeeze=squeeze, full=full, params=thisparams)

            if full:
                refs.update(ref)
            else:
                refs += ref

        return refs
