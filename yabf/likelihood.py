"""
Framework for MCMC likelihoods (and parameters).
"""
import warnings
from collections import OrderedDict
from copy import copy

from .parameters import Param, _ActiveParameter

import numpy as np
from cached_property import cached_property


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
    all_parameters = []

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
        return {p.name: p for p in self.parameter_list}

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

    def _fill_params(self, params):
        fiducial = copy(self.fiducial_params)
        fiducial.update(params)
        return fiducial


class Component(ParameterComponent):
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


class Likelihood(ParameterComponent):

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
        for param in params:
            if type(param) is not dict and param not in self.all_parameter_dct:
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

        dct = {param: p.pop(0) for param in self.active_params}

        for cmp in self.components:
            dct[cmp.name] = {param: p.pop(0) for param in cmp.active_params}

        return dct

    @cached_property
    def flat_active_params(self):
        result = [(self, p) for p in self.active_params]

        for cmp in self.components:
            result += [(cmp, p) for p in cmp.active_params]

        return result

    def logl(self, model=None, **params):
        params = self._fill_params(params)

        if model is None:
            model = self.model(self.get_ctx(**params), **params)

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


class LikelihoodContainer:
    def __init__(self, likelihoods, components=None, params=None, derived=None):

        self.likelihoods = likelihoods
        if getattr(likelihoods, "__len__", 0) < 1:
            raise ValueError("likelihoods should be a list of at least one likelihood")

        for lk in likelihoods:
            assert isinstance(lk, Likelihood)
            lk._in_active_mode = True # ensure everything is in active mode.

        self.components = components or []
        for cmp in self.components:
            assert isinstance(cmp, Component)

            for lk in self.likelihoods:
                assert cmp.name not in lk.components, "Do not use the same component both globally and in a likelihood!"

        self.global_params = params or []
        for prm in self.global_params:
            assert isinstance(prm, Param)

        self.derived = derived or []
        for d in self.derived:
            assert callable(d)

    def _parameter_list_to_dict(self, p):
        """
        This defines the order in which the parameters should arrive
        """
        dct = {param: p.pop(0) for param in self.global_params}

        for cmp in self.components:
            dct[cmp.name] = {param: p.pop(0) for param in cmp.active_params}

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
        p = set(self.active_params.keys())
        for cmp in self.components:
            p.update(set(cmp.active_params.keys()))

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
        result = [(self, p) for p in self.active_params]

        for cmp in self.components:
            result += [(cmp, p) for p in cmp.active_params]

        for lk in self.likelihoods:
            result += lk.flat_active_params

        return result

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

        # Don't do the likelihood context here, since it should be done
        # individually per-likelihood before calling its model.
        # for lk in self.likelihoods:
        #     ctx.update(lk.get_ctx(**params[lk.name]))

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
        params = self._fill_params(params)

        prior = np.sum([p.logprior(params[k]) for k, p in self.active_params.items()])

        for cmp in self.components:
            prior += np.sum([p.logprior(params[cmp.name][k]) for k, p in cmp.active_params.items()])

        for lk in self.likelihoods:
            prior += lk.logprior

        return prior

    def logp(self, models=None,**params):
        logl = self.logl(models, **params) # this fills the params
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
