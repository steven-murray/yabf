"""Framework for likelihoods."""

from __future__ import annotations

import collections
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from functools import reduce
from typing import Any

import attr
import numpy as np
from attr import validators
from cached_property import cached_property

from . import mpi, utils
from .component import Component, ParameterComponent, _ComponentTree

logger = logging.getLogger(__name__)


class _LikelihoodInterface(ABC):
    """An abstract base class for likelihoods, defining the methods they must expose."""

    @abstractmethod
    def mock(
        self,
        model=None,
        ctx: [None, dict] = None,
        params: [None, Sequence, dict] = None,
    ):
        """Create a mock dataset given a set of parameters.

        Parameters
        ----------
        model
            The model to make a mock of.
        ctx
            Optional dictionary of component calculations.
        params
            The parameter values to use in the calculation. If a list (or tuple), must
            be of the same length as the active params of this component. If a dict,
            keys must be strings corresponding to the names of the active parameters,
            but not all parameters must be present (non-present keys are given default
            values specified by the component itself, or instance-level
            :attr:`fiducial_params`). By default, use all fiducial parameters.
        """

    @abstractmethod
    def derived_quantities(self, model=None, ctx=None, params=None):
        """Generate specified derived quantities given a set of parameters.

        Parameters
        ----------
        model
            The model to make a mock of.
        ctx
            Optional dictionary of component calculations.
        params
            The parameter values to use in the calculation. If a list (or tuple), must
            be of the same length as the active params of this component. If a dict,
            keys must be strings corresponding to the names of the active parameters,
            but not all parameters must be present (non-present keys are given default
            values specified by the component itself, or instance-level
            :attr:`fiducial_params`). By default, use all fiducial parameters.
        """

    @abstractmethod
    def logprior(self, params=None):
        """Generate the total logprior for all active parameters.

        Parameters
        ----------
        params
            The parameter values to use in the calculation. If a list (or tuple), must
            be of the same length as the active params of this component. If a dict,
            keys must be strings corresponding to the names of the active parameters,
            but not all parameters must be present (non-present keys are given default
            values specified by the component itself, or instance-level
            :attr:`fiducial_params`). By default, use all fiducial parameters.
        """

    @abstractmethod
    def get_ctx(self, params=None):
        """Generate the context, by running all components.

        Parameters
        ----------
        params
            The parameter values to use in the calculation. If a list (or tuple), must
            be of the same length as the active params of this component. If a dict,
            keys must be strings corresponding to the names of the active parameters,
            but not all parameters must be present (non-present keys are given default
            values specified by the component itself, or instance-level
            :attr:`fiducial_params`). By default, use all fiducial parameters.
        """

    @abstractmethod
    def logl(self, model=None, ctx=None, params=None):
        """Return the log-likelihood at the given parameters.

        Parameters
        ----------
        model
            The model to make a mock of.
        ctx
            Optional dictionary of component calculations.
        params
            The parameter values to use in the calculation. If a list (or tuple), must
            be of the same length as the active params of this component. If a dict,
            keys must be strings corresponding to the names of the active parameters,
            but not all parameters must be present (non-present keys are given default
            values specified by the component itself, or instance-level
            :attr:`fiducial_params`). By default, use all fiducial parameters.
        """

    @abstractmethod
    def logp(self, model=None, params=None):
        """Return the log-posterior at the given parameters.

        Parameters
        ----------
        model
            The model to make a mock of.
        params
            The parameter values to use in the calculation. If a list (or tuple), must
            be of the same length as the active params of this component. If a dict,
            keys must be strings corresponding to the names of the active parameters,
            but not all parameters must be present (non-present keys are given default
            values specified by the component itself, or instance-level
            :attr:`fiducial_params`). By default, use all fiducial parameters.
        """
        return self.logprior(**params) + self.logl(model, **params)

    @abstractmethod
    def __call__(self, params=None, ctx=None, with_prior: bool = True):
        """Return a tuple of the log-posterior and derived quantities at given params.

        Parameters
        ----------
        ctx
            Optional dictionary of component calculations.
        params
            The parameter values to use in the calculation. If a list (or tuple), must
            be of the same length as the active params of this component. If a dict,
            keys must be strings corresponding to the names of the active parameters,
            but not all parameters must be present (non-present keys are given default
            values specified by the component itself, or instance-level
            :attr:`fiducial_params`). By default, use all fiducial parameters.
        with_prior
            Whether to add the prior to the output posterior.
        """

    @cached_property
    def child_provides(self) -> set[str]:
        """All the provided quantities from this and child components."""
        return reduce(
            lambda x, y: x | y,
            (set(cmp.child_provides) for cmp in self._subcomponents),
            set(),
        )


def tuplify(x: Any) -> tuple:
    """Convert a value to a tuple."""
    return tuple(x) if hasattr(x, "__len__") else (x,)


@attr.s(frozen=True, kw_only=True)
class Likelihood(ParameterComponent, _LikelihoodInterface):
    _data = attr.ib(default=None)
    components = attr.ib(factory=tuple, converter=tuplify)
    _data_seed = attr.ib(validator=validators.optional(validators.instance_of(int)))
    _store_data = attr.ib(False, converter=bool)

    _plugins = {}

    def __init_subclass__(cls, is_abstract=False, **kwargs):
        """Enable plugins."""
        super().__init_subclass__(**kwargs)
        if not is_abstract:
            cls._plugins[cls.__name__] = cls

    @components.validator
    def _cmp_valid(self, att, val):
        for cmp in val:
            assert isinstance(cmp, Component), (
                f"component {cmp.name} is not a valid Component"
            )

    @cached_property
    def using_mock_data(self):
        """Whether we are using mock data."""
        return self._data is None and hasattr(self, "_mock")

    @_data_seed.default
    def _data_seed_default(self):
        if not self.using_mock_data:
            return None

        if mpi.more_than_one_process:
            raise TypeError(
                "if using MPI and auto-generated mock data, data_seed must be set"
            )
        return np.random.randint(0, 2**32 - 1)

    @cached_property
    def _subcomponents(self):
        return self.components

    @cached_property
    def data(self):
        """The data associated with the likelihood."""
        if self._data is None and not hasattr(self, "_mock"):
            raise AttributeError(
                "You have not passed any data and no mock method found"
            )

        if self.using_mock_data:
            # We only want to set the seed for data creation, and then
            # randomize afterwards.
            with utils.seed_as(self._data_seed):
                return self.mock()

        return self._data

    @classmethod
    def check_component_requirements(cls, components):
        """Check that the requirements of each sub-components are met."""
        return True

    def get_ctx(self, ctx=None, ignore_components=None, params=None) -> dict:
        """Obtain a full context dictionary for given parameters.

        Parameters
        ----------
        ctx
            Optional dictionary of component calculations.
        ignore_components
            A list of names of (sub-)components to *not* run.
        params
            The parameter values to use in the calculation. If a list (or tuple), must
            be of the same length as the active params of this component. If a dict,
            keys must be strings corresponding to the names of the active parameters,
            but not all parameters must be present (non-present keys are given default
            values specified by the component itself, or instance-level
            :attr:`fiducial_params`). By default, use all fiducial parameters.

        Returns
        -------
        ctx
            The fully-calculated set of values from all sub-components.
        """
        params = self._fill_params(params)

        if ctx is None:
            ctx = {}
        else:
            assert isinstance(ctx, collections.abc.Mapping)
            ctx = deepcopy(ctx)

        if all(k in ctx for k in self.child_provides):
            # We have all the keys in ctx already, don't update it.
            return ctx

        if ignore_components is None:
            ignore_components = [cmp.name for cmp, _ in self.common_components]

            for cmp, locs in self.common_components:
                ctx.update(
                    cmp(ctx=ctx, params=utils.get_loc_from_dict(params, locs[0]))
                )

        for cmp in self._subcomponents:
            if cmp.name in ignore_components:
                continue

            res = cmp(params=params[cmp.name])
            ctx.update(res)

        return ctx

    def _get_model_ctx_params(
        self, params, model=None, ctx=None, ignore_components=None
    ):
        params = self._fill_params(params)

        ctx = self.get_ctx(ctx=ctx, ignore_components=ignore_components, params=params)

        if model is None:
            model = self.reduce_model(ctx=ctx, params=params)

        return model, ctx, params

    def mock(self, model=None, ctx=None, ignore_components=None, params=None):
        """Create mock data at given parameters.

        Parameters
        ----------
        model
            The model to make a mock of.
        ctx
            An optional dictionary of calculated data into which will be inserted the
            results of this calculation.
        ignore_components
            A list of names of (sub-)components to *not* run.
        params
            The parameter values to use in the calculation. If a list (or tuple), must
            be of the same length as the active params of this component. If a dict,
            keys must be strings corresponding to the names of the active parameters,
            but not all parameters must be present (non-present keys are given default
            values specified by the component itself, or instance-level
            :attr:`fiducial_params`). By default, use all fiducial parameters.

        """
        model, _, params = self._get_model_ctx_params(
            params, model, ctx, ignore_components
        )
        return self._mock(model, **params)

    def derived_quantities(
        self, model=None, ctx=None, ignore_components=None, params=None
    ) -> list:
        """Obtain all derived quantities specified for the object.

        Parameters
        ----------
        model
            The model to make a mock of.
        ctx
            An optional dictionary of calculated data into which will be inserted the
            results of this calculation.
        ignore_components
            A list of names of (sub-)components to *not* run.
        params
            The parameter values to use in the calculation. If a list (or tuple), must
            be of the same length as the active params of this component. If a dict,
            keys must be strings corresponding to the names of the active parameters,
            but not all parameters must be present (non-present keys are given default
            values specified by the component itself, or instance-level
            :attr:`fiducial_params`). By default, use all fiducial parameters.

        Returns
        -------
        dquants

        """
        model, ctx, params = self._get_model_ctx_params(
            params, model, ctx, ignore_components
        )

        dquants = []
        for d in self.derived:
            if isinstance(d, str):
                # Append local quantity
                dquants.append(getattr(self, d)(model, ctx, **params))
            elif callable(d):
                dquants.append(d(model, ctx, **params))
            else:
                raise ValueError(f"{d} is not a valid entry for derived")

        for cmp in self._subcomponents:
            dquants += cmp.derived_quantities(ctx, params[cmp.name])

        return dquants

    def logprior(self, params=None):
        """Compute the log prior."""
        # This can be called in non-active mode, it will just return zero.
        params = self._fill_params(params, transform=False)

        prior = 0

        for apar in self.child_active_params:
            value = utils.get_loc_from_dict(params, apar.determines[0])
            prior += apar.logprior(value)

        return prior

    def reduce_model(self, ctx=None, ignore_components=None, params=None):
        """
        Reduce the model data produced by the components.

        This function returns the most reduced model data required to calculate the
        likelihood.
        """
        params = self._fill_params(params)

        ctx = self.get_ctx(ctx=ctx, ignore_components=ignore_components, params=params)

        top_level_params = {p: v for p, v in params.items() if not isinstance(v, dict)}
        return self._reduce(ctx, **top_level_params)

    def _reduce(self, ctx, **params):
        """Basic reduction just returns the ctx as a whole."""
        return ctx

    def logl(self, model=None, ctx=None, ignore_components=None, params=None):
        """Compute the log likelihood."""
        model, _, params = self._get_model_ctx_params(
            params, model, ctx, ignore_components
        )
        top_level_params = {
            p: v
            for p, v in params.items()
            if not isinstance(v, collections.abc.Mapping)
        }
        return self.lnl(model, **top_level_params)

    def logp(self, model=None, ctx=None, ignore_components=None, params=None):
        """Compute the log posterior."""
        untrans_params = self._fill_params(params, transform=False)
        return self.logprior(untrans_params) + self.logl(
            model=model, ctx=ctx, ignore_components=ignore_components, params=params
        )

    def __call__(
        self, params=None, ctx=None, ignore_components=None, with_prior: bool = True
    ):
        """Compute the log-posterior of the model given parameters."""
        if params is None:
            params = {}
        transformed_params = self._fill_params(params)

        ctx = self.get_ctx(ctx, ignore_components, transformed_params)

        ignore = [
            cmp.name for cmp in self.components
        ]  # ignore everything since ctx is full

        model = self.reduce_model(
            ctx, ignore_components=ignore, params=transformed_params
        )

        llfunc = self.logp if with_prior else self.logl
        return (
            llfunc(model, ignore_components=ignore, params=params),  # not transformed!
            self.derived_quantities(
                model, ctx, ignore_components=ignore, params=transformed_params
            ),
        )

    @cached_property
    def child_derived(self):
        """The derived quantities from this and all child components."""
        derived = self.derived

        for cmp in self._subcomponents:
            derived = derived + cmp.derived

        return derived

    def __getstate__(self):
        """Return a dictionary defining the likelihood."""
        dct = super().__getstate__()
        if self._store_data:
            dct.update(_data=self.data)
        return dct


@attr.s
class LikelihoodContainer(_LikelihoodInterface, _ComponentTree):
    likelihoods = attr.ib()
    _name = attr.ib(
        None, kw_only=True, validator=validators.optional(validators.instance_of(str))
    )

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
        """The name of the likelihood."""
        return self._name or "_".join(lk.name for lk in self.likelihoods)

    def get_ctx(self, params=None):
        """Generate the context by running all components in all likelihoods."""
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
                    params=params[lk.name],
                )
            )
        return ctx

    def reduce_model(self, ctx=None, params=None):
        """Get models from all likelihoods, as a dict of {likelihood_name: reduce}."""
        params = self._fill_params(params)

        if ctx is None:
            ctx = self.get_ctx(params)

        return {
            lk.name: lk.reduce_model(
                ctx[lk.name],
                ignore_components=[cmp.name for cmp in lk.components],
                params=params[lk.name],
            )
            for lk in self.likelihoods
        }

    def get_logls(self, model=None, ctx=None, params=None):
        """Compute all log-likelihoods in this container."""
        params = self._fill_params(params)

        if model is None:
            model = self.reduce_model(ctx, params)

        return {
            lk.name: lk.lnl(model[lk.name], **params[lk.name])
            for lk in self.likelihoods
        }

    def logl(self, model=None, ctx=None, params=None):
        """Compute the total log-likelihood."""
        params = self._fill_params(params)
        logls = self.get_logls(model, ctx, params)
        return sum(logls.values())

    def logprior(self, params=None):
        """Compute the log-prior."""
        params = self._fill_params(params)
        return Likelihood.logprior(self, params=params)

    def logp(self, model=None, ctx=None, params=None):
        """Compute the log-posterior."""
        logger.info(f"Params: {params}")
        params = self._fill_params(params)
        logl = self.logl(model, ctx, params)  # this fills the params
        logprior = self.logprior(params)
        logger.info(f"logl: {logl}, prior: {logprior}")
        if np.isnan(logl) or np.isinf(logl):
            logger.warn(f"Got bad logl: {logl}, with params: {params}")
        if np.isnan(logprior):
            logger.warn(f"Got bad logprior: {logprior} with params: {params}")
        if np.isinf(logprior):
            logger.warn(f"prior out of bounds for params: {params}")
        return logl + logprior

    @cached_property
    def derived(self):
        """All derived properties of the inference."""
        return sum((lk.child_derived for lk in self.likelihoods), ())

    @property
    def child_derived(self):
        """All child derived properties of the inference. Alias for derived."""
        return self.derived

    def derived_quantities(self, model=None, ctx=None, params=None):
        """Compute the derived quantities of the model."""
        params = self._fill_params(params)

        if ctx is None:
            ctx = self.get_ctx(params)

        if model is None:
            model = self.reduce_model(ctx, params)

        dquants = []
        for lk in self.likelihoods:
            dquants += lk.derived_quantities(
                model[lk.name], ctx[lk.name], params[lk.name]
            )

        return dquants

    def mock(self, model=None, ctx=None, ignore_components=None, params=None):
        """Generate mock data according to the model."""
        if ctx is None:
            ctx = self.get_ctx(params)

        if model is None:
            model = self.reduce_model(ctx, params)

        return {
            lk.name: lk.mock(model=model[lk.name], ctx=ctx[lk.name], params=params)
            for lk in self.likelihoods
        }

    def __call__(self, params=None, ctx=None, with_prior: bool = True):
        """Compute the log-posterior or log-likelihood of the model."""
        params = self._fill_params(params)
        if ctx is None:
            ctx = self.get_ctx(params)
        model = self.reduce_model(ctx=ctx, params=params)
        llfunc = self.logp if with_prior else self.logl

        return (
            llfunc(model=model, params=params),
            self.derived_quantities(model, ctx, params),
        )
