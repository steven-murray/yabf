"""Module defining the API for Samplers."""

import attr
import numpy as np
import yaml
from attr.validators import instance_of
from cached_property import cached_property
from pathlib import Path
from scipy.optimize import curve_fit as _curve_fit
from scipy.optimize import minimize

from yabf.core.likelihood import _LikelihoodInterface

from . import mpi
from .likelihood import Likelihood
from .plugin import plugin_mount_factory


@attr.s
class Sampler(metaclass=plugin_mount_factory()):
    likelihood = attr.ib(validator=[instance_of(_LikelihoodInterface)])
    _output_dir = attr.ib(default="", converter=Path, validator=instance_of(Path))
    _output_prefix = attr.ib(converter=Path, validator=instance_of(Path))
    _save_full_config = attr.ib(default=True, converter=bool)
    sampler_kwargs = attr.ib(default={})

    def __attrs_post_init__(self):
        """Define extra attributes depending on the input ones."""
        self.mcsamples = None

        # Save the configuration
        if self._save_full_config:
            with open(self.config_filename, "w") as fl:
                yaml.dump(self.likelihood, fl)

    @likelihood.validator
    def _lk_vld(self, attribute, val):
        assert isinstance(
            val, _LikelihoodInterface
        ), "likelihood must expose a _LikelihoodInterface"
        assert (
            len(val.child_active_params) > 0
        ), "likelihood does not have any active parameters!"

    @_output_prefix.default
    def _op_default(self):
        return self.likelihood.name

    @cached_property
    def output_dir(self):
        """The directory into which the sampler will write information."""
        if not self._output_prefix.absolute():
            direc = self._output_dir / self._output_prefix.parent
        else:
            direc = self._output_prefix.parent

        if not direc.exists():
            direc.mkdir(parents=True)

        return direc

    @cached_property
    def output_file_prefix(self):
        return self._output_prefix.name

    @cached_property
    def config_filename(self):
        return self.output_dir / f"{self.output_file_prefix}_config.yml"

    @cached_property
    def nparams(self):
        return self.likelihood.total_active_params

    @cached_property
    def _sampler(self):
        return self._get_sampler(**self.sampler_kwargs)

    @cached_property
    def _sampling_fn(self):
        return self._get_sampling_fn(self._sampler)

    def _get_sampler(self, **kwargs):
        """
        Return an object that contains the sampling settings.

        The returned object may also contain a method to perform sampling.

        This could actually be nothing, and the class could rely purely
        on the :func:`_get_sampling_fn` method to create the sampler.
        """
        return None

    def _get_sampling_fn(self, sampler):
        pass

    def sample(self, **kwargs):
        samples = self._sample(self._sampling_fn, **kwargs)

        mpi.sync_processes()

        self.mcsamples = self._samples_to_mcsamples(samples)
        return self.mcsamples

    def _sample(self, sampling_fn, **kwargs):
        pass

    def _samples_to_mcsamples(self, samples):
        """Return posterior samples, with shape (<...>, NPARAMS, NITER)."""
        pass


def run_map(likelihood, x0=None, bounds=None, **kwargs):
    """Run a maximum a-posteriori fit."""

    def objfunc(p):
        return -likelihood.logp(params=p)

    if x0 is None:
        x0 = np.array([apar.fiducial for apar in likelihood.child_active_params])

    eps = kwargs.get("options", {}).get("eps", 1e-8)
    if bounds is None:
        bounds = []
        for apar in likelihood.child_active_params:
            bounds.append(
                (
                    apar.min + 2 * eps if apar.min > -np.inf else None,
                    apar.max - 2 * eps if apar.max < np.inf else None,
                )
            )

    elif not bounds:
        bounds = None

    res = minimize(objfunc, x0, bounds=bounds, **kwargs)
    return res


def curve_fit(likelihood: Likelihood, x0=None, bounds=None, **kwargs):
    """Use scipy's curve_fit to do LM to find the MAP.

    Parameters
    ----------
    likelihood
        In this case the likelihood must be a subclass of Chi2.
    x0
        The initial guess
    bounds
        A list of tuples of parameters bounds, or False if no bounds are to be set. If
        None, use the min/max bounds on each parameter in the likelihood.
    """

    def model(x, *p):
        return likelihood.reduce_model(params=p)

    if x0 is None:
        x0 = np.array([apar.fiducial for apar in likelihood.child_active_params])

    eps = kwargs.get("options", {}).get("eps", 1e-8)
    if bounds is None:
        bounds = (
            [apar.min + 2 * eps for apar in likelihood.child_active_params],
            [apar.max - 2 * eps for apar in likelihood.child_active_params],
        )

    elif not bounds:
        bounds = (-np.inf, np.inf)

    res = _curve_fit(
        model,
        xdata=np.linspace(0, 1, len(likelihood.data)),
        ydata=likelihood.data,
        p0=x0,
        sigma=likelihood.sigma,
        bounds=bounds,
        **kwargs,
    )

    return res
