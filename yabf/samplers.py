"""
Module defining the API for Samplers
"""

import attr
from attr.validators import instance_of
import numpy as np
import pypolychord as ppc
from cached_property import cached_property
from emcee import EnsembleSampler
from getdist import MCSamples
from pypolychord.priors import UniformPrior
from pypolychord.settings import PolyChordSettings
import os
import warnings

from . import Likelihood
from .plugin import plugin_mount_factory
from . import yaml


@attr.s
class Sampler(metaclass=plugin_mount_factory()):
    likelihood = attr.ib(validator=instance_of(Likelihood))
    _output_prefix = attr.ib(validator=instance_of(str))
    _save_full_config = attr.ib(default=True, converter=bool)
    sampler_kwargs = attr.ib(default={})

    def __attrs_post_init__(self):
        self.mcsamples = None

        # Save the configuration
        if self._save_full_config:
            with open(self.config_filename) as fl:
                yaml.dump(self.likelihood, fl)

    @_output_prefix.default
    def _op_default(self):
        return self.likelihood.name

    @cached_property
    def output_dir(self):
        dir = os.path.abspath(os.path.dirname(self._output_prefix))

        # Try to create the directory.
        try:
            os.makedirs(dir)
            warnings.warn("Proposed output directory '{}' did not exist. Created it.".format(dir))
        except FileExistsError:
            pass
        return dir

    @cached_property
    def output_file_prefix(self):
        return os.path.basename(self._output_prefix)

    @cached_property
    def config_filename(self):
        return os.path.join(self.output_dir, self.output_file_prefix) + "_config.yml"

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
        Return an object that contains the sampling settings and potentially a
        method to actually perform sampling

        This could actually be nothing, and the class could rely purely
        on the :func:`_get_sampling_fn` method to create the sampler.
        """
        return None

    def _get_sampling_fn(self, sampler):
        pass

    def sample(self, **kwargs):
        samples = self._sample(self._sampling_fn, **kwargs)
        self.mcsamples = self._samples_to_mcsamples(samples)
        return self.mcsamples

    def _sample(self, sampling_fn, **kwargs):
        pass

    def _samples_to_mcsamples(self, samples):
        """Return posterior samples, with shape (<...>, NPARAMS, NITER)"""
        pass


class emcee(Sampler):

    @cached_property
    def nwalkers(self):
        return self.sampler_kwargs.pop("nwalkers", self.nparams * 2)

    def _get_sampler(self, **kwargs):
        # This is bad, but I have to access this before passing kwargs,
        # otherwise nwalkers is passed twice.
        if 'nwalkers' in kwargs:
            del kwargs['nwalkers']

        return EnsembleSampler(
            log_prob_fn=self.likelihood,
            ndim=self.nparams,
            nwalkers=self.nwalkers,
            **kwargs
        )

    def _get_sampling_fn(self, sampler):
        return sampler.run_mcmc

    def _sample(self, sampling_fn, **kwargs):
        refs = np.array(self.likelihood.generate_refs(n=self.nwalkers))

        sampling_fn(
            refs.T,
            **kwargs
        )

        return self._sampler

    def _samples_to_mcsamples(self, samples):
        # 'samples' is whatever is returned by _sample, in this case it is
        # the entire EnsembleSampler
        return MCSamples(
            samples=samples.get_chain(flat=False, discard=500),
            names=[a.name for a in self.likelihood.child_active_params],
            labels=[p.latex for p in self.likelihood.child_active_params]
        )


class polychord(Sampler):
    @staticmethod
    def _flat_array(elements):
        lst = []

        if hasattr(elements, "__len__"):
            for e in elements:
                lst += polychord._flat_array(e)
        else:
            lst.append(elements)

        return lst

    @cached_property
    def nderived(self):
        # this is a bad hack
        return len(self._flat_array(self.likelihood()[1]))

    @cached_property
    def log_prior_volume(self):
        prior_volume = 0
        for p in self.likelihood.child_active_params:
            if np.isinf(p.min) or np.isinf(p.max):
                raise ValueError("Polychord requires bounded priors")
            prior_volume += np.log(p.max - p.min)
        return prior_volume

    @cached_property
    def prior(self):
        # Determine proper prior.
        def pr(hypercube):
            ret = []
            for p, h in zip(self.likelihood.child_active_params, hypercube):
                ret.append(UniformPrior(p.min, p.max)(h))
            return ret

        return pr

    @cached_property
    def posterior(self):
        def posterior(p):
            lnl, derived = self.likelihood(p)
            return max(lnl + self.log_prior_volume, 0.99 * np.nan_to_num(-np.inf)), np.array(self._flat_array(derived))

        return posterior

    def _make_paramnames_files(self, mcsamples):
        paramnames = [(p.name, p.latex) for p in self.likelihood.child_active_params]

        # also have to add derived...
        paramnames += [(f'der{i}', f'der{i}') for i in range(self.nderived)]
        mcsamples.make_paramnames_files(paramnames)

    def _get_sampler(self, **kwargs):
        if "file_root" in kwargs:
            warnings.warn("file_root was defined in sampler_kwargs, but is replaced by output_prefix")
            del kwargs['file_root']

        if "base_dir" in kwargs:
            warnings.warn("base_dir was defined in sampler_kwargs, but is replaced by output_prefix")
            del kwargs['base_dir']

        return PolyChordSettings(
            self.nparams, self.nderived, base_dir=self.output_dir,
            file_root=self.output_file_prefix, **kwargs
        )

    def _get_sampling_fn(self, sampler):
        return ppc.run_polychord

    def _sample(self, sampling_fn, **kwargs):
        settings = self._sampler

        return sampling_fn(
            self.posterior, self.nparams, self.nderived,
            settings=settings, prior=self.prior
        )

    def _samples_to_mcsamples(self, samples):
        self._make_paramnames_files(samples)
        return samples.posterior
