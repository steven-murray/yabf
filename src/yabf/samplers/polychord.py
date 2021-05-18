"""The polychord sampler, wrapped into yabf."""

import numpy as np
import pypolychord as ppc
import warnings
from cached_property import cached_property
from pypolychord.priors import UniformPrior
from pypolychord.settings import PolyChordSettings

from ..core import mpi
from ..core.samplers import Sampler


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
    def __derived_sample(self):
        # a bad hack to get a sample of derived quantities
        # in order to understand the shape of the derived quantities
        return self.likelihood()[1]

    @cached_property
    def derived_shapes(self):
        return [getattr(d, "shape", 0) for d in self.__derived_sample]

    @cached_property
    def nderived(self):
        # this is a bad hack
        return len(self._flat_array(self.__derived_sample))

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
            return (
                max(lnl + self.log_prior_volume, 0.99 * np.nan_to_num(-np.inf)),
                np.array(self._flat_array(derived)),
            )

        return posterior

    @staticmethod
    def _index_to_string(*indx):
        return "_" + "_".join(str(i) for i in indx)

    @staticmethod
    def _index_to_latex(*indx):
        return r"_{" + ",".join(str(i) for i in indx) + r"}"

    def get_derived_paramnames(self):
        """Create a list of tuples specifying derived parameter names."""
        names = []
        for name, shape in zip(self.likelihood.child_derived, self.derived_shapes):
            if shape == 0:
                names.append((name, name))
            else:
                names.extend(
                    [
                        (
                            name + self._index_to_string(*ind),
                            name + self._index_to_latex(*ind),
                        )
                        for ind in np.ndindex(*shape)
                    ]
                )
        return names

    def _make_paramnames_files(self, mcsamples):
        paramnames = [
            (p.name + "*", p.latex) for p in self.likelihood.child_active_params
        ]

        # also have to add derived...
        paramnames += self.get_derived_paramnames()
        mcsamples.make_paramnames_files(paramnames)

    def _get_sampler(self, **kwargs):
        if "file_root" in kwargs:
            warnings.warn(
                "file_root was defined in sampler_kwargs, "
                "but is replaced by output_prefix"
            )
            del kwargs["file_root"]

        if "base_dir" in kwargs:
            warnings.warn(
                "base_dir was defined in sampler_kwargs, "
                "but is replaced by output_prefix"
            )
            del kwargs["base_dir"]

        return PolyChordSettings(
            self.nparams,
            self.nderived,
            base_dir=str(self.output_dir),
            file_root=str(self.output_file_prefix),
            nlive=kwargs.pop("nlive", 100 * self.nparams),
            **kwargs,
        )

    def _get_sampling_fn(self, sampler):
        return ppc.run_polychord

    def _sample(self, sampling_fn, **kwargs):
        settings = self._sampler

        return sampling_fn(
            self.posterior,
            self.nparams,
            self.nderived,
            settings=settings,
            prior=self.prior,
        )

    def _samples_to_mcsamples(self, samples):
        if mpi.am_single_or_primary_process:
            self._make_paramnames_files(samples)
            # do initialization...
            samples.posterior

        mpi.sync_processes()
        return samples.posterior
