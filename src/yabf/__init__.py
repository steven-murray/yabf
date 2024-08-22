"""Top-level package for yabf."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

from .core import mpi, samplers
from .core.configio import (
    load_from_yaml,
    load_likelihood_from_yaml,
    load_sampler_from_yaml,
)
from .core.likelihood import Component, Likelihood, LikelihoodContainer
from .core.parameters import Param, Parameter, ParameterVector, ParamVec
from .core.samplers import run_map
from .samplers import emcee, polychord
