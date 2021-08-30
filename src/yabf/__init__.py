"""Top-level package for yabf."""
try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass

from .core import mpi, samplers  # noqa
from .core.configio import (  # noqa
    load_from_yaml,
    load_likelihood_from_yaml,
    load_sampler_from_yaml,
)
from .core.likelihood import Component, Likelihood, LikelihoodContainer  # noqa
from .core.parameters import Param, Parameter, ParameterVector, ParamVec  # noqa
from .core.samplers import run_map  # noqa
from .samplers import *
