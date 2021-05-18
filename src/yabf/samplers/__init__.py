"""Implementations of popular samplers."""

try:
    from .emcee import emcee
except ImportError:
    pass

try:
    from .polychord import polychord
except ImportError:
    pass
