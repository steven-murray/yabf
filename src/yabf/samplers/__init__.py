"""Implementations of popular samplers."""

import contextlib

__all__ = []

with contextlib.suppress(ImportError):
    from ._emcee import emcee

    __all__.append("emcee")

with contextlib.suppress(ImportError):
    from ._polychord import polychord

    __all__.append("polychord")
