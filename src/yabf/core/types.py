"""Derived types for use in yabf."""

from __future__ import annotations

import typing as tp

import numpy as np

numeric = tp.Union[int, float, np.number]
