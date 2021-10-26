"""Derived types for use in yabf."""
from __future__ import annotations

import numpy as np
import typing as tp

numeric = tp.Union[int, float, np.number]
