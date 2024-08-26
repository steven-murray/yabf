"""Module defining data loaders for YAML files."""

import inspect
import pickle
from functools import wraps
from typing import Callable

import numpy as np
import yaml

_DATA_LOADERS = {}


class LoadError(ValueError):
    pass


def data_loader(tag=None):
    """A decorator that defines a function that loads data."""

    def inner(fnc):
        _DATA_LOADERS[fnc.__name__] = fnc

        new_tag = tag or fnc.__name__.split("_loader")[0]
        fnc.tag = new_tag

        # ensure it only takes path to data.
        assert len(inspect.signature(fnc).parameters) == 1

        @wraps(fnc)
        def wrapper(data):
            try:
                return fnc(data)
            except FileNotFoundError:
                raise
            except Exception as e:
                raise LoadError(str(e)) from e

        def yaml_fnc(loader, node):
            return wrapper(node.value)

        yaml.add_constructor(f"!{new_tag}", yaml_fnc, Loader=yaml.FullLoader)
        yaml.add_constructor(f"!{new_tag}", yaml_fnc, Loader=yaml.Loader)

        return wrapper

    return inner


def get_loader(name) -> Callable:
    """Get a data loader."""
    if name not in _DATA_LOADERS:
        for fnc in _DATA_LOADERS.values():
            if fnc.tag == name:
                return fnc

    return _DATA_LOADERS[name]


@data_loader("pkl")
def pickle_loader(data):
    """A function that loads pickle data."""
    with open(data, "rb") as f:
        return pickle.load(f)


@data_loader()
def npz_loader(data):
    """A function that loads npz data."""
    return dict(np.load(data))


@data_loader()
def npy_loader(data):
    """A function that loads npy data."""
    return np.load(data)


@data_loader("load")
def composite_loader(data):
    """A function that tries different loaders for a given data file until success."""
    for name, loader in _DATA_LOADERS.items():
        if name == "composite_loader":
            continue
        try:
            return loader(data)
        except LoadError:
            pass

    raise LoadError(f"None of the specified loaders were able to load the data: {data}")
