"""Various utility functions."""
import collections
import numpy as np
from contextlib import contextmanager
from frozendict import frozendict


def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_loc_from_dict(dct, loc):
    """
    Take a string loc and return a sub-dict corresponding to that loc.

    i.e. "foo.bar" would return dict['foo']['bar']

    empty string returns top-level dict.
    """
    if loc == "":
        return dict
    else:
        locs = loc.split(".")
        d = dct
        for ll in locs:
            try:
                d = d[ll]
            except KeyError:
                raise KeyError(f"loc {loc} does not exist in dict {dct}")

        return d


def add_loc_to_dict(dct, loc, val, raise_if_not_exist=False):
    locs = loc.split(".")
    imax = len(locs) - 1

    this = dct
    for i, loc in enumerate(locs):
        if i == imax:
            if isinstance(val, collections.abc.Mapping):
                this[loc] = recursive_update(this[loc], val)
            else:
                this[loc] = val

        else:
            if loc not in this:
                if raise_if_not_exist:
                    raise KeyError(f"{loc} not in dict")

                this[loc] = {}
            this = this[loc]
    return dct


@contextmanager
def seed_as(seed):
    """A context manager which sets a random seed and then randomizes upon exit."""
    post_seed = np.random.randint(0, 2 ** 32 - 1)
    np.random.seed(seed)
    yield None
    np.random.seed(post_seed)


def recursive_frozendict(dct):
    for k, v in dct.items():
        if isinstance(v, collections.abc.Mapping):
            dct[k] = recursive_frozendict(v)
    return frozendict(dct)
