import importlib
import sys
from os import path

import yaml
from scipy import stats

from . import Param, LikelihoodContainer, Likelihood, Component, DataLoader, CompositeLoader, Sampler


def _absfile(yml, fname):
    if path.isabs(fname):
        return fname
    else:
        return path.join(path.dirname(path.abspath(yml)), fname)


def _get_included(fname, dct, key="include"):
    # Add any included parameters
    inc = dct.pop(key, None)
    if inc:
        with open(_absfile(fname, inc), 'rb') as f:
            dct.update(yaml.load(f))


def _ensure_float(dct, name):
    if name in dct:
        dct[name] = float(dct[name])


def _construct_params(fname, dct):
    params = dct.pop("params", {})

    _get_included(fname, params)

    parameters = []
    for pname, p in params.items():
        _ensure_float(p, 'min')
        _ensure_float(p, 'max')

        # The ref value needs to be made into a scipy.stats object
        ref = p.pop("ref", None)
        if ref:
            _ensure_float(ref, "loc")
            _ensure_float(ref, "scale")

            ref = getattr(stats, ref.pop('dist'))(**ref)

        parameters.append(Param(pname, ref=ref, **p))

    return parameters


def _construct_data(fname, dct, key="kwargs"):
    if key not in ['kwargs', 'data']:
        raise ValueError("key must be 'kwargs' or 'data'")

    loader = DataLoader._plugins[dct.pop("data_loader", "CompositeLoader")]

    data_dct = dct.get(key, {})

    if type(data_dct) is dict:
        _get_included(fname, data_dct)

        # Load data
        if loader is CompositeLoader:
            loader = DataLoader._plugins[data_dct.pop("data_loader", "CompositeLoader")]

        data = {}
        for key, val in data_dct.items():
            if key.startswith("dummy"):
                data.update(loader().load(val))
            else:
                data.update({key: loader().load(val)})
        return data
    else:
        out = loader().load(data_dct)
        return out


def _construct_derived(fname, dct):
    return dct.pop("derived", [])


def _construct_fiducial(fname, dct):
    fiducial = dct.pop("fiducial", {})
    _get_included(fname, fiducial)
    return fiducial


def _construct_components(fname, dct):
    comp = dct.get("components", {})
    components = []

    for name, c in comp.items():
        try:
            cls = Component._plugins[name]
        except KeyError:
            raise ImportError(
                "The component {} is not importable. Ensure you "
                "have set the correct import_paths and external_modules".format(name)
            )

        cmp_data = _construct_data(fname, c)
        params = _construct_params(fname, c)
        derived = _construct_derived(fname, c)
        fiducial = _construct_fiducial(fname, c)

        components.append(cls(name=c.get("name", None), params=params, derived=derived, fiducial=fiducial, **cmp_data))

    return components


def _construct_likelihoods(fname, config):
    lks = config.get("likelihoods")
    likelihoods = []

    for name, lk in lks.items():
        try:
            likelihood = lk['likelihood']
        except KeyError:
            raise KeyError("Every likelihood requires a key:val pair of likelihood: class_name")

        try:
            cls = Likelihood._plugins[likelihood]
        except KeyError:
            raise ImportError(
                "The likelihood {} is not importable. Ensure you "
                "have set the correct import_paths and external_modules".format(name)
            )

        data = _construct_data(fname, lk, key="data")
        kwargs = _construct_data(fname, lk)
        params = _construct_params(fname, lk)
        derived = _construct_derived(fname, lk)
        fiducial = _construct_fiducial(fname, lk)
        components = _construct_components(fname, lk)

        if not data:  # need to convert from dict to None if no data passed.
            data = None

        likelihoods.append(
            cls(name=name,
                params=params, derived=derived, fiducial=fiducial, data=data,
                components=components, **kwargs))

    return likelihoods


def _import_plugins(config):
    # First set import paths and import libraries
    paths = config.get("import_paths", [])
    for path in paths:
        sys.path.append(path)

    modules = config.get("external_modules", [])
    for module in modules:
        importlib.import_module(module)


def load_likelihood_from_yaml(fname):
    with open(fname) as f:
        config = yaml.load(f)

    _import_plugins(config)

    # Load outer components
    name = config.get("name", path.splitext(path.basename(fname))[0])
    components = _construct_components(fname, config)
    derived = _construct_derived(fname, config)
    fiducial = _construct_fiducial(fname, config)
    params = _construct_fiducial(fname, config)
    likelihoods = _construct_likelihoods(fname, config)

    if any([len(components), len(derived), len(fiducial), len(params)]) or len(likelihoods) > 1:
        # If any of the external components are non-empty, we need to build a container
        return LikelihoodContainer(
            name=name, components=components, derived=derived, fiducial=fiducial, params=params, likelihoods=likelihoods
        )
    else:
        # Otherwise just return the only likelihood, which is self-contained.
        return likelihoods[0]


def load_sampler_from_yaml(fname):
    """
    Return a sampler and any sampling arguments specified in the yaml file
    """
    with open(fname) as f:
        config = yaml.load(f)

    _import_plugins(config)

    if type(config.get("likelihoods")) is dict:
        likelihood = load_likelihood_from_yaml(fname)
    else:
        likelihood = load_likelihood_from_yaml(config.get("likelihoods"))

    if type(config.get("sampler")) is not dict:
        _get_included(fname, config, "sampler")

    sampler = Sampler._plugins[config.get("sampler")]
    init = config.get("init", {})
    runkw = config.get("sample", {})

    return sampler(likelihood=likelihood, sampler_kwargs=init), runkw
