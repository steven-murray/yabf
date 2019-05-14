import collections
import importlib
import sys
from os import path

from scipy import stats

from . import Param, LikelihoodContainer, Likelihood, Component, DataLoader, CompositeLoader, Sampler
from . import yaml


# from yamlinclude import YamlIncludeConstructor

# YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)


def _absfile(yml, fname):
    if path.isabs(fname):
        return fname
    else:
        return path.join(path.dirname(path.abspath(yml)), fname)


def _ensure_float(dct, name):
    if name in dct:
        dct[name] = float(dct[name])


def _construct_params(dct):
    params = dct.pop("params", {})

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


def _construct_data(dct, key="kwargs"):
    if key not in ['kwargs', 'data']:
        raise ValueError("key must be 'kwargs' or 'data'")

    loader = DataLoader._plugins[dct.pop("data_loader", "CompositeLoader")]

    if key == 'data' and key not in dct:
        return None

    data_dct = dct.get(key, {})

    if type(data_dct) is dict:
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


def _construct_derived(dct):
    return dct.pop("derived", [])


def _construct_fiducial(dct):
    fiducial = dct.pop("fiducial", {})
    return fiducial


def _construct_components(dct):
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

        cmp_data = _construct_data(c)
        params = _construct_params(c)
        derived = _construct_derived(c)
        fiducial = _construct_fiducial(c)

        components.append(cls(name=c.get("name", None), params=params, derived=derived, fiducial=fiducial, **cmp_data))

    return components


def _construct_likelihoods(config):
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

        data = _construct_data(lk, key="data")
        kwargs = _construct_data(lk)
        params = _construct_params(lk)
        derived = _construct_derived(lk)
        fiducial = _construct_fiducial(lk)
        components = _construct_components(lk)

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


def _load_str_or_file(stream):
    try:
        st = open(stream)
        stream = st.read()
        st.close()
        file_not_found = False
    except FileNotFoundError:
        file_not_found = True

    try:
        return yaml.load(stream)
    except Exception as e:
        if file_not_found:
            msg = """
            If you passed a filename, it does not exist. Otherwise, the stream passed
            has invalid syntax. Passed:
            
            {} 
            """.format(stream)
        else:
            msg = """YML file passed has invalid syntax for yabf. {}""".format(e)

        raise Exception("Could not load yabf YML. {}".format(msg))


def _recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = _recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def load_likelihood_from_yaml(stream, name=None, override=None):
    config = _load_str_or_file(stream)

    if override:
        config = _recursive_update(config, override)

    # First, check if the thing just loaded in fine (i.e. it was written by YAML
    # on the object itself).
    if isinstance(config, Likelihood):
        return config

    _import_plugins(config)

    # Load outer components
    name = config.get("name", name)
    components = _construct_components(config)
    derived = _construct_derived(config)
    fiducial = _construct_fiducial(config)
    params = _construct_params(config)
    likelihoods = _construct_likelihoods(config)

    if any([len(components), len(derived), len(fiducial), len(params)]) or len(likelihoods) > 1:
        if name is None:
            name = " ".join([lk.name for lk in likelihoods])

        # If any of the external components are non-empty, we need to build a container
        return LikelihoodContainer(
            name=name, components=components, derived=derived, fiducial=fiducial, params=params, likelihoods=likelihoods
        )
    else:
        # Otherwise just return the only likelihood, which is self-contained.
        return likelihoods[0]


def _construct_sampler(config, likelihood):
    sampler = Sampler._plugins[config.pop("sampler")]
    init = config.pop("init", {})
    runkw = config.pop("sample", {})

    return sampler(likelihood=likelihood, sampler_kwargs=init, **config), runkw


def load_from_yaml(stream, name=None, override=None):
    config = _load_str_or_file(stream)
    if override:
        config = _recursive_update(config, override)

    _import_plugins(config)

    if type(config.get("likelihoods")) is dict:
        likelihood = load_likelihood_from_yaml(stream, name)
    else:
        likelihood = load_likelihood_from_yaml(config.get("likelihoods"))

    return _construct_sampler(config, likelihood)


def load_sampler_from_yaml(stream, likelihood, override=None):
    """
    Return a sampler and any sampling arguments specified in the yaml file
    """
    config = _load_str_or_file(stream)
    if override:
        config = _recursive_update(config, override)

    return _construct_sampler(config, likelihood)
