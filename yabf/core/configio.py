import importlib
import sys
from os import path

from scipy import stats

from yabf import Param, LikelihoodContainer, Likelihood, Component
from yabf.core import yaml
from yabf.core.likelihood import LikelihoodInterface
from . import utils
from .io import DataLoader, CompositeLoader
from .samplers import Sampler


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
    return dct.pop("fiducial", {})


def _construct_components(dct):
    comp = dct.get("components", {})
    components = []

    for name, cmp in comp.items():
        try:
            cls = cmp['class']
        except KeyError:
            raise KeyError("Every component requires a key:val pair of class: class_name")

        try:
            cls = Component._plugins[cls]
        except KeyError:
            raise ImportError(
                "The component {} is not importable. Ensure you "
                "have set the correct import_paths and external_modules".format(name)
            )

        cmp_data = _construct_data(cmp)
        params = _construct_params(cmp)
        derived = _construct_derived(cmp)
        fiducial = _construct_fiducial(cmp)
        subcmp = _construct_components(cmp)

        components.append(
            cls(
                name=name,
                params=params,
                derived=derived,
                fiducial=fiducial,
                components=subcmp,
                **cmp_data
            )
        )

    return components


def _construct_likelihoods(config):
    lks = config.get("likelihoods")
    likelihoods = []

    for name, lk in lks.items():
        try:
            likelihood = lk['class']
        except KeyError:
            raise KeyError("Every likelihood requires a key:val pair of class: class_name")

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
        data_seed = config.get("data_seed", None)

        likelihoods.append(
            cls(name=name, params=params, derived=derived, fiducial=fiducial, data=data,
                data_seed=data_seed, components=components, **kwargs
                )
        )

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
    stream_probably_yamlcode = False

    try:
        st = open(stream)
        stream = st.read()
        st.close()
        file_not_found = False
    except FileNotFoundError:
        file_not_found = True
    except OSError:
        stream_probably_yamlcode = True
        file_not_found = False
    try:
        return yaml.load(stream)
    except Exception as e:
        if file_not_found:
            msg = """
            If you passed a filename, it does not exist. Otherwise, the stream passed
            has invalid syntax. Passed:
            
            {} 
            """.format(stream)
        elif stream_probably_yamlcode:
            msg = """
            YAML code passed has invalid syntax for yabf.
            """
        else:
            msg = """YML file passed has invalid syntax for yabf. {}""".format(e)

        raise Exception("Could not load yabf YML. {}".format(msg))


def load_likelihood_from_yaml(stream, name=None, override=None):
    config = _load_str_or_file(stream)

    if override:
        config = utils.recursive_update(config, override)

    # First, check if the thing just loaded in fine (i.e. it was written by YAML
    # on the object itself).
    if isinstance(config, LikelihoodInterface):
        return config

    _import_plugins(config)

    # Load outer components
    name = config.get("name", name)
    likelihoods = _construct_likelihoods(config)

    if len(likelihoods) > 1:
        # Need to build a container
        return LikelihoodContainer(name=name, likelihoods=likelihoods)
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
        config = utils.recursive_update(config, override)

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
        config = utils.recursive_update(config, override)

    return _construct_sampler(config, likelihood)
