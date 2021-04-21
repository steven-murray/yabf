"""Module defining routines for reading/writing config files."""
import importlib
import sys
from os import path
from scipy import stats

from . import utils, yaml
from .component import Component
from .io import CompositeLoader, DataLoader
from .likelihood import Likelihood, LikelihoodContainer, _LikelihoodInterface
from .parameters import Param
from .samplers import Sampler


def _absfile(yml, fname):
    if path.isabs(fname):
        return fname
    else:
        return path.join(path.dirname(path.abspath(yml)), fname)


def _ensure_float(dct, name):
    if name in dct:
        dct[name] = float(dct[name])


def _construct_dist(dct):
    _ensure_float(dct, "loc")
    _ensure_float(dct, "scale")

    return getattr(stats, dct.pop("dist"))(**dct)


def _construct_params(dct):
    params = dct.pop("params", {})

    parameters = []
    for pname, p in params.items():
        _ensure_float(p, "min")
        _ensure_float(p, "max")

        # The ref value needs to be made into a scipy.stats object
        ref = p.pop("ref", None)
        if ref:
            ref = _construct_dist(ref)

        prior = p.pop("prior", None)
        if prior:
            prior = _construct_dist(prior)

        pmaps = p.pop("parameter_mappings", None)
        if pmaps:
            pmaps = [eval("lambda x: {}".format(pmap)) for pmap in pmaps]

        parameters.append(Param(pname, prior=prior, ref=ref, transforms=pmaps, **p))

    return parameters


def _construct_data(dct, key="kwargs"):
    if key not in ["kwargs", "data"]:
        raise ValueError("key must be 'kwargs' or 'data'")

    loader = DataLoader._plugins[dct.pop("data_loader", "CompositeLoader")]

    if key == "data" and key not in dct:
        return None

    data_dct = dct.get(key, {})

    if not isinstance(data_dct, dict):
        return loader().load(data_dct)

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


def _construct_derived(dct):
    return dct.pop("derived", [])


def _construct_fiducial(dct):
    return dct.pop("fiducial", {})


def _construct_components(dct):
    comp = dct.get("components", {})
    components = []

    for name, cmp in comp.items():
        try:
            cls = cmp["class"]
        except KeyError:
            raise KeyError(
                "Every component requires a key:val pair of class: class_name"
            )

        try:
            cls = Component._plugins[cls]
        except KeyError:
            raise ImportError(
                f"The component '{name}' is not importable. Ensure you "
                "have set the correct import_paths and external_modules"
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
                **cmp_data,
            )
        )

    return components


def _construct_likelihoods(config, ignore_data=False):
    lks = config.get("likelihoods")
    likelihoods = []

    for name, lk in lks.items():
        try:
            likelihood = lk["class"]
        except KeyError:
            raise KeyError(
                "Every likelihood requires a key:val pair of class: class_name"
            )

        try:
            cls = Likelihood._plugins[likelihood]
        except KeyError:
            raise ImportError(
                f"The likelihood '{name}' is not importable. Ensure you "
                "have set the correct import_paths and external_modules"
            )

        data = _construct_data(lk, key="data") if not ignore_data else None
        kwargs = _construct_data(lk)
        params = _construct_params(lk)
        derived = _construct_derived(lk)
        fiducial = _construct_fiducial(lk)
        components = _construct_components(lk)
        data_seed = config.get("data_seed", None)

        likelihoods.append(
            cls(
                name=name,
                params=params,
                derived=derived,
                fiducial=fiducial,
                data=data,
                data_seed=data_seed,
                components=components,
                **kwargs,
            )
        )

    return likelihoods


def _import_plugins(config):
    # First set import paths and import libraries
    paths = config.get("import_paths", [])
    for pth in paths:
        sys.path.append(pth)

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
            msg = f"""
            If you passed a filename, it does not exist. Otherwise, the stream passed
            has invalid syntax. Passed

            {stream}
            """
        elif stream_probably_yamlcode:
            msg = """
            YAML code passed has invalid syntax for yabf.
            """
        else:
            msg = f"""YML file passed has invalid syntax for yabf. {e}"""

        raise Exception(f"Could not load yabf YML. {msg}")


def load_likelihood_from_yaml(stream, name=None, override=None, ignore_data=False):
    config = _load_str_or_file(stream)

    if override:
        config = utils.recursive_update(config, override)

    # First, check if the thing just loaded in fine (i.e. it was written by YAML
    # on the object itself).
    if isinstance(config, _LikelihoodInterface):
        return config

    _import_plugins(config)

    # Load outer components
    name = config.get("name", name)

    likelihoods = _construct_likelihoods(config, ignore_data=ignore_data)

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


def load_from_yaml(stream, name=None, override=None, ignore_data=False):
    config = _load_str_or_file(stream)
    if override:
        config = utils.recursive_update(config, override)

    _import_plugins(config)

    if type(config.get("likelihoods")) is dict:
        likelihood = load_likelihood_from_yaml(stream, name, ignore_data=ignore_data)
    else:
        likelihood = load_likelihood_from_yaml(
            config.get("likelihoods"), ignore_data=ignore_data
        )

    return _construct_sampler(config, likelihood)


def load_sampler_from_yaml(stream, likelihood, override=None):
    """Return a sampler and any sampling arguments specified in the yaml file."""
    config = _load_str_or_file(stream)
    if override:
        config = utils.recursive_update(config, override)

    return _construct_sampler(config, likelihood)
