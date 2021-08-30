"""Module defining routines for reading/writing config files."""
import importlib
import sys
import yaml
from os import path
from pathlib import Path
from scipy import stats
from typing import Tuple

from . import utils
from .component import Component
from .likelihood import Likelihood, LikelihoodContainer, _LikelihoodInterface
from .parameters import Param, ParamVec
from .samplers import Sampler


def _absfile(yml, fname):
    if path.isabs(fname):
        return fname
    else:
        return path.join(path.dirname(path.abspath(yml)), fname)


def _ensure_float(dct, name):
    if name in dct:
        try:
            dct[name] = float(dct[name])
        except TypeError:
            pass


def _construct_dist(dct):
    _ensure_float(dct, "loc")
    _ensure_float(dct, "scale")

    return getattr(stats, dct.pop("dist"))(**dct)


def _construct_params(dct, config_path: Path):
    params = dct.pop("params", {})

    if isinstance(params, list):
        return params
    elif isinstance(params, str):
        params, _ = _read_sub_yaml(params, config_path.parent)

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

        pmaps = p.pop("transforms", None)
        if pmaps:
            pmaps = [eval(f"lambda x: {pmap}") for pmap in pmaps]

        if "length" in p:
            parameters.extend(
                list(ParamVec(pname, prior=prior, ref=ref, **p).get_params())
            )
        else:
            parameters.append(Param(pname, prior=prior, ref=ref, transforms=pmaps, **p))

    return parameters


def _construct_derived(dct):
    return dct.pop("derived", [])


def _construct_fiducial(dct):
    return dct.pop("fiducial", {})


def _read_sub_yaml(cmp: str, pth: Path) -> Tuple[dict, Path]:
    cmp = Path(cmp)
    if not cmp.exists():
        cmp = pth / cmp
    if not cmp.exists():
        raise OSError(f"Included component/likelihood sub-YAML does not exist: {cmp}")

    with open(cmp) as fl:
        out = yaml.load(fl)

    return out, cmp


def _construct_components(dct, config_path: Path):
    comp = dct.pop("components", [])
    components = []

    for cmp in comp:
        if isinstance(cmp, str):
            cmp, new_path = _read_sub_yaml(cmp, config_path.parent)
        else:
            new_path = config_path

        components.append(_construct_component(cmp, new_path))

    return components


def _construct_component(cmp, new_path):

    try:
        cls = cmp.pop("class")
    except KeyError:
        raise KeyError("Every component requires a key:val pair of class: class_name")

    try:
        cls = Component._plugins[cls]
    except KeyError:
        raise ImportError(
            f"The component '{cmp['name']}' is not importable. Ensure you "
            "have set the correct import_paths and external_modules"
        )

    params = _construct_params(cmp, new_path)
    derived = _construct_derived(cmp)
    fiducial = _construct_fiducial(cmp)
    subcmp = _construct_components(cmp, new_path)
    return cls(
        name=cmp.pop("name"),
        params=params,
        derived=derived,
        fiducial=fiducial,
        components=subcmp,
        **cmp,
    )


def _construct_likelihoods(config, config_path: Path, ignore_data=False):
    lks = config.get("likelihoods")
    likelihoods = []

    for lk in lks:
        # If the user input a path to a YAML file, read it first.
        if isinstance(lk, str):
            lk, new_path = _read_sub_yaml(lk, config_path.parent)
        else:
            new_path = config_path

        likelihoods.append(_construct_likelihood(lk, new_path))

    return likelihoods


def _construct_likelihood(lk: dict, config_path: Path, ignore_data=False):
    try:
        likelihood = lk.pop("class")
    except KeyError:
        raise KeyError("Every likelihood requires a key:val pair of class: class_name")

    try:
        cls = Likelihood._plugins[likelihood]
    except KeyError:
        raise ImportError(
            f"The likelihood '{lk['name']}' is not importable. Ensure you "
            "have set the correct import_paths and external_modules"
        )

    params = _construct_params(lk, config_path)
    derived = _construct_derived(lk)
    fiducial = _construct_fiducial(lk)
    components = _construct_components(lk, config_path)
    data_seed = lk.get("data_seed")

    return cls(
        name=lk.pop("name"),
        params=params,
        derived=derived,
        fiducial=fiducial,
        data_seed=data_seed,
        components=components,
        **lk,
    )


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
        with open(stream) as st:
            stream = st.read()
        file_not_found = False
    except FileNotFoundError:
        file_not_found = True
    except OSError:
        stream_probably_yamlcode = True
        file_not_found = False
    try:
        return yaml.load(stream, Loader=yaml.FullLoader)
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

    likelihoods = _construct_likelihoods(
        config, Path(getattr(stream, "name", stream)), ignore_data=ignore_data
    )

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
