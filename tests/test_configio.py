import yaml

from yabf import Likelihood, LikelihoodContainer, load_likelihood_from_yaml


def test_round_trip():
    # subclass
    yml = """
likelihoods:
    small:
        class: SimpleLikelihood
        components:
            shared:
                class: SimpleComponent
    big:
        class: SimpleLikelihood
        components:
            unshared:
                class: SimpleComponent
            shared:
                class: SimpleComponent
    """

    lk = load_likelihood_from_yaml(yml)

    assert isinstance(lk, LikelihoodContainer)

    assert "small" in lk._subcomponent_names
    assert "big" in lk._subcomponent_names
    print(list(lk.child_components.keys()))
    print(lk.likelihoods[0]._subcomponents)
    assert "small.shared" in lk.child_components
    assert "big.unshared" in lk.child_components
    print(lk.common_components)
    assert "small.shared" in lk.common_components[0][1]

    out = yaml.dump(lk)
    lk2 = yaml.load(out, Loader=yaml.FullLoader)

    assert lk == lk2


def test_paramvec():
    yml = """
likelihoods:
    vec:
        class: ParameterVecLikelihood
        params:
          x:
            length: 3
"""
    lk = load_likelihood_from_yaml(yml)
    assert isinstance(lk, Likelihood)
    assert len(lk.child_active_params) == 3
    assert "x_0" in lk.child_active_param_dct
