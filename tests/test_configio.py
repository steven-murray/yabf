import yaml

from yabf import Likelihood, LikelihoodContainer, load_likelihood_from_yaml


def test_round_trip():
    # subclass
    yml = """
likelihoods:
    - name: small
      class: SimpleLikelihood
      components:
        - name: shared
          class: SimpleComponent
    - name: big
      class: SimpleLikelihood
      components:
        - name: unshared
          class: SimpleComponent
        - name: shared
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
    lk2 = yaml.load(out, Loader=yaml.Loader)

    assert lk == lk2


def test_paramvec():
    yml = """
likelihoods:
  - name: vec
    class: ParameterVecLikelihood
    params:
      x:
        length: 3
"""
    lk = load_likelihood_from_yaml(yml)
    assert isinstance(lk, Likelihood)
    assert len(lk.child_active_params) == 3
    assert "x_0" in lk.child_active_params
