from yabf import yaml, LikelihoodContainer, load_likelihood_from_yaml


def test_round_trip():
    # subclass
    yml = """
likelihoods:
    small:
        likelihood: SimpleLikelihood
    big:
        likelihood: SimpleLikelihood
        components:
            SimpleComponent:
                name: internal
components:
    SimpleComponent:
        name: external    
    """

    lk = load_likelihood_from_yaml(yml)

    assert isinstance(lk, LikelihoodContainer)

    assert "small" in lk._get_subcomponent_names()
    assert "big" in lk._get_subcomponent_names()
    assert "internal" in lk._get_subcomponent_names()
    assert "external" in lk._get_subcomponent_names()

    out = yaml.dump(lk)
    lk2 = yaml.load(out)

    assert lk == lk2
