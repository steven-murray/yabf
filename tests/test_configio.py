from yabf import LikelihoodContainer, load_likelihood_from_yaml
import yaml


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
    assert "small.shared" in sum(list(lk.common_components.values()), [])

    out = yaml.dump(lk)
    lk2 = yaml.load(out, Loader=yaml.FullLoader)

    assert lk == lk2
