from chi2_components import LinearComponent, Chi2, run, Poly
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import sys
    from yabf import Param

    sampler = sys.argv[1]

    x = np.linspace(0, 10, 100)

    np.random.seed(1234)

    lk = Chi2(
        x = x,
        components=(
            Poly(
                n=4,
                x=x,
                params=(
                    Param('p0', fiducial=4, min=-10, max=10, ref=stats.norm(4, 1)),
                    Param('p1', fiducial=3, min=-10, max=10, ref=stats.norm(3, 1)),
                    Param('p2', fiducial=2, min=-10, max=10, ref=stats.norm(2, 1)),
                    Param('p3', fiducial=1, min=-10, max=10, ref=stats.norm(1, 1)),
                )
            ),
        ),
    )

    mean = run(sampler, lk, "_single_poly4")

    fig, ax = plt.subplots(1, 2, sharex=True, gridspec_kw={"hspace": 0.05}, figsize=(15, 10))

    ax[0].plot(lk.x, lk.data, label="Data")
    ax[0].plot(lk.x, lk.reduce_model(**lk.fiducial_params), label="True Params")
    ax[1].plot(lk.x, lk.data - lk.reduce_model(**lk.fiducial_params))

    meanpar = lk._fill_params(lk._parameter_list_to_dict(mean))

    ax[0].plot(lk.x, lk.reduce_model(**meanpar), label="Mean Params")
    ax[1].plot(lk.x, lk.data - lk.reduce_model(**meanpar), label="Mean Params")
    ax[0].legend()

    plt.savefig(f"single_poly4_{sampler}_residual.pdf")