from chi2_components import LinearComponent, Chi2, run
import numpy as np
from scipy import stats

if __name__ == "__main__":
    import sys
    from yabf import Param, LikelihoodContainer

    sampler = sys.argv[1]

    x = np.linspace(0, 10, 100)
    x_hi = x + 15

    lk_lo = Chi2(
        name="chi2_lo",
        x = x,
        components=(
            LinearComponent(
                x=x,
                params=(
                    Param('p1_lo', fiducial=3, min=-10, max=10, alias_for="p1", ref=stats.norm(3, 1)),
                )
            ),
        ),
    )

    lk_hi = Chi2(
        name="chi2_hi",
        x = x_hi,
        components=(
            LinearComponent(
                x=x,
                params=(
                    Param('p1_hi', fiducial=4, min=-10, max=10, alias_for="p1", ref=stats.norm(4, 1)),

                )
            ),
        ),
    )

    lk = LikelihoodContainer(
        likelihoods=(lk_lo, lk_hi),
        params=(
            Param('p0', fiducial=-5, min=-20, max=20, ref=stats.norm(-5, 1)),
        )
    )

    run(sampler, lk, "_double_linear")