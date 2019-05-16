import sys

import numpy as np
from scipy import stats

from yabf import Param
from .chi2_components import LinearComponent, Chi2, run

if __name__ == "__main__":
    sampler = sys.argv[1]

    x = np.linspace(0, 10, 100)

    lk = Chi2(
        name="chi2",
        x=x,
        components=(
            LinearComponent(
                x=x,
                params=(
                    Param('p0', fiducial=0, min=-10, max=10, ref=stats.norm(0, 1)),
                    Param('p1', fiducial=3, min=-10, max=10, ref=stats.norm(3, 1)),
                )
            ),
        ),
    )

    run(sampler, lk)
