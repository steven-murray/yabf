import numpy as np
from chi2_components import Chi2, run, Poly
from scipy import stats
import matplotlib.pyplot as plt
from yabf import mpi

if __name__ == "__main__":
    import sys
    from yabf import Param, LikelihoodContainer

    sampler = sys.argv[1]

    x = np.linspace(0, 10, 100)
    x_hi = x + 15

    lk_lo = Chi2(
        name='chi2_lo',
        data_seed=124,
        x=x,
        components=(
            Poly(
                n=4,
                x=x,
                params=(
                    Param('p0_lo', fiducial=4, min=-10, max=10, ref=stats.norm(4, 1), alias_for='p0'),

                )
            ),
        ),
    )

    lk_hi = Chi2(
        name='chi2_hi',
        data_seed=1234,
        x=x,
        components=(
            Poly(
                n=4,
                x=x,
                params=(
                    Param('p0_hi', fiducial=4, min=-10, max=10, ref=stats.norm(4, 1), alias_for='p0'),

                )
            ),
        ),
    )

    lk = LikelihoodContainer(
        likelihoods=(lk_lo, lk_hi),
        params=(
            Param('p1', fiducial=3, min=-10, max=10, ref=stats.norm(3, 1)),
            Param('p2', fiducial=2, min=-10, max=10, ref=stats.norm(2, 1)),
            Param('p3', fiducial=1, min=-10, max=10, ref=stats.norm(1, 1)),
        )
    )

    mean = run(sampler, lk, "_double_poly4")

    if mpi.am_single_or_primary_process:
        fig, ax = plt.subplots(1, 2, sharex=True, gridspec_kw={"hspace": 0.05}, figsize=(15, 10))

        ax[0].plot(lk_lo.x, lk_lo.data, color="C0", label="Data")
        ax[0].plot(lk_hi.x, lk_hi.data, color="C0")

        fid_model = lk._reduce_all(**lk.fiducial_params)
        ax[0].plot(lk_lo.x, fid_model['chi2_lo'], color="C1", label="True Params")
        ax[0].plot(lk_hi.x, fid_model['chi2_hi'], color="C1")

        ax[1].plot(lk_lo.x, lk_lo.data - fid_model['chi2_lo'], color="C1")
        ax[1].plot(lk_hi.x, lk_hi.data - fid_model['chi2_hi'], color="C1")

        meanpar = lk._fill_params(lk._parameter_list_to_dict(mean))

        mean_model = lk._reduce_all(**meanpar)
        ax[0].plot(lk_lo.x, mean_model['chi2_lo'], color="C2", label="True Params")
        ax[0].plot(lk_hi.x, mean_model['chi2_hi'], color="C2")

        ax[1].plot(lk_lo.x, lk_lo.data - mean_model['chi2_lo'], color="C2")
        ax[1].plot(lk_hi.x, lk_hi.data - mean_model['chi2_hi'], color="C2")

        ax[0].legend()

        plt.savefig(f"double_poly4_{sampler}_residual.pdf")
