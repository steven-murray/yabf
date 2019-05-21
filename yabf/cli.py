# -*- coding: utf-8 -*-

"""Console script for yabf."""

import sys
import click
from . import load_sampler_from_yaml, load_likelihood_from_yaml, load_from_yaml, mpi
import shutil
from yabf.core import utils

from os import path
from getdist import plots
try:
    from matplotlib import pyplot as plt
    HAVE_MPL = True
except:
    HAVE_MPL = False

YABF = r"""
 .----------------.  .----------------.  .----------------.  .----------------. 
| .--------------. || .--------------. || .--------------. || .--------------. |
| |  ____  ____  | || |      __      | || |   ______     | || |  _________   | |
| | |_  _||_  _| | || |     /  \     | || |  |_   _ \    | || | |_   ___  |  | |
| |   \ \  / /   | || |    / /\ \    | || |    | |_) |   | || |   | |_  \_|  | |
| |    \ \/ /    | || |   / ____ \   | || |    |  __'.   | || |   |  _|      | |
| |    _|  |_    | || | _/ /    \ \_ | || |   _| |__) |  | || |  _| |_       | |
| |   |______|   | || ||____|  |____|| || |  |_______/   | || | |_____|      | |
| |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------' 
"""


@click.command()
@click.argument("yaml_file", type=click.Path(exists=True, dir_okay=False))
@click.option('-p/-P', '--plot/--no-plot', default=True)
@click.option('-s', '--sampler-file', default=None,
              type=click.Path(exists=True, dir_okay=False))
@click.option("-w/-W", '--write/--no-write', default=True)
@click.option("--prefix", default=None)
@click.option('-f', '--plot-format', default='pdf',
              type=click.Choice(['pdf', 'png'], case_sensitive=False))
def main(yaml_file, plot, sampler_file, write, prefix, plot_format):
    """Console script for yabf."""

    terminal_width = shutil.get_terminal_size((80, 20))[0]
    if mpi.am_single_or_primary_process:
        print("="*terminal_width)
        print(YABF)
        print("-"*terminal_width)
        print()

    # Make the file prefix equivalent to the YAML file, unless over-ridden.
    if prefix is None:
        prefix = path.splitext(path.basename(yaml_file))[0]

    if sampler_file is not None:
        likelihood = load_likelihood_from_yaml(yaml_file)
        sampler, runkw = load_sampler_from_yaml(sampler_file, likelihood, override={"output_prefix": prefix})
    else:
        sampler, runkw = load_from_yaml(yaml_file, override={"output_prefix": prefix})

    if mpi.am_single_or_primary_process:
        print(f"Sampler: [{sampler.__class__.__name__}]")
        print(f"\t{sampler.sampler_kwargs}")
        print(f"\t{runkw}")
        print(f"\tOutput Directory:\t{sampler.output_dir}")
        print(f"\tOutput Prefix:\t{sampler.output_file_prefix}")


        print()
        print("Model:")
        print(f"\tComponents.Params: ")
        for lc in likelihood.child_base_parameter_dct:
            print(f"\t\t{lc}\t{utils.get_loc_from_dict(likelihood.fiducial_params, lc)}")

        print(f"\tFiducial Parameters: {likelihood.fiducial_params}")
        print()

        print("-"*terminal_width)
        print("Starting:\n")

    mpi.sync_processes()
    mcsamples = sampler.sample(**runkw)
    mpi.sync_processes()

    if mpi.am_single_or_primary_process:
        print("Done.\n")
        print("-"*terminal_width)
        print("Basic Chain Diagnostics")
        print("-"*terminal_width)

        try:
            gr = mcsamples.getGelmanRubin()
        except:
            gr = "unavailable"

        print("Gelman-Rubin Statistic: ", gr)
        print()
        print("Correlation Lengths")
        print("-"*terminal_width)
        for i, p in enumerate(mcsamples.getParamNames().names):
            corrlength = mcsamples.getCorrelationLength(i, weight_units=False)
            print("{}:\t{:1.3e}".format(p.name, corrlength))

        print()
        print("-"*terminal_width)
        print("Mean (+- std) Posterior Values:")
        print('-'*terminal_width)
        mean = mcsamples.getMeans()
        std = mcsamples.getVars()

        for m,s,p in zip(mean, std, mcsamples.getParamNames().names):
            print("{p}:\t{mean:1.3e} +- {std:1.3e}".format(p=p.name, mean=m, std=s))

        if plot and HAVE_MPL:
            g = plots.getSubplotPlotter()
            g.triangle_plot(mcsamples, params=list(likelihood.child_active_params),shaded=True)

            plt.savefig(prefix+"_corner.{}".format(plot_format))

        if write:
            mcsamples.saveAsText(prefix, make_dirs='/' in prefix)

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
