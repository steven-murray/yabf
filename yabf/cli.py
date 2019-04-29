# -*- coding: utf-8 -*-

"""Console script for yabf."""

import sys
import click
from . import load_sampler_from_yaml

from os import path
from getdist import plots
try:
    from matplotlib import pyplot as plt
    HAVE_MPL = True
except:
    HAVE_MPL = False

@click.command()
@click.argument("yaml_file", type=click.Path(exists=True, dir_okay=False))
@click.option('--plot/--no-plot', default=True)
def main(yaml_file, plot):
    """Console script for edges_estimate."""
    sampler, runkw = load_sampler_from_yaml(yaml_file)

    mcsamples = sampler.sample(**runkw)

    print("-------------------------------")
    print("Basic Chain Diagnostics")
    print("-------------------------------")

    try:
        gr = mcsamples.getGelmanRubin()
    except:
        gr = "unavailable"

    print("Gelman-Rubin Statistic: ", gr)
    print()
    print("Correlation Lengths")
    print("-------------------")
    for i, p in enumerate(mcsamples.getParamNames().names):
        corrlength = mcsamples.getCorrelationLength(i, weight_units=False)
        print("{}:\t{:1.3e}".format(p.name, corrlength))

    print()
    print("-------------------------------")
    print("Mean (+- std) Posterior Values:")
    print('-------------------------------')
    mean = mcsamples.getMeans()
    std = mcsamples.getVars()

    for m,s,p in zip(mean, std, mcsamples.getParamNames().names):
        print("{p}:\t{mean:1.3e} +- {std:1.3e}".format(p=p.name, mean=m, std=s))

    if plot and HAVE_MPL:
        g = plots.getSubplotPlotter()
        g.triangle_plot(mcsamples, shaded=True)
        plt.savefig(path.join("corner.pdf"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
