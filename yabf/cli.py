# -*- coding: utf-8 -*-

"""Console script for yabf."""

import sys
import click
from . import load_sampler_from_yaml, load_likelihood_from_yaml, load_from_yaml

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
@click.option('-s', '--sampler-file', default=None,
              type=click.Path(exists=True, dir_okay=False))
@click.option("-w/-W", '--write/--no-write', default=True)
@click.option("--prefix", default=None)
@click.option('-f', '--plot-format', default='pdf',
              type=click.Choice(['pdf', 'png'], case_sensitive=False))
def main(yaml_file, plot, sampler_file, write, prefix, plot_format):
    """Console script for yabf."""

    if sampler_file is not None:
        likelihood = load_likelihood_from_yaml(yaml_file)
        sampler, runkw = load_sampler_from_yaml(sampler_file, likelihood)
    else:
        sampler, runkw = load_from_yaml(yaml_file)

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
        g.triangle_plot(mcsamples, params=list(likelihood.child_active_params),shaded=True)

        if prefix is None:
            prefix = path.splitext(path.basename(yaml_file))[0]
        plt.savefig(prefix+"_corner.{}".format(plot_format))

    if write:
        mcsamples.saveAsText(prefix, make_dirs=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
