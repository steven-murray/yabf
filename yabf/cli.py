# -*- coding: utf-8 -*-

"""Console script for yabf."""

import click
import sys
from getdist import plots
from os import path
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from yabf.core import utils
from yabf.core.likelihood import _LikelihoodInterface

from . import load_from_yaml, load_likelihood_from_yaml, load_sampler_from_yaml, mpi

try:
    from matplotlib import pyplot as plt

    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

console = Console(width=100)


@click.command()
@click.argument("yaml_file", type=click.Path(exists=True, dir_okay=False))
@click.option("-p/-P", "--plot/--no-plot", default=True)
@click.option(
    "-s", "--sampler-file", default=None, type=click.Path(exists=True, dir_okay=False)
)
@click.option("-w/-W", "--write/--no-write", default=True)
@click.option("--prefix", default=None)
@click.option(
    "-f",
    "--plot-format",
    default="pdf",
    type=click.Choice(["pdf", "png"], case_sensitive=False),
)
def main(yaml_file, plot, sampler_file, write, prefix, plot_format):
    """Console script for yabf."""
    if mpi.am_single_or_primary_process:
        console.print(
            Panel("Welcome to yabf!", box=box.DOUBLE_EDGE),
            style="bold",
            justify="center",
        )

    # Make the file prefix equivalent to the YAML file, unless over-ridden.
    if prefix is None:
        prefix = path.splitext(path.basename(yaml_file))[0]

    if sampler_file is not None:
        likelihood = load_likelihood_from_yaml(yaml_file)
        sampler, runkw = load_sampler_from_yaml(
            sampler_file, likelihood, override={"output_prefix": prefix}
        )
    else:
        sampler, runkw = load_from_yaml(yaml_file, override={"output_prefix": prefix})

    if mpi.am_single_or_primary_process:
        console.print(Rule(f"Sampler [{sampler.__class__.__name__}] "))
        console.print(f"[bold]Sampler Options:[/] {sampler.sampler_kwargs}")
        console.print(f"[bold]Run Options:[/] {runkw}")
        console.print(f"[bold]Output Directory:[/]\t{sampler.output_dir}")
        console.print(f"[bold]Output Prefix:[/]\t{sampler.output_file_prefix}")

        console.print()
        console.print(Rule("Model"))

        console.print("[bold]Likelihoods[/]")
        for lk in likelihood._subcomponents:
            if isinstance(lk, _LikelihoodInterface):
                console.print(lk.name)
        console.print()

        console.print("[bold]Components[/]")
        for loc, cmp in likelihood.child_components.items():
            if not isinstance(cmp, _LikelihoodInterface):
                console.print(loc)
        console.print()

        console.print(
            f"[bold]Active Parameters[/] [blue]({len(likelihood.child_active_params)})[/] "
        )
        _len = max(len(p.name) for p in likelihood.child_active_params)
        _dlen = max(len(str(p.determines)) for p in likelihood.child_active_params)
        for p in likelihood.child_active_params:
            det = str(p.determines).replace("'", "").replace("(", "").replace(")", "")
            fid = "[" + str(p.fiducial) + "]"
            console.print(f"{p.name:<{_len}}  {fid:<8} -----> {det:<{_dlen}}")

        fit_params = sum((p.determines for p in likelihood.child_active_params), ())
        console.print()
        console.print("[bold]In-active parameters[/]")
        for lc in likelihood.child_base_parameter_dct:
            if lc not in fit_params:
                console.print(
                    f"{lc} = {utils.get_loc_from_dict(likelihood.fiducial_params, lc)}"
                )

        console.print()
        console.print(Rule("Starting MCMC"))

    mpi.sync_processes()
    mcsamples = sampler.sample(**runkw)
    mpi.sync_processes()

    if mpi.am_single_or_primary_process:
        console.print("Done.\n")
        console.print(Rule())
        console.print()
        console.print(Rule("[bold]Basic Chain Diagnostics[/]"))

        try:
            gr = mcsamples.getGelmanRubin()
        except Exception:
            gr = "unavailable"

        console.print("Gelman-Rubin Statistic: ", gr)
        console.print()
        console.print(Rule("Correlation Lengths"))
        for i, p in enumerate(mcsamples.getParamNames().names):
            corrlength = mcsamples.getCorrelationLength(i, weight_units=False)
            console.print("{}:\t{:1.3e}".format(p.name, corrlength))

        console.print()
        console.print("Mean (+- std) Posterior Values:")
        mean = mcsamples.getMeans()
        std = mcsamples.getVars()

        for m, s, p in zip(mean, std, mcsamples.getParamNames().names):
            console.print(
                "{p}:\t{mean:1.3e} +- {std:1.3e}".format(p=p.name, mean=m, std=s)
            )

        if plot and HAVE_MPL:
            g = plots.getSubplotPlotter()
            g.triangle_plot(
                mcsamples, params=list(likelihood.child_active_params), shaded=True
            )

            plt.savefig(f"{prefix}_corner.{plot_format}")

        if write:
            mcsamples.saveAsText(prefix, make_dirs="/" in prefix)

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
