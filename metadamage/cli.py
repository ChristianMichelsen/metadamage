# Standard Library
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Union

# Third Party
import click
from click import Context
from click_help_colors import HelpColorsCommand, HelpColorsGroup
import typer

# First Party
from metadamage import utils
from metadamage.__version__ import __version__

out_dir_default = Path("./data/out/")
first_time = True


def version_callback(value: bool):
    if value:
        typer.echo(f"Metadamage CLI, version: {__version__}")
        raise typer.Exit()


class CustomHelpColorsCommand(HelpColorsCommand):
    """Colorful command line main help. Colors one of:
    "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white", "reset",
    "bright_black", "bright_red", "bright_green", "bright_yellow",
    "bright_blue", "bright_magenta", "bright_cyan", "bright_white"
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.help_headers_color = "yellow"
        self.help_options_color = "blue"


class CustomHelpColorsGroup(HelpColorsGroup):
    # colorfull command line for subcommands
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.help_headers_color = "yellow"
        self.help_options_color = "blue"


class ColorfulApp(typer.Typer):
    def __init__(self, *args, cls=CustomHelpColorsGroup, **kwargs) -> None:
        super().__init__(*args, cls=cls, **kwargs)

    def command(
        self, *args, cls=CustomHelpColorsCommand, **kwargs
    ) -> typer.Typer.command:
        return super().command(*args, cls=cls, **kwargs)


class OrderedCommands(click.Group):
    def list_commands(self, ctx: Context) -> Iterable[str]:
        return self.commands.keys()


cli_app = ColorfulApp(cls=OrderedCommands)
# cli_app = ColorfulApp(chain=True)


@cli_app.callback()
def callback(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=version_callback
    ),
):
    """
    Metagenomics Ancient Damage: metadamage.

    First run it with the fit command:

    \b
        $ metadamage fit --help

    And subsequently visualize the results using the dashboard:

    \b
        $ metadamage dashboard --help

    """


# https://typer.tiangolo.com/tutorial/parameter-types/path/


@cli_app.command("fit")
def cli_fit(
    # Paths: input filename(s) and output directory
    filenames: List[Path] = typer.Argument(...),
    out_dir: Path = typer.Option(out_dir_default),
    # Fit options
    max_fits: Optional[int] = typer.Option(None, help="[default: None (All fits)]"),
    max_cores: int = 1,
    # Filters
    min_alignments: int = 10,
    min_y_sum: int = 10,
    # Subsitution Bases
    substitution_bases_forward: utils.SubstitutionBases = typer.Option(
        utils.SubstitutionBases.CT
    ),
    substitution_bases_reverse: utils.SubstitutionBases = typer.Option(
        utils.SubstitutionBases.GA
    ),
    # boolean flags
    forced: bool = typer.Option(False, "--forced"),
    # Other
    dask_port: int = 8787,
):
    """Fitting Ancient Damage.

    FILENAME is the name of the file(s) to fit (with the ancient-model)

    run as e.g.:

    \b
        $ metadamage fit --verbose --max-fits 10 --max-cores 2 ./data/input/data_ancient.txt

    or by for two files:

    \b
        $ metadamage fit --verbose --max-fits 10 --max-cores 2 ./data/input/data_ancient.txt ./data/input/data_control.txt

    """

    from metadamage.main import main

    d_cfg = {
        "out_dir": out_dir,
        #
        "max_fits": max_fits,
        "max_cores": max_cores,
        # "max_position": max_position,
        #
        "min_alignments": min_alignments,
        "min_y_sum": min_y_sum,
        #
        # note: convert Enum to actual value
        "substitution_bases_forward": substitution_bases_forward.value,
        "substitution_bases_reverse": substitution_bases_reverse.value,
        #
        "forced": forced,
        #
        "dask_port": dask_port,
        #
        "version": "0.0.0",
    }

    cfg = utils.Config(**d_cfg)
    cfg.add_filenames(filenames)
    main(filenames, cfg)


@cli_app.command("dashboard")
def cli_dashboard(
    dir: Path = typer.Argument(out_dir_default),
    debug: bool = typer.Option(False, "--debug"),
    dashboard_port: int = 8050,
):
    """Dashboard: Visualizing Ancient Damage.

    DIR is the output directory for the fits. By default using ./data/out/

    run as e.g.:

    \b
        $ metadamage dashboard

    or for another directory than default:

    \b
        $ metadamage dashboard ./other/dir

    """

    from metadamage import dashboard

    counts_dir = dir / "counts/"
    if not (counts_dir.exists() and counts_dir.is_dir()):
        typer.echo("Please choose a valid directory")
        raise typer.Abort()

    verbose = True if debug else False
    if not debug:
        dashboard.utils.open_browser_in_background()
    dashboard_app = dashboard.app.get_app(dir, verbose=verbose)
    dashboard_app.run_server(debug=debug, host="0.0.0.0", port=dashboard_port)


def cli_main():
    cli_app(prog_name="metadamage")
