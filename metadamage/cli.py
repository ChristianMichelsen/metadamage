# Standard Library
from pathlib import Path
from typing import List, Optional

# Third Party
import typer

# First Party
from metadamage import cli_utils


out_dir_default = Path("./data/out/")

#%%

cli_app = cli_utils.get_cli_app()


@cli_app.callback()
def callback(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=cli_utils.version_callback
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
    min_k_sum: int = 10,
    min_N_at_each_pos: int = 1,
    # Subsitution Bases
    substitution_bases_forward: cli_utils.SubstitutionBases = typer.Option(
        cli_utils.SubstitutionBases.CT
    ),
    substitution_bases_reverse: cli_utils.SubstitutionBases = typer.Option(
        cli_utils.SubstitutionBases.GA
    ),
    # boolean flags
    bayesian: bool = typer.Option(False, "--bayesian"),
    forced: bool = typer.Option(False, "--forced"),
    # Other
    dask_port: int = 8787,
):
    """Fit ancient damage.

    FILENAME is the name of the file(s) to fit (with the ancient-model)

    run as e.g.:

    \b
        $ metadamage fit --verbose --max-fits 10 --max-cores 2 ./data/input/data_ancient.txt

    or by for two files:

    \b
        $ metadamage fit --verbose --max-fits 10 --max-cores 2 ./data/input/data_ancient.txt ./data/input/data_control.txt

    For help, run:

    \b
        $ metadamage fit --help

    """

    # First Party
    import metadamage as meta

    # from metadamage import utils
    # from metadamage.main import main

    d_cfg = {
        "out_dir": out_dir,
        #
        "max_fits": max_fits,
        "max_cores": max_cores,
        # "max_position": max_position,
        #
        # "min_alignments": min_alignments,
        "min_k_sum": min_k_sum,
        "min_N_at_each_pos": min_N_at_each_pos,
        #
        # note: convert Enum to actual value
        "substitution_bases_forward": substitution_bases_forward.value,
        "substitution_bases_reverse": substitution_bases_reverse.value,
        #
        "bayesian": bayesian,
        "forced": forced,
        #
        "dask_port": dask_port,
        #
        "version": "0.0.0",
    }

    cfg = meta.utils.Config(**d_cfg)

    filenames = meta.utils.remove_bad_files(filenames)

    cfg.add_filenames(filenames)
    meta.main.main(cfg, filenames)


@cli_app.command("dashboard")
def cli_dashboard(
    results_dir: Path = typer.Argument(out_dir_default / "results"),
    debug: bool = typer.Option(False, "--debug"),
    dashboard_port: int = 8050,
    dashboard_host: str = "0.0.0.0",
):
    """Visualize ancient damage.

    DIR is the output directory for the fits. By default using ./data/out/

    run as e.g.:

    \b
        $ metadamage dashboard

    or for another directory than default:

    \b
        $ metadamage dashboard ./other/dir


    For help, run:

    \b
        $ metadamage dashboard --help

    """

    # First Party
    from metadamage import dashboard

    # counts_dir = dir / "counts/"
    # if not (counts_dir.exists() and counts_dir.is_dir()):
    if not results_dir.exists():
        typer.echo("Please choose a valid directory")
        raise typer.Abort()

    # verbose = True if debug else False
    if not debug:
        dashboard.utils.open_browser_in_background()

    dashboard_app = dashboard.app.get_app(results_dir)

    dashboard_app.run_server(
        debug=debug,
        host=dashboard_host,
        port=dashboard_port,
    )


@cli_app.command("convert")
def cli_convert(
    dir_parquet: Path = typer.Option(out_dir_default / "fit_predictions"),
    dir_csv: Path = typer.Option(out_dir_default / "csv" / "fit_predictions"),
    parallel: bool = typer.Option(False, "--parallel"),
):
    """Parquet to CSV conversion.

    Convert all parquet files in an directory to csv files.

    run as e.g.:

    \b
        $ metadamage convert

    or for other directories than default:

    \b
        $ metadamage convert --dir-parquet ./data/out/fit_results --dir-csv ./data/out/csv/fit_results

    For help, run:

    \b
        $ metadamage convert --help

    """
    # First Party
    from metadamage.parquet_convert import convert_fit_predictions

    print_message = (
        f"Converting parquet files in {dir_parquet}. \n"
        f"Output directory is {dir_csv}. \n"
    )
    if parallel:
        print_message += "Running in parallel mode"
    else:
        print_message += "Running in seriel mode"

    typer.echo(print_message)
    convert_fit_predictions(dir_parquet, dir_csv, parallel=parallel)


#%%


def cli_main():
    cli_app(prog_name="metadamage")
