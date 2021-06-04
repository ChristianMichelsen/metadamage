from pathlib import Path
from typing import List, Optional

import typer

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
    filenames: List[Path] = typer.Argument(...),
    out_dir: Path = typer.Option(out_dir_default),
    max_cores: int = 1,
    bayesian: bool = typer.Option(False, "--bayesian"),
    forced: bool = typer.Option(False, "--forced"),
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

    cfgs = meta.utils.Configs(
        filenames=filenames,
        out_dir=out_dir,
        max_cores=max_cores,
        bayesian=bayesian,
        forced=forced,
    )

    meta.main.main(cfgs)


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
        $ metadamage convert --dir-parquet ./data/out/results --dir-csv ./data/out/csv/results

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
