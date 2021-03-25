from click_help_colors import HelpColorsCommand, HelpColorsGroup
import click
from click import Context
import typer
from typing import Iterable


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


def get_cli_app():
    cli_app = ColorfulApp(cls=OrderedCommands)
    # cli_app = ColorfulApp(chain=True)
    return cli_app


def version_callback(value: bool):
    from metadamage.__version__ import __version__
    if value:
        typer.echo(f"Metadamage CLI, version: {__version__}")
        raise typer.Exit()
