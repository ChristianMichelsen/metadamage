from datetime import datetime
import logging
from multiprocessing import current_process
from pathlib import Path

from rich.logging import RichHandler

from clitest.meta_fit.progressbar import console


class Log:
    def __init__(self, name):
        self.has_run_before = False
        self.start_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.name = name

    def setup(self):

        if self.has_run_before or current_process().name != "MainProcess":
            # print("Has been setup before", current_process())
            return None
        # print("Setting up logs", current_process())

        # if running from CLI:
        if Path("pyproject.toml").is_file():
            log_dir = "./logs"

        # if running from iPython / VS Code
        elif Path("cli.py").is_file():
            log_dir = "../logs"

        else:
            log_dir = "./logs"
            # raise AssertionError(f"Could not figure out logging dir")

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        filename = f"{log_dir}/log--{self.start_time}.txt"

        # Create handlers
        # stream_handler = logging.StreamHandler()
        stream_handler = RichHandler(rich_tracebacks=True, console=console)
        file_handler = logging.FileHandler(filename)

        # Configure level and formatter and add it to handlers
        stream_handler.setLevel(logging.WARNING)
        file_handler.setLevel(logging.DEBUG)  # error and above is logged to a file

        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )

        # Add handlers to the logger
        logger = logging.getLogger(self.name)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)

        self.has_run_before = True
