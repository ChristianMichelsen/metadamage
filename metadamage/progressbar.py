# Third Party
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


#%%


class MyProgress(Progress):
    def get_renderables(self):
        for task in self.tasks:
            if task.fields.get("progress_type") == "overall":
                self.columns = progress_bar_overall
                yield Panel(self.make_tasks_table([task]))

            elif task.fields.get("progress_type") == "shortname":
                self.columns = progress_bar_shortname
                yield self.make_tasks_table([task])

            elif task.fields.get("progress_type") == "bDamage":
                self.columns = progress_bar_LCA
                yield self.make_tasks_table([task])

            elif task.fields.get("progress_type") == "status":
                self.columns = progress_bar_status
                yield self.make_tasks_table([task])

            elif task.fields.get("progress_type") == "LCA":
                self.columns = progress_bar_LCA
                yield self.make_tasks_table([task])


progress_bar_overall = (
    "[bold green]{task.description}:",
    SpinnerColumn(),
    BarColumn(bar_width=None, complete_style="green"),
    "[progress.percentage]{task.percentage:>3.0f}%",
    "• Files: [progress.percentage]{task.completed} / {task.total}",
    # "• Remaining:",
    # TimeRemainingColumn(),
    "• Time Elapsed:",
    TimeElapsedColumn(),
)

# fmt: off
progress_bar_shortname = (
    TextColumn(" " * 4 + "[blue]{task.fields[name]}"),
)
# fmt: on


progress_bar_status = (
    TextColumn(" " * 8 + "{task.fields[status]}:"),
    BarColumn(bar_width=20, complete_style="green"),
    "[progress.percentage]{task.percentage:>3.0f}%",
    "• Time Elapsed:",
    TimeElapsedColumn(),
    "• {task.fields[name]} [progress.percentage]{task.completed:>4} / {task.total:>4}",
)


progress_bar_LCA = (
    TextColumn(" " * 8 + "{task.fields[status]}:"),
    BarColumn(bar_width=20, complete_style="green"),
    "[progress.percentage]{task.percentage:>3.0f}%",
    "• Time Elapsed:",
    TimeElapsedColumn(),
)

#%%

console = Console()
progress = MyProgress(console=console)
