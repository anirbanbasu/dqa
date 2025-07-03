import signal
import sys
from nicegui import ui
from rich import print as print
from dqa.common import package_metadata

from typer import Typer

typer_app = Typer(
    name=package_metadata["Name"],
    help=package_metadata["Summary"],
    add_completion=False,
    no_args_is_help=True,
)


@typer_app.command()
def launch(native: bool = False):
    def build_ui():
        dark = ui.dark_mode()
        with ui.header(add_scroll_padding=False).classes(
            "bg-stone-300 dark:bg-stone-900"
        ):
            ui.image("assets/logo.svg").classes("self-center").style("width: 250px;")
            ui.space()
            ui.button(
                icon="brightness_4",
                on_click=dark.toggle,
            ).props("round flat").classes("self-center text-white")
        with ui.row(align_items="center"):
            ui.icon("").props("outline round").classes("shadow-lg")
            with ui.timeline(side="right"):
                ui.timeline_entry(
                    "NiceGUI discovered.",
                    title="Discovery",
                    subtitle="June 09, 2025",
                    icon="search",
                )
                ui.timeline_entry(
                    "Started making a nice-ish UI.",
                    title="Hands-on",
                    subtitle="June 10, 2025",
                    icon="build",
                )
                ui.timeline_entry(
                    "Exploring the possibilities of NiceGUI.",
                    title="Exploration",
                    subtitle="Present",
                    icon="tips_and_updates",
                )

    def sigint_handler(signal, frame):
        """
        Signal handler to shut down the server gracefully.
        """
        # Is this handler necessary since we are not doing anything and uvicorn already handles this?
        print("[green]Attempting graceful shutdown[/green], please wait...")
        # This is absolutely necessary to exit the program
        sys.exit(0)

    # TODO: Should we also catch SIGTERM, SIGKILL, etc.? What about Windows?
    signal.signal(signal.SIGINT, sigint_handler)

    print(
        f"[green]Initiating startup[/green] of [bold]{package_metadata['Name']} {package_metadata['Version']}[/bold], [red]press CTRL+C to exit...[/red]"
    )
    build_ui()
    ui.run(reload=False, native=native, title=package_metadata["Name"])


if __name__ in {"__main__", "__mp_main__"}:
    typer_app()
