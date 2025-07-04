import asyncio
import random
import signal
import sys
from nicegui import ui
from rich import print as print
from dqa.common import package_metadata
from datetime import datetime, timezone

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
        async def send():
            if message.value.strip() != "":
                with chat_container:
                    ui.chat_message(
                        text=message.value,
                        name="You",
                        avatar="https://avatars2.githubusercontent.com/u/625357",
                        stamp=datetime.now(timezone.utc).isoformat(),
                        sent=True,
                    ).classes("w-3/4 ml-auto")
                    message.value = ""

                    for j in range(random.randint(1, 3)):
                        response = ""
                        agent_msg = ui.chat_message(
                            text=response,
                            name="Agent",
                            avatar="https://avatars2.githubusercontent.com/u/157382655",
                            text_html=True,
                            stamp=str(
                                {
                                    "model": "model_name",
                                    "timestamp:": datetime.now(
                                        timezone.utc
                                    ).isoformat(),
                                }
                            ),
                        ).classes("w-3/4 mr-auto")
                        spinner = ui.spinner(type="hourglass")
                        for c in f"This is an <i>auto-generated</i> streaming response from <b>the agent</b>, message number {j + 1}.":
                            response += c
                            await asyncio.sleep(0.05)
                            agent_msg.clear()
                            with agent_msg:
                                ui.html(response)
                        chat_container.remove(spinner)

        dark = ui.dark_mode()
        with ui.header(add_scroll_padding=False, wrap=False).classes(
            "bg-stone-300 dark:bg-stone-900"
        ):
            ui.image("assets/logo.svg").classes("self-center").style("width: 250px;")
            ui.space()
            ui.button(
                icon="brightness_4",
                on_click=dark.toggle,
            ).props("round flat").classes("self-center text-white")

        # chat_container = ui.column().classes("w-full h-dvh")
        chat_container = ui.scroll_area().classes("w-full h-svh pt-2 mt-2")
        # chat_container = ui.tab_panel(name="chat").classes("w-full h-svh pt-2 mt-2")

        with ui.footer().classes("resize-none"):
            with ui.row(wrap=False, align_items="stretch").classes("w-full mb-2 pb-2"):
                message = (
                    ui.textarea(
                        label="Your message",
                        placeholder="Type a message here and send it...",
                    )
                    .classes("w-3/4")
                    .on(type="keydown.enter", handler=send)
                )
                ui.button(text="Ask", icon="send").classes("w-1/4").on(
                    type="click", handler=send
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

    ui.run(
        reload=False,
        native=native,
        title=package_metadata["Name"],
        window_size=(1200, 800) if native else None,
    )


if __name__ in {"__main__", "__mp_main__"}:
    typer_app()
