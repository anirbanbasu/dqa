import signal
import sys
import gradio as gr

from rich import print as print

from dotenv import load_dotenv

import dspy

from dqa.utils import parse_env
from dqa.webapp.modules import QASignature


class GradioApp:
    _APP_NAME = "Difficult Questions Attempted"
    _APP_NAME_SHORT = "DQA"
    _APP_LOGO_PATH = "assets/logo.svg"

    _MD_EU_AI_ACT_TRANSPARENCY = """
    **European Union AI Act Transparency notice**: By using this app, you are interacting with an artificial intelligence (AI) system.
    _You are advised not to take any of its responses as facts_. The AI system is not a substitute for professional advice.
    If you are unsure about any information, please consult a professional in the field.
    """

    _custom_theme = gr.themes.Soft(
        primary_hue="lime",
        secondary_hue="emerald",
        radius_size="sm",
    )

    def __init__(self):
        print(f"Found and parsed a .env file: [bold]{load_dotenv()}[/bold]")
        self.interface: gr.Blocks = None
        self.lm = dspy.LM(f"ollama/{parse_env('LLM__OLLAMA_MODEL')}")
        dspy.configure(
            lm=self.lm,
            experimental=True,
            cache=True,
        )

        self.program = dspy.streamify(dspy.ChainOfThought(QASignature))

    async def respond_to_question(self, question: str):
        intermediate_output: str = ""
        async for chunk in self.program(question=question):
            if hasattr(chunk, "output") and hasattr(chunk, "reasoning"):
                yield chunk.output, chunk.reasoning
            else:
                intermediate_output += str(chunk.choices[0].delta.content)
                yield None, intermediate_output

    def create_ui(self):
        with gr.Blocks(
            title=GradioApp._APP_NAME_SHORT,
            # See theming guide at https://www.gradio.app/guides/theming-guide
            theme=GradioApp._custom_theme,
            fill_width=True,
            fill_height=True,
            # Setting the GRADIO_ANALYTICS_ENABLED environment variable to "True" will have no effect.
            analytics_enabled=False,
            # Delete the cache content every day that is older than a day
            delete_cache=(86400, 86400),
        ) as self.interface:
            gr.set_static_paths(paths=[GradioApp._APP_LOGO_PATH])
            gr.Image(
                GradioApp._APP_LOGO_PATH,
                width=300,
                show_fullscreen_button=False,
                show_download_button=False,
                show_label=False,
                container=False,
            )
            gr.Markdown(GradioApp._MD_EU_AI_ACT_TRANSPARENCY)
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    text_question = gr.Textbox(
                        label="Question",
                        placeholder="What is the capital of Japan?",
                        lines=4,
                    )
                    gr.Examples(
                        [
                            "What is the most culturally important city of Japan? Give reasons for your answer.",
                            "Heidi had 12 apples. She traded 6 apples for 3 oranges with Peter. Then, she went to the market to buy 6 more oranges. On the way back, she ate one apple. How many oranges does Heidi have left?",
                        ],
                        text_question,
                    )
                btn_ask = gr.Button("Ask", size="md", variant="primary")
            md_answer = gr.Markdown(label="Answer", show_label=True, container=True)
            md_reasoning = gr.Markdown(
                label="Reasoning", show_label=True, container=True
            )

            text_question.submit(
                fn=self.respond_to_question,
                inputs=[text_question],
                scroll_to_output=True,
                outputs=[md_answer, md_reasoning],
            )
            btn_ask.click(
                fn=self.respond_to_question,
                inputs=[text_question],
                scroll_to_output=True,
                outputs=[md_answer, md_reasoning],
            )

        return self.interface

    def run(self):
        """Run the Gradio app by launching a server."""
        self.create_ui()
        allowed_static_file_paths = [
            GradioApp._APP_LOGO_PATH,
        ]
        if self.interface is not None:
            self.interface.queue().launch(
                show_api=True,
                show_error=True,
                allowed_paths=allowed_static_file_paths,
                # Enable monitoring only for debugging purposes?
                # enable_monitoring=True,
            )
        else:
            raise RuntimeError("No interface was initialised!")

    def shutdown(self):
        """Shutdown the Gradio app."""
        # Do some cleanup as needed
        if self.interface is not None and self.interface.is_running:
            self.interface.close()


def main():
    def sigint_handler(signal, frame):
        """
        Signal handler to shut down the server gracefully.
        """
        print("[green]Attempting graceful shutdown, please wait...[/green]")
        app.shutdown()
        # Is it necessary to call close on all interfaces?
        gr.close_all()
        # This is absolutely necessary to exit the program
        sys.exit(0)

    # TODO: Should we also catch SIGTERM, SIGKILL, etc.? What about Windows?
    signal.signal(signal.SIGINT, sigint_handler)

    app = GradioApp()
    app.run()


if __name__ == "__main__":
    main()
