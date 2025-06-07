import signal
import sys
import gradio as gr
from gradio import ChatMessage

from rich import print as print

from dotenv import load_dotenv

from dqa.agent.dqa import DQAAgent
from dqa.common import EnvironmentVariables, ic
from langchain_core.messages import AIMessage, ToolMessage

from dqa.utils import parse_env


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
        self.program = DQAAgent(
            use_mcp=parse_env(
                EnvironmentVariables.DQA_USE_MCP,
                EnvironmentVariables.DEFAULT_DQA_USE_MCP,
                type_cast=bool,
            )
        )

    async def respond_to_question(
        self, question: str, chat_history: list[dict], request: gr.Request
    ):
        if request is None:
            raise gr.Error(
                "Request object is not available but is required to identify the session. Cannot communicate with the agent!"
            )
        if question is None or question.strip() == "":
            raise gr.Error("Please enter a question to get an answer.")
        if chat_history is None:
            chat_history = []
        chat_history.append(ChatMessage(role="user", content=question))
        yield None, chat_history
        # Use request.session_hash to distiguish between different sessions when dethaling with the agent's memory
        tools_used = set()
        async for chunk in self.program.astream(
            query_history=chat_history, context_id=request.session_hash
        ):
            ic(chunk)
            if isinstance(chunk, AIMessage):
                chat_history.append(
                    ChatMessage(role="assistant", content=chunk.content)
                )
            elif isinstance(chunk, ToolMessage):
                tools_used.add(chunk.name)
                chat_history[-1].metadata = {
                    "title": f"🛠️ Used tool(s) {', '.join(list(tools_used))}"
                }
            yield None, chat_history

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
                    chatbot = gr.Chatbot(
                        type="messages",
                        label="Chat with the agent",
                    )
                    with gr.Row(equal_height=True):
                        text_question = gr.Textbox(
                            label="Question",
                            placeholder="What is the capital of Japan?",
                            scale=3,
                        )
                        btn_ask = gr.Button(
                            f"Ask {self.program.llm_config['model']}",
                            size="lg",
                            variant="primary",
                        )
            gr.Examples(
                [
                    "What is the most culturally important city of Japan? Explain the reasoning behind your answer.",
                    "Heidi had 12 apples. She traded 6 apples for 3 oranges with Peter and bought 6 more oranges from a shop. She ate one apple on her way home. How many oranges does Heidi have left?",
                    "Is it possible to find the indefinite integral of sin(x)/x? If yes, what is the value?",
                    "I am an odd number. Take away one letter and I become even. What number am I?",
                    "Using only an addition, how do you add eight 8's and get the number 1000?",
                    "Express the number 2025 as a sum of the cubes of monotonically increasing positive integers.",
                    "Zoe is 54 years old and her mother is 80, how many years ago was Zoe's mother's age some multiple of her age?",
                ],
                text_question,
            )

            text_question.submit(
                fn=self.respond_to_question,
                inputs=[text_question, chatbot],
                outputs=[text_question, chatbot],
            )

            btn_ask.click(
                fn=self.respond_to_question,
                inputs=[text_question, chatbot],
                outputs=[text_question, chatbot],
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
