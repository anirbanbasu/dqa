import json
import signal
import sys
import uuid
import gradio as gr
from gradio import ChatMessage

from rich import print as print

from dotenv import load_dotenv

from dqa.agent.orchestrator import OrchestratorLLM
from dqa.common import ic

from dqa.acp.client import get_acp_client_session


# from dqa.utils import parse_env


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
        self.ui: gr.Blocks = None

    async def get_session_data(self, session_id: str) -> str:
        """
        If the session ID is not provided, a new one will be created by the ACP client.
        """
        async with get_acp_client_session(
            existing_session_id=session_id
        ) as client_session:
            session_id = client_session._session.id
            gr.Info(
                f"Connected to ACP client session with ID: {session_id}", duration=4
            )
            run = await client_session.run_sync(
                agent="get_session_llm_config",
                input=[],
            )
            model_info = (
                json.loads(run.output[-1].parts[-1].content)
                if run.output[-1].parts[-1].name == f"llm_config_{session_id}"
                else {}
            )
            gr.Info(f"Retrieving MCP features for session {session_id}", duration=4)
            run = await client_session.run_sync(
                agent="list_session_mcp_features",
                input=[],
            )
            tools_info = (
                json.loads(run.output[-1].parts[-1].content)
                if run.output[-1].parts[-1].name == f"mcp_features_{session_id}"
                else {}
            )
            gr.Info(f"Retrieving chat history for session {session_id}", duration=4)
            run = await client_session.run_sync(
                agent="get_session_chat_history",
                input=[],
            )
            session_chat_history = (
                json.loads(run.output[-1].parts[-1].content)
                if run.output[-1].parts[-1].name == f"chat_history_{session_id}"
                else []
            )

        gradio_chat_history = []
        tools_used = set()
        for chat_message in session_chat_history:
            ic(chat_message)
            if chat_message["role"] == "user" or chat_message["role"] == "assistant":
                if len(chat_message["blocks"]) > 0:
                    if chat_message["role"] == "assistant":
                        ai_message_id = uuid.uuid4().hex
                    else:
                        ai_message_id = None
                    gradio_chat_history.append(
                        ChatMessage(
                            role=chat_message["role"],
                            content="".join(
                                [block["text"] for block in chat_message["blocks"]]
                            ),
                            metadata=({"id": ai_message_id} if ai_message_id else {}),
                        )
                    )
                if len(tools_used) > 0 and ai_message_id:
                    gradio_chat_history.append(
                        ChatMessage(
                            role="assistant",
                            content="",
                            metadata={
                                "title": "🛠️ Tool(s) used",
                                "log": f"{', '.join(list(tools_used))}",
                                "parent_id": ai_message_id,
                            },
                        )
                    )
                    tools_used.clear()
            elif chat_message["role"] == "tool":
                if "tool_call_id" in chat_message["additional_kwargs"]:
                    tools_used.add(chat_message["additional_kwargs"]["tool_call_id"])
                ic(chat_message, tools_used)
        return session_id, gradio_chat_history, model_info, tools_info

    # def delete_session_orchestrator(self, session_id):
    #     """
    #     Delete the session with the given session_id, if it exists.
    #     """
    #     if session_id in self.sessions:
    #         del self.sessions[session_id]
    #         gr.Warning(f"Deleted session orchestrator with ID: {session_id}")

    # def purge_stale_session_orchestrators(self):
    #     """
    #     Purge stale session orchestrators that are older than a certain threshold.
    #     """
    #     purgeable_sessions_orchestrators = []
    #     for session_id, orchestrator in self.sessions.items():
    #         if orchestrator.is_purgeable():
    #             purgeable_sessions_orchestrators.append(session_id)
    #     for session_id in purgeable_sessions_orchestrators:
    #         del self.sessions[session_id]
    #     if len(purgeable_sessions_orchestrators) > 0:
    #         gr.Warning(
    #             f"Purged stale session orchestrators with IDs: {', '.join(purgeable_sessions_orchestrators)}"
    #         )

    def create_ui(self):
        def orchestrator_model_info_changed(session_id, model_info):
            """
            Update the chatbot with the model information from the orchestrator.
            """
            if model_info and session_id:
                return gr.update(
                    label=f"Chat using {model_info[OrchestratorLLM.OLLAMA]['model']}. Session ID: {session_id}"
                )
            else:
                return gr.update(label="Chat")

        def orchestrator_tools_info_changed(tools_info):
            """
            Update the UI tools list with the tools information from the orchestrator.
            """
            if tools_info:
                return gr.update(
                    label=f"{len(tools_info)} MCP features available to the agent",
                    value=tools_info,
                    visible=True,
                )
            else:
                return gr.update(label=None, value={}, visible=False)

        async def clear_chat_history(session_id: str):
            """
            Clear the chat history for the given session ID.
            """
            async with get_acp_client_session(
                existing_session_id=session_id
            ) as client_session:
                run = await client_session.run_sync(
                    agent="reset_session_chat_history",
                    input=[],
                )
                gr.Warning(
                    f"Chat history for {session_id} cleared successfully. There is no memory of your previous conversations."
                )
                return json.loads(run.output[-1].parts[-1].content)

        def log_question_to_chat(question: str, chat_history: list):
            """
            Log the question to the chat history.
            """
            if question is None or question.strip() == "":
                raise gr.Error("Please enter a question to get an answer.")
            chat_history.append(ChatMessage(role="user", content=question))
            return None, chat_history

        async def respond_to_question(
            chat_history,
            session_id,
        ):
            async with get_acp_client_session(
                existing_session_id=session_id
            ) as client_session:
                tools_used = set()
                question = chat_history[-1]["content"]
                ai_message_id = uuid.uuid4().hex
                chat_history.append(
                    ChatMessage(
                        role="assistant", content="", metadata={"id": ai_message_id}
                    )
                )
                async for event in client_session.run_stream(
                    agent="chat",
                    input=[question],
                ):
                    match event.type:
                        case "message.part":
                            if event.part.metadata and event.part.metadata.tool_name:
                                # This is a tool call or its result
                                # ic(event.part.metadata)
                                if event.part.metadata.tool_output:
                                    tools_used.add(event.part.metadata.tool_name)
                                    chat_history[
                                        -1
                                    ].content += f"🛠️ Evaluated tool: **{event.part.metadata.tool_name}**.\n"
                                else:
                                    chat_history[
                                        -1
                                    ].content += f"🛠️ Calling tool: **{event.part.metadata.tool_name}**.\n"
                            else:
                                chat_history[-1].content += event.part.content
                                yield chat_history
                        case "run.completed":
                            chat_history[-1].content = (
                                event.run.output[-1].parts[-1].content
                            )
                if tools_used:
                    chat_history.append(
                        ChatMessage(
                            role="assistant",
                            content="",
                            metadata={
                                "title": "🛠️ Tool(s) used",
                                "log": f"{', '.join(list(tools_used))}",
                                "parent_id": ai_message_id,
                            },
                        )
                    )
                yield chat_history

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
        ) as self.ui:
            gr.set_static_paths(paths=[GradioApp._APP_LOGO_PATH])
            # Keep the session ID in the browser state, i.e., as a cookie. If the active session is found
            # in the sessions dictionary, it will be used to retrieve the session orchestrator. Otherwise,
            # a new session orchestrator will be created.
            bstate_session_id = gr.BrowserState(
                default_value=None,
                storage_key="dqa_orchestrator_session_id",
            )

            state_orchestrator_model_info = gr.State()
            state_orchestrator_mcp_features_info = gr.State()

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
                ui_chatbot = gr.Chatbot(
                    type="messages",
                    scale=3,
                    height=500,
                )
                ui_json_mcp_features = gr.JSON(
                    show_indices=True,
                    open=True,
                    visible=False,
                )

            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    ui_text_question = gr.Textbox(
                        label="Question",
                        lines=4,
                        placeholder="e.g., What is the capital of Japan?",
                    )
                    gr.Examples(
                        label="Example questions",
                        examples=[
                            "What is the most culturally important city of Japan? Explain the reasoning behind your answer.",
                            "Heidi had 12 apples. She traded 6 apples for 3 oranges with Peter and bought 6 more oranges from a shop. She ate one apple on her way home. How many oranges does Heidi have left?",
                            "Is it possible to find the indefinite integral of sin(x)/x? If yes, what is the value?",
                            "I am an odd number. Take away one letter and I become even. What number am I?",
                            "Using only an addition, how do you add eight 8's and get the number 1000?",
                            "Watson borrowed 100 Euros from Holmes, yesterday, in Paris. Upon returning to London today, how much does Watson owe Holmes in pounds?",
                            "Express the number 2025 as a sum of the cubes of monotonically increasing positive integers.",
                            "Zoe is 54 years old and her mother is 80, how many years ago was Zoe's mother's age some integer multiple of her age?",
                        ],
                        examples_per_page=5,
                        inputs=ui_text_question,
                    )
                ui_btn_ask = gr.Button(
                    "Ask", size="lg", variant="primary", min_width=160
                )

            gr.on(
                fn=log_question_to_chat,
                triggers=[ui_text_question.submit, ui_btn_ask.click],
                inputs=[ui_text_question, ui_chatbot],
                outputs=[ui_text_question, ui_chatbot],
                queue=True,
            ).success(
                fn=respond_to_question,
                inputs=[ui_chatbot, bstate_session_id],
                outputs=[ui_chatbot],
            )

            ui_chatbot.clear(
                fn=clear_chat_history,
                inputs=[bstate_session_id],
                outputs=[ui_chatbot],
                queue=True,
            )

            self.ui.load(
                queue=True,
                fn=self.get_session_data,
                inputs=[bstate_session_id],
                outputs=[
                    bstate_session_id,
                    ui_chatbot,
                    state_orchestrator_model_info,
                    state_orchestrator_mcp_features_info,
                ],
            )

            state_orchestrator_model_info.change(
                fn=orchestrator_model_info_changed,
                inputs=[bstate_session_id, state_orchestrator_model_info],
                outputs=[ui_chatbot],
            )

            state_orchestrator_mcp_features_info.change(
                fn=orchestrator_tools_info_changed,
                inputs=[state_orchestrator_mcp_features_info],
                outputs=[ui_json_mcp_features],
            )

    def run(self):
        """Run the Gradio app by launching a server."""
        self.create_ui()
        allowed_static_file_paths = [
            GradioApp._APP_LOGO_PATH,
        ]
        if self.ui is not None:
            self.ui.queue().launch(
                show_api=False,
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
        if self.ui is not None and self.ui.is_running:
            self.ui.close()


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
