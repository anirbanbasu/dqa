import asyncio
import signal
import sys
import uuid
import gradio as gr
from gradio import ChatMessage

from rich import print as print

from dotenv import load_dotenv

from llama_index.core.agent.workflow import (
    AgentOutput,
    ToolCallResult,
    AgentStream,
)

# from dqa.agent.langchain_agent import DQAAgent
from dqa.agent.orchestrator import DQAOrchestrator, OrchestratorLLM
from dqa.common import ic

# from langchain_core.messages import AIMessage, ToolMessage

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
        self.sessions: dict[str, DQAOrchestrator] = {}

    def get_session_id(self, session_id: str, request: gr.Request) -> str:
        """
        Get the session ID from the request.
        If the session ID is not provided, create a new one.
        """
        if not session_id or session_id.strip() == "":
            session_id = uuid.uuid4().hex
        # initialise the session orchestrator here once.
        orchestrator = self.get_session_orchestrator(session_id=session_id)
        model_info = {**orchestrator.llm_config}
        tools_info = [
            {
                "name": tool.__dict__["_metadata"].name,
                "description": tool.__dict__["_metadata"].description,
            }
            for tool in orchestrator.mcp_features
        ]
        gradio_chat_history = []
        tools_used = set()
        for chat_message in orchestrator.get_chat_history():
            if chat_message.role == "user" or chat_message.role == "assistant":
                if chat_message.content != "":
                    if chat_message.role == "assistant":
                        ai_message_id = uuid.uuid4().hex
                    else:
                        ai_message_id = None
                    gradio_chat_history.append(
                        ChatMessage(
                            role=chat_message.role,
                            content=chat_message.content,
                            metadata=({"id": ai_message_id} if ai_message_id else {}),
                        )
                    )
                if len(tools_used) > 0 and ai_message_id:
                    ic(ai_message_id)
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
            elif chat_message.role == "tool":
                if "tool_call_id" in chat_message.additional_kwargs:
                    tools_used.add(chat_message.additional_kwargs["tool_call_id"])
                ic(chat_message, tools_used)
        return session_id, gradio_chat_history, model_info, tools_info

    def get_session_orchestrator(self, session_id: str) -> DQAOrchestrator:
        """
        Retrieve the session data for the given session_id.
        Create and return a new session if the session by the given session_id does not exist.
        """
        if not session_id or session_id.strip() == "":
            raise ValueError("Session ID cannot be empty or whitespace.")
        self.purge_stale_session_orchestrators()
        if session_id not in self.sessions:
            self.sessions[session_id] = DQAOrchestrator(session_id)
            print(f"Created new session orchestrator with ID: {session_id}")
        else:
            print(f"Retrieved existing session orchestrator with ID: {session_id}")
        return self.sessions.get(session_id)

    def delete_session_orchestrator(self, session_id):
        """
        Delete the session with the given session_id, if it exists.
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            print(f"Deleted session orchestrator with ID: {session_id}")

    def purge_stale_session_orchestrators(self):
        """
        Purge stale session orchestrators that are older than a certain threshold.
        """
        purgeable_sessions_orchestrators = []
        for session_id, orchestrator in self.sessions.items():
            if orchestrator.is_purgeable():
                purgeable_sessions_orchestrators.append(session_id)
        for session_id in purgeable_sessions_orchestrators:
            del self.sessions[session_id]
        if len(purgeable_sessions_orchestrators) > 0:
            print(
                f"Purged stale session orchestrators with IDs: {', '.join(purgeable_sessions_orchestrators)}"
            )

    def create_ui(self):
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
            loop = asyncio.get_event_loop()
            orchestrator = await loop.run_in_executor(
                None, self.get_session_orchestrator, session_id
            )
            # yield None, chat_history
            tools_used = set()
            question = chat_history[-1]["content"]
            workflow_handler = orchestrator.run(query=question)
            ai_message_id = uuid.uuid4().hex
            chat_history.append(
                ChatMessage(
                    role="assistant", content="", metadata={"id": ai_message_id}
                )
            )
            async for event in workflow_handler.stream_events():
                if not isinstance(event, AgentStream):
                    ic(event)
                if isinstance(event, AgentStream):
                    chat_history[-1].content += event.delta
                    yield chat_history
                elif isinstance(event, ToolCallResult):
                    tools_used.add(event.tool_name)
                    chat_history[
                        -1
                    ].content += f"🛠️ Evaluating: **{event.tool_name}**.\n"
                    ic(event.tool_kwargs)
                    yield chat_history
                elif isinstance(event, AgentOutput):
                    chat_history[-1].content = "".join(
                        [block.text for block in event.response.blocks]
                    )
                else:
                    pass
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

            ui_text_question.submit(
                fn=log_question_to_chat,
                inputs=[ui_text_question, ui_chatbot],
                outputs=[ui_text_question, ui_chatbot],
                queue=True,
            ).then(
                fn=respond_to_question,
                inputs=[ui_chatbot, bstate_session_id],
                outputs=[ui_chatbot],
            )

            ui_btn_ask.click(
                fn=log_question_to_chat,
                inputs=[ui_text_question, ui_chatbot],
                outputs=[ui_text_question, ui_chatbot],
                queue=True,
            ).then(
                fn=respond_to_question,
                inputs=[ui_chatbot, bstate_session_id],
                outputs=[ui_chatbot],
            )

            self.ui.load(
                queue=True,
                fn=self.get_session_id,
                inputs=[bstate_session_id],
                outputs=[
                    bstate_session_id,
                    ui_chatbot,
                    state_orchestrator_model_info,
                    state_orchestrator_mcp_features_info,
                ],
            )

            @state_orchestrator_model_info.change(
                inputs=[bstate_session_id, state_orchestrator_model_info],
                outputs=[ui_chatbot],
            )
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

            @state_orchestrator_mcp_features_info.change(
                inputs=[state_orchestrator_mcp_features_info],
                outputs=[ui_json_mcp_features],
            )
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
