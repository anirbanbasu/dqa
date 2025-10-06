import logging
import signal
import sys
from typing import List
from uuid import uuid4


from a2a.types import (
    Message,
)
from a2a.utils import get_message_text

import httpx
from pydantic import TypeAdapter
from dqa import ParsedEnvVars
import gradio as gr

from dqa.client.a2a_mixin import A2AClientMixin
from dqa.model.mhqa import (
    MHQAAgentInputMessage,
    MHQAAgentSkills,
    MHQADeleteHistoryInput,
    MHQAHistoryInput,
    MHQAInput,
    MHQAResponse,
)

logger = logging.getLogger(__name__)


class GradioApp(A2AClientMixin):
    _APP_LOGO_PATH = "docs/images/logo.svg"
    # https://www.svgrepo.com/svg/529291/user-rounded
    _ICON_USER_AVATAR = "docs/images/icon-user-avatar.svg"
    # https://www.svgrepo.com/svg/368593/chatbot
    _ICON_BOT_AVATAR = "docs/images/icon-bot-avatar.svg"
    # https://www.svgrepo.com/svg/496582/send-2
    _ICON_BTN_SEND = "docs/images/icon-btn-send.svg"
    # https://www.svgrepo.com/svg/499905/delete
    _ICON_BTN_DELETE = "docs/images/icon-btn-delete.svg"

    _MD_EU_AI_ACT_TRANSPARENCY = """
    **European Union AI Act Transparency notice**: By using this app, you are interacting with an artificial intelligence (AI) system.
    _You are advised not to take any of its responses as facts_. The AI system is not a substitute for professional advice.
    If you are unsure about any information, please consult a professional in the field.
    """

    def __init__(self):
        # self.ui = None
        self._mhqa_a2a_uvicorn_host = ParsedEnvVars().APP_A2A_SRV_HOST
        self._mhqa_a2a_uvicorn_port = ParsedEnvVars().APP_MHQA_A2A_SRV_PORT
        self._mhqa_a2a_base_url = (
            f"http://{self._mhqa_a2a_uvicorn_host}:{self._mhqa_a2a_uvicorn_port}"
        )

    def convert_mhqa_response_to_chat_messages(self, response: MHQAResponse):
        chat_messages = []
        chat_messages.append(
            gr.ChatMessage(
                role="user",
                content=response.user_input,
            )
        )
        if response.agent_output:
            message_id = (
                str(uuid4())
                if response.tool_invocations and len(response.tool_invocations) > 0
                else None
            )
            chat_messages.append(
                gr.ChatMessage(
                    role="assistant",
                    content=response.agent_output,
                    metadata={"id": message_id} if message_id else None,
                )
            )

            for tool_invocation in response.tool_invocations:
                chat_messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=f"Inputs: {tool_invocation.input}\nOutputs: {tool_invocation.output}\nMetadata: {tool_invocation.metadata}",
                        metadata={
                            "parent_id": message_id,
                            "title": f"⚙️ Tool used: {tool_invocation.name}",
                        },
                    )
                )

        return chat_messages

    def component_main_content(self):
        with gr.Column() as component:
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.Image(
                        GradioApp._APP_LOGO_PATH,
                        show_fullscreen_button=False,
                        show_download_button=False,
                        show_label=False,
                        container=False,
                    )
                    gr.Markdown(
                        "Select an existing chat to continue, or start a new chat. Alternatively, manually add a new chat ID."
                    )
                    txt_chat_id = gr.Textbox(
                        label="Manually add a new chat ID",
                        info="Enter a chat ID or, leave blank to create new. To add to the list, press Enter.",
                        placeholder="Enter a new chat ID",
                        show_copy_button=False,
                        lines=1,
                        max_lines=1,
                    )
                    list_task_ids = gr.List(
                        wrap=True,
                        line_breaks=True,
                        headers=["Chat IDs"],
                        column_widths=["160px"],
                        interactive=False,
                    )
                    state_selected_chat_id = gr.State(value=None)
                    btn_chat_delete = gr.Button(
                        "Delete selected chat",
                        size="sm",
                        variant="stop",
                        icon=GradioApp._ICON_BTN_DELETE,
                        interactive=False,
                    )
                    gr.Markdown(GradioApp._MD_EU_AI_ACT_TRANSPARENCY)
                with gr.Column(scale=3):
                    bstate_chat_histories = gr.BrowserState(
                        storage_key=ParsedEnvVars().BROWSER_STATE_CHAT_HISTORIES,
                        secret=ParsedEnvVars().BROWSER_STATE_SECRET,
                    )
                    chatbot = gr.Chatbot(
                        type="messages",
                        label="Chat history (a new chat will be created if none if selected)",
                        avatar_images=[
                            GradioApp._ICON_USER_AVATAR,
                            GradioApp._ICON_BOT_AVATAR,
                        ],
                    )
                    with gr.Row(equal_height=True):
                        txt_input = gr.Textbox(
                            scale=3,
                            lines=4,
                            label="Your message",
                            info="Enter your non-trivial question to ask the AI agent.",
                            placeholder="Type a message and press Shift+Enter, or click the Send button.",
                            show_copy_button=False,
                        )
                        btn_send = gr.Button(
                            "Send", size="lg", icon=GradioApp._ICON_BTN_SEND, scale=1
                        )
                    with gr.Column():
                        gr.Examples(
                            label="Example of input messages",
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
                            inputs=[txt_input],
                        )

            @gr.on(
                triggers=[bstate_chat_histories.change, self.ui.load],
                inputs=[bstate_chat_histories],
                outputs=[list_task_ids],
            )
            async def btn_chats_refresh_required(bstate_chat_histories: dict):
                if bstate_chat_histories:
                    yield (list(bstate_chat_histories.keys()))
                else:
                    yield []

            async def refresh_chat_history_from_agent(chat_id: str) -> list:
                validated_response = []
                logger.info(f"Refreshing remote chat history for chat ID: {chat_id}")
                async with httpx.AsyncClient(timeout=600) as httpx_client:
                    client, _ = await self.obtain_a2a_client(
                        httpx_client=httpx_client,
                        base_url=self._mhqa_a2a_base_url,
                    )

                    message_payload = MHQAAgentInputMessage(
                        skill=MHQAAgentSkills.GetChatHistory,
                        data=MHQAHistoryInput(
                            thread_id=chat_id,
                        ),
                    )

                    send_message = Message(
                        role="user",
                        parts=[
                            {"kind": "text", "text": message_payload.model_dump_json()}
                        ],
                        message_id=str(uuid4()),
                    )
                    streaming_response = client.send_message(send_message)
                    logger.info("Parsing streaming response from the A2A endpoint")
                    response_adapter = TypeAdapter(List[MHQAResponse])
                    async for response in streaming_response:
                        if isinstance(response, Message):
                            full_message_content = get_message_text(response)
                            validated_response = response_adapter.validate_json(
                                full_message_content
                            )
                chat_history = []
                for past_message in validated_response:
                    chat_history.extend(
                        self.convert_mhqa_response_to_chat_messages(past_message)
                    )
                return chat_history

            @gr.on(
                triggers=[state_selected_chat_id.change],
                inputs=[state_selected_chat_id, bstate_chat_histories],
                outputs=[btn_chat_delete, chatbot, bstate_chat_histories],
            )
            async def state_selected_chat_id_changed(
                selected_chat_id: str, chat_histories: dict
            ):
                try:
                    if selected_chat_id and selected_chat_id.strip() != "":
                        refreshed_history = await refresh_chat_history_from_agent(
                            selected_chat_id
                        )
                        chat_histories[selected_chat_id] = refreshed_history
                        yield (
                            gr.update(interactive=True),
                            gr.update(
                                value=chat_histories.get(selected_chat_id, []),
                                label=f"Chat ID: {selected_chat_id}",
                            ),
                            chat_histories,
                        )
                    else:
                        yield (
                            gr.update(interactive=False),
                            gr.update(
                                value=[],
                                label="Chat history (a new chat will be created if none if selected)",
                            ),
                            chat_histories,
                        )
                except Exception as e:
                    raise gr.Error(e)

            @gr.on(
                triggers=[list_task_ids.select],
                outputs=[state_selected_chat_id],
            )
            async def list_task_ids_selected(evt: gr.SelectData):
                yield evt.value

            async def delete_remote_chat_history(chat_id: str):
                logger.info(f"Deleting remote chat history for chat ID: {chat_id}")
                async with httpx.AsyncClient(timeout=600) as httpx_client:
                    client, _ = await self.obtain_a2a_client(
                        httpx_client=httpx_client,
                        base_url=self._mhqa_a2a_base_url,
                    )

                    message_payload = MHQAAgentInputMessage(
                        skill=MHQAAgentSkills.ResetChatHistory,
                        data=MHQADeleteHistoryInput(
                            thread_id=chat_id,
                        ),
                    )

                    send_message = Message(
                        role="user",
                        parts=[
                            {"kind": "text", "text": message_payload.model_dump_json()}
                        ],
                        message_id=str(uuid4()),
                    )
                    streaming_response = client.send_message(send_message)
                    async for response in streaming_response:
                        if isinstance(response, Message):
                            full_message_content = get_message_text(response)
                logger.info(full_message_content)

            @gr.on(
                triggers=[btn_chat_delete.click],
                inputs=[bstate_chat_histories, state_selected_chat_id],
                outputs=[bstate_chat_histories, state_selected_chat_id],
            )
            async def btn_chat_delete_clicked(
                browser_state_chat_histories: dict, selected_chat_id
            ):
                if selected_chat_id and browser_state_chat_histories:
                    if selected_chat_id in browser_state_chat_histories:
                        await delete_remote_chat_history(selected_chat_id)
                        del browser_state_chat_histories[selected_chat_id]
                        selected_chat_id = None
                    else:
                        gr.Warning(
                            f"Selected chat ID {selected_chat_id} was not found in histories."
                        )
                else:
                    gr.Warning("No chat was selected to delete.")
                yield browser_state_chat_histories, selected_chat_id

            @gr.on(
                triggers=[txt_chat_id.submit],
                inputs=[txt_chat_id, bstate_chat_histories],
                outputs=[bstate_chat_histories, state_selected_chat_id, txt_chat_id],
            )
            async def btn_new_chat_clicked(
                new_chat_id: str, browser_state_chat_histories: dict
            ):
                if not new_chat_id or new_chat_id.strip() == "":
                    new_chat_id = uuid4().hex
                else:
                    new_chat_id = new_chat_id.strip()
                    new_chat_id = new_chat_id.replace(" ", "")
                if not browser_state_chat_histories:
                    browser_state_chat_histories = {}
                browser_state_chat_histories[new_chat_id] = []
                yield browser_state_chat_histories, new_chat_id, None

            @gr.on(
                triggers=[btn_send.click, txt_input.submit],
                inputs=[
                    txt_input,
                    state_selected_chat_id,
                    bstate_chat_histories,
                    chatbot,
                ],
                outputs=[
                    txt_input,
                    bstate_chat_histories,
                    state_selected_chat_id,
                    chatbot,
                ],
            )
            async def btn_echo_clicked(
                txt_input: str,
                state_selected_chat: str,
                browser_state_chat_histories: dict,
                chat_history: list,
            ):
                selected_chat_id = (
                    state_selected_chat if state_selected_chat else uuid4().hex
                )
                try:
                    if txt_input and txt_input.strip() != "":
                        if not browser_state_chat_histories:
                            browser_state_chat_histories = {}

                        logger.info(f"Sending message to A2A endpoint: {txt_input}")
                        async with httpx.AsyncClient(timeout=600) as httpx_client:
                            client, _ = await self.obtain_a2a_client(
                                httpx_client=httpx_client,
                                base_url=self._mhqa_a2a_base_url,
                            )

                            message_payload = MHQAAgentInputMessage(
                                skill=MHQAAgentSkills.Respond,
                                data=MHQAInput(
                                    thread_id=selected_chat_id,
                                    user_input=txt_input,
                                ),
                            )

                            waiting_messages = (
                                self.convert_mhqa_response_to_chat_messages(
                                    MHQAResponse(
                                        thread_id=selected_chat_id,
                                        user_input=txt_input,
                                        agent_output="🤔 thinking, please wait...",
                                    )
                                )
                            )
                            chat_history.extend(waiting_messages)
                            last_added_messages = len(waiting_messages)

                            yield (
                                None,
                                browser_state_chat_histories,
                                selected_chat_id,
                                chat_history,
                            )

                            send_message = Message(
                                role="user",
                                parts=[
                                    {
                                        "kind": "text",
                                        "text": message_payload.model_dump_json(),
                                    }
                                ],
                                message_id=str(uuid4()),
                            )

                            streaming_response = client.send_message(send_message)
                            logger.info(
                                "Parsing streaming response from the A2A endpoint"
                            )
                            async for response in streaming_response:
                                if isinstance(response, Message):
                                    full_message_content = get_message_text(response)
                                    agent_response = MHQAResponse.model_validate_json(
                                        full_message_content
                                    )
                                    new_messages = (
                                        self.convert_mhqa_response_to_chat_messages(
                                            agent_response
                                        )
                                    )
                                    if last_added_messages > 0:
                                        del chat_history[-last_added_messages:]
                                    chat_history.extend(new_messages)
                                    last_added_messages = len(new_messages)

                                    browser_state_chat_histories[selected_chat_id] = (
                                        chat_history
                                    )
                                    yield (
                                        None,
                                        browser_state_chat_histories,
                                        selected_chat_id,
                                        chat_history,
                                    )
                    else:
                        gr.Warning(
                            f"No input message was provided for chat ID {selected_chat_id}."
                        )
                except Exception as e:
                    yield (
                        gr.update(value="", interactive=True),
                        browser_state_chat_histories,
                        selected_chat_id,
                        chat_history,
                    )
                    raise gr.Error(e)

            return component

    def construct_ui(self):
        with gr.Blocks(
            fill_width=True,
            fill_height=True,
            theme=gr.themes.Monochrome(font="ui-sans-serif"),
        ) as self.ui:
            gr.set_static_paths(
                paths=[
                    GradioApp._APP_LOGO_PATH,
                    GradioApp._ICON_USER_AVATAR,
                    GradioApp._ICON_BOT_AVATAR,
                    GradioApp._ICON_BTN_SEND,
                    GradioApp._ICON_BTN_DELETE,
                ]
            )
            self.component_main_content()

        return self.ui

    def shutdown(self):
        if self.ui and self.ui.is_running:
            self.ui.close()


def main():
    app = GradioApp()

    def sigint_handler(signal, frame):
        """
        Signal handler to shut down the server gracefully.
        """
        print("Attempting graceful shutdown, please wait...")
        if app:
            app.shutdown()
        # Is it necessary to call close on all interfaces?
        gr.close_all()
        # This is absolutely necessary to exit the program
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    try:
        app.construct_ui().queue().launch(
            share=False,
            ssr_mode=False,
            show_api=False,
            mcp_server=False,
        )
    except InterruptedError:
        logger.warning("Gradio server interrupted, shutting down...")
    except Exception as e:
        logger.error(f"Error starting Gradio server. {e}")


if __name__ == "__main__":
    main()
