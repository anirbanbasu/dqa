import hashlib
import logging
import signal
import sys
from uuid import uuid4


from a2a.types import Message
from a2a.utils import get_message_text

import httpx
from pydantic import ValidationError
from dqa import EnvVars
import gradio as gr

from dqa.client.a2a_mixin import A2AClientMixin
from dqa.model.mhqa import (
    MHQAAgentInputMessage,
    MHQAAgentSkills,
    MHQADeleteHistoryInput,
    MHQAHistoryInput,
    MHQAInput,
    MHQAResponse,
    MHQAResponseStatus,
    MHQAResponsesTypeAdapter,
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
        self._mhqa_a2a_uvicorn_host = EnvVars.APP_A2A_SRV_HOST
        self._mhqa_a2a_uvicorn_port = EnvVars.APP_MHQA_A2A_SRV_PORT
        self._mhqa_a2a_base_url = (
            EnvVars.APP_MHQA_A2A_REMOTE_URL
            or f"http://{self._mhqa_a2a_uvicorn_host}:{self._mhqa_a2a_uvicorn_port}"
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
                    metadata={
                        "id": message_id,
                        "status": "done"
                        if response.status
                        in [MHQAResponseStatus.completed, MHQAResponseStatus.failed]
                        else "pending",
                    }
                    if message_id
                    else None,
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
                        show_label=False,
                        container=False,
                        interactive=False,
                        buttons=[],
                    )

                    state_selected_chat_id = gr.State(value=None)
                    state_oauth_username = gr.State(value=None)
                    state_oauth_userid = gr.State(value=None)
                    with gr.Group():
                        md_welcome_msg = gr.Markdown(
                            padding=True,
                        )
                        txt_chat_id = gr.Textbox(
                            label="Manually add a new chat ID",
                            # info="Enter a chat ID or, leave blank to create a new UUID. To add to the list, press Enter.",
                            placeholder="Enter a new chat ID",
                            lines=1,
                            max_lines=1,
                            buttons=["copy"],
                        )
                        list_task_ids = gr.List(
                            wrap=True,
                            line_breaks=True,
                            headers=["Chat IDs"],
                            min_width=128,
                            column_widths=[128],
                            pinned_columns=1,
                            interactive=False,
                            static_columns=[0],
                            show_search="filter",
                            buttons=[],
                        )
                        btn_chat_delete = gr.Button(
                            "Delete selected chat",
                            size="sm",
                            variant="stop",
                            icon=GradioApp._ICON_BTN_DELETE,
                            interactive=False,
                        )
                with gr.Column(scale=3):
                    bstate_chat_histories = gr.BrowserState(
                        storage_key=EnvVars.BROWSER_STATE_CHAT_HISTORIES,
                        secret=EnvVars.BROWSER_STATE_SECRET,
                    )
                    with gr.Group():
                        gr.Markdown(GradioApp._MD_EU_AI_ACT_TRANSPARENCY, padding=True)
                        chatbot = gr.Chatbot(
                            label="Chat history (a new chat will be created if none if selected)",
                            show_label=True,
                            avatar_images=[
                                GradioApp._ICON_USER_AVATAR,
                                GradioApp._ICON_BOT_AVATAR,
                            ],
                            latex_delimiters=[
                                {
                                    "left": "$$",
                                    "right": "$$",
                                    "display": True,
                                },
                                {
                                    "left": "\\[",
                                    "right": "\\]",
                                    "display": True,
                                },
                            ],
                            buttons=[],
                        )
                        with gr.Row(equal_height=True):
                            txt_input = gr.Textbox(
                                scale=3,
                                lines=4,
                                label="Your message",
                                info="Enter your non-trivial question to ask the AI agent.",
                                placeholder="Type a message and press Shift+Enter, or click the Send button.",
                                buttons=["copy"],
                            )
                            btn_send = gr.Button(
                                "Send",
                                size="lg",
                                icon=GradioApp._ICON_BTN_SEND,
                                scale=1,
                            )
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
                        examples_per_page=5,
                        inputs=[txt_input],
                    )

            @gr.on(
                triggers=[self.ui.load],
                outputs=[state_oauth_username, state_oauth_userid, md_welcome_msg],
            )
            def capture_oauth_user(request: gr.Request):
                welcome_msg = "Select an existing chat to continue, or start a new chat. Alternatively, manually add a new chat ID."
                if request:
                    headers_dict = dict(request.headers)
                    header_details_msg = (
                        "---- Incoming HTTP Request Details ----\n"
                        f"IP Address: {request.client.host}\n"
                        f"Headers: {headers_dict}\n"
                        f"Query Parameters: {dict(request.query_params)}\n"
                        f"Session Hash: {request.session_hash}\n"
                        f"Username: {headers_dict.get('x-auth-user-name', None)} ({headers_dict.get('x-auth-user-id', None)})\n"
                        "--------"
                    )
                    logger.info(header_details_msg)
                    oauth_username: str | None = headers_dict.get(
                        "x-auth-user-name", None
                    )
                    oauth_userid: str | None = headers_dict.get("x-auth-user-id", None)
                    if oauth_userid and oauth_userid.strip() != "":
                        logger.info(
                            f"OAuth username obtained: '{oauth_username}' ({oauth_userid})"
                        )
                    if oauth_username and oauth_username.strip() != "":
                        welcome_msg = (
                            f"_Welcome, **{oauth_username}**!_  \n{welcome_msg}"
                        )
                    if oauth_userid and oauth_userid.strip() != "":
                        return oauth_username, oauth_userid, welcome_msg
                    else:
                        return (
                            oauth_username,
                            hashlib.sha256(oauth_username.encode()).hexdigest()
                            if oauth_username
                            else None,
                            welcome_msg,
                        )
                return None, None, welcome_msg

            @gr.on(
                triggers=[bstate_chat_histories.change, self.ui.load],
                inputs=[bstate_chat_histories],
                outputs=[list_task_ids],
            )
            async def btn_chats_refresh_required(browser_state_chat_histories: list):
                # ic(browser_state_chat_histories, type(browser_state_chat_histories))
                if browser_state_chat_histories:
                    # # TODO: Validate that the data in browser state is indeed a list of strings
                    # ic(browser_state_chat_histories, type(browser_state_chat_histories))
                    need_to_clear_chat_ids = []
                    for item in browser_state_chat_histories:
                        # ic(item, type(item))
                        if not isinstance(item, str):
                            need_to_clear_chat_ids.append(item)
                            logger.warning(
                                f"Invalid data found in browser state for chat histories. Will remove it.\n{item}"
                            )
                    for item_to_clear in need_to_clear_chat_ids:
                        # ic(item_to_clear, type(item_to_clear))
                        browser_state_chat_histories.remove(item_to_clear)
                    # FIXME: The check for list type should not be necessary
                    yield (
                        browser_state_chat_histories
                        if isinstance(browser_state_chat_histories, list)
                        else []
                    )
                    # yield browser_state_chat_histories
                else:
                    yield []

            async def refresh_chat_history_from_agent(
                chat_id: str, oauth_userid: str | None = None
            ) -> list:
                validated_response = []
                logger.info(f"Refreshing remote chat history for chat ID: {chat_id}")
                async with httpx.AsyncClient() as httpx_client:
                    client, _ = await self.obtain_a2a_client(
                        httpx_client=httpx_client,
                        base_url=self._mhqa_a2a_base_url,
                    )

                    message_payload = MHQAAgentInputMessage(
                        skill=MHQAAgentSkills.GetChatHistory,
                        data=MHQAHistoryInput(
                            thread_id=chat_id
                            if not oauth_userid
                            else f"{oauth_userid}__{chat_id}",
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
                    # response_adapter = TypeAdapter(List[MHQAResponse])
                    async for response in streaming_response:
                        if response[0].status.message:
                            full_message_content = get_message_text(
                                response[0].status.message
                            )
                            validated_response = MHQAResponsesTypeAdapter.validate_json(
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
                trigger_mode="always_last",
                inputs=[
                    state_selected_chat_id,
                    state_oauth_userid,
                ],
                outputs=[btn_chat_delete, chatbot],
            )
            async def state_selected_chat_id_changed(
                selected_chat_id: str,
                # browser_state_chat_histories: list,
                oauth_userid: str | None = None,
            ):
                try:
                    # if not browser_state_chat_histories:
                    #     browser_state_chat_histories = []
                    # ic(browser_state_chat_histories, type(browser_state_chat_histories))
                    if selected_chat_id and selected_chat_id.strip() != "":
                        yield {
                            btn_chat_delete: gr.update(interactive=False),
                            chatbot: gr.update(
                                # value=[],
                                label=f"Fetching historical messages for chat ID: {selected_chat_id}",
                            ),
                        }
                        refreshed_history = await refresh_chat_history_from_agent(
                            selected_chat_id, oauth_userid
                        )
                        yield {
                            btn_chat_delete: gr.update(interactive=True),
                            chatbot: gr.update(
                                value=refreshed_history,
                                label=f"Chat ID: {selected_chat_id}",
                            ),
                            # bstate_chat_histories: browser_state_chat_histories,
                        }
                    else:
                        yield {
                            btn_chat_delete: gr.update(interactive=False),
                            chatbot: gr.update(
                                value=[],
                                label="Chat history (a new chat will be created if none if selected)",
                            ),
                            # bstate_chat_histories: browser_state_chat_histories,
                        }
                except Exception as e:
                    raise gr.Error(e)

            @gr.on(
                triggers=[list_task_ids.select],
                outputs=[state_selected_chat_id],
            )
            async def list_task_ids_selected(evt: gr.SelectData):
                yield evt.value

            async def delete_remote_chat_history(
                chat_id: str, oauth_userid: str | None = None
            ):
                logger.info(f"Deleting remote chat history for chat ID: {chat_id}")
                async with httpx.AsyncClient() as httpx_client:
                    client, _ = await self.obtain_a2a_client(
                        httpx_client=httpx_client,
                        base_url=self._mhqa_a2a_base_url,
                    )

                    message_payload = MHQAAgentInputMessage(
                        skill=MHQAAgentSkills.ResetChatHistory,
                        data=MHQADeleteHistoryInput(
                            thread_id=chat_id
                            if not oauth_userid
                            else f"{oauth_userid}__{chat_id}",
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
                        if response[0].status.message:
                            full_message_content = get_message_text(
                                response[0].status.message
                            )
                logger.info(full_message_content)

            @gr.on(
                triggers=[btn_chat_delete.click],
                inputs=[
                    bstate_chat_histories,
                    state_selected_chat_id,
                    state_oauth_userid,
                ],
                outputs=[bstate_chat_histories, state_selected_chat_id],
            )
            async def btn_chat_delete_clicked(
                browser_state_chat_histories: list,
                selected_chat_id,
                oauth_userid: str | None = None,
            ):
                if selected_chat_id and browser_state_chat_histories:
                    if selected_chat_id in browser_state_chat_histories:
                        gr.Info(f"Requested deletion of chat ID: {selected_chat_id}...")
                        await delete_remote_chat_history(selected_chat_id, oauth_userid)
                        browser_state_chat_histories.remove(selected_chat_id)
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
                new_chat_id: str, browser_state_chat_histories: list
            ):
                if not new_chat_id or new_chat_id.strip() == "":
                    new_chat_id = uuid4().hex
                else:
                    new_chat_id = new_chat_id.strip()
                    new_chat_id = new_chat_id.replace(" ", "")
                if not browser_state_chat_histories or isinstance(
                    browser_state_chat_histories, dict
                ):
                    browser_state_chat_histories = []
                browser_state_chat_histories.append(new_chat_id)
                yield browser_state_chat_histories, new_chat_id, None

            @gr.on(
                triggers=[btn_send.click, txt_input.submit],
                inputs=[
                    txt_input,
                    bstate_chat_histories,
                    state_selected_chat_id,
                    chatbot,
                    state_oauth_userid,
                ],
                outputs=[
                    txt_input,
                    bstate_chat_histories,
                    state_selected_chat_id,
                    chatbot,
                ],
            )
            async def btn_send_clicked(
                user_query: str,
                browser_state_chat_histories: list,
                state_selected_chat: str,
                chat_history: list,
                oauth_userid: str | None = None,
            ):
                selected_chat_id = (
                    state_selected_chat if state_selected_chat else uuid4().hex
                )
                if not browser_state_chat_histories:
                    browser_state_chat_histories = []

                if selected_chat_id not in browser_state_chat_histories:
                    browser_state_chat_histories.append(selected_chat_id)
                try:
                    if user_query and user_query.strip() != "":
                        temp_user_message = self.convert_mhqa_response_to_chat_messages(
                            MHQAResponse(
                                thread_id=selected_chat_id
                                if not oauth_userid
                                else f"{oauth_userid}__{selected_chat_id}",
                                user_input=user_query,
                                agent_output="Attempting to find an answer, 🤔 please wait...",
                            )
                        )

                        chat_history.extend(temp_user_message)
                        last_added_messages = len(temp_user_message)

                        yield {
                            txt_input: None,
                            chatbot: chat_history,
                            state_selected_chat_id: selected_chat_id,
                        }
                        logger.info(f"Sending message to A2A endpoint: {user_query}")
                        async with httpx.AsyncClient() as httpx_client:
                            client, _ = await self.obtain_a2a_client(
                                httpx_client=httpx_client,
                                base_url=self._mhqa_a2a_base_url,
                            )

                            message_payload = MHQAAgentInputMessage(
                                skill=MHQAAgentSkills.Respond,
                                data=MHQAInput(
                                    thread_id=selected_chat_id
                                    if not oauth_userid
                                    else f"{oauth_userid}__{selected_chat_id}",
                                    user_input=user_query,
                                ),
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
                                if response[0].status.message:
                                    full_message_content = get_message_text(
                                        response[0].status.message
                                    )
                                    if (
                                        full_message_content
                                        and full_message_content.strip() != ""
                                    ):
                                        agent_response: MHQAResponse | None = None
                                        try:
                                            agent_response = (
                                                MHQAResponse.model_validate_json(
                                                    full_message_content
                                                )
                                            )
                                        except ValidationError as ve:
                                            logger.warning(
                                                f"Validation error while parsing response. {ve}"
                                            )
                                            agent_response = MHQAResponse(
                                                thread_id=selected_chat_id
                                                if not oauth_userid
                                                else f"{oauth_userid}__{selected_chat_id}",
                                                user_input=user_query,
                                                agent_output=full_message_content,
                                                tool_invocations=[],
                                                status=MHQAResponseStatus.failed,
                                            )
                                            gr.Warning(full_message_content)
                                        if (
                                            agent_response
                                            and agent_response.agent_output
                                            and agent_response.agent_output.strip()
                                            != ""
                                        ):
                                            new_messages = self.convert_mhqa_response_to_chat_messages(
                                                agent_response
                                            )
                                            if last_added_messages > 0:
                                                del chat_history[-last_added_messages:]
                                            chat_history.extend(new_messages)
                                            last_added_messages = len(new_messages)

                                            yield {
                                                bstate_chat_histories: browser_state_chat_histories,
                                                state_selected_chat_id: selected_chat_id,
                                                chatbot: chat_history,
                                            }
                    else:
                        gr.Warning(
                            f"No input message was provided for chat ID {selected_chat_id}."
                        )
                    # ic(browser_state_chat_histories, type(browser_state_chat_histories))
                    yield {
                        txt_input: None,
                        bstate_chat_histories: browser_state_chat_histories,
                        state_selected_chat_id: selected_chat_id,
                        chatbot: chat_history,
                    }
                except Exception as e:
                    yield {
                        txt_input: user_query,
                        bstate_chat_histories: browser_state_chat_histories,
                        state_selected_chat_id: selected_chat_id,
                        chatbot: chat_history,
                    }
                    raise gr.Error(e)

            return component

    def construct_ui(self):
        with gr.Blocks(
            fill_width=True,
            fill_height=True,
            analytics_enabled=False,
            theme=gr.themes.Monochrome(font=gr.themes.GoogleFont("Sora")),
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
        app.construct_ui().queue(api_open=False).launch(
            share=False,
            ssr_mode=False,
            mcp_server=False,
            pwa=False,
            footer_links=[],
        )
    except InterruptedError:
        logger.warning("Gradio server interrupted, shutting down...")
    except Exception as e:
        logger.error(f"Error starting Gradio server. {e}")


if __name__ == "__main__":
    main()
