import logging
import signal
import sys
from typing import List
from uuid import uuid4


from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import (
    Message,
)
from a2a.utils import get_message_text

import httpx
from pydantic import TypeAdapter
from py_a2a_dapr import env
import gradio as gr

from py_a2a_dapr.model.echo_task import (
    DeleteEchoHistoryInput,
    EchoAgentA2AInputMessage,
    EchoAgentSkills,
    EchoHistoryInput,
    EchoInput,
    EchoResponse,
    EchoResponseWithHistory,
)

logger = logging.getLogger(__name__)


class GradioApp:
    def __init__(self):
        # self.ui = None
        self._echo_a2a_uvicorn_host = env.str("APP_A2A_SRV_HOST", "127.0.0.1")
        self._echo_a2a_uvicorn_port = env.int("APP_ECHO_A2A_SRV_PORT", 32769)
        self._echo_a2a_base_url = (
            f"http://{self._echo_a2a_uvicorn_host}:{self._echo_a2a_uvicorn_port}"
        )

    def convert_echo_response_to_chat_messages(self, response: EchoResponse):
        chat_messages = []
        chat_messages.append(
            gr.ChatMessage(
                role="user",
                content=response.user_input,
            )
        )
        msg_id = str(uuid4())
        chat_messages.append(
            gr.ChatMessage(
                role="assistant",
                content="",
                metadata={
                    "title": response.timestamp.isoformat(),
                    "status": "done",
                    "parent_id": msg_id,
                },
            )
        )
        chat_messages.append(
            gr.ChatMessage(
                role="assistant",
                content=response.output,
                metadata={
                    "id": msg_id,
                },
            )
        )
        return chat_messages

    def component_single_a2a_actor(self):
        with gr.Column() as component:
            with gr.Accordion(
                label="Information on the last-used A2A agent", open=False
            ):
                json_agent_card = gr.JSON(label="A2A agent card")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.Markdown("""
                                ## Existing chats
                                Select an existing chat to continue, or start a new chat.
                                """)
                    with gr.Row(equal_height=True):
                        btn_chat_delete = gr.Button(
                            "Delete chat", size="sm", variant="stop", interactive=False
                        )
                        btn_new_chat = gr.Button("New chat", size="sm")
                    list_task_ids = gr.List(
                        wrap=True,
                        line_breaks=True,
                        headers=["Chat IDs"],
                        column_widths=["160px"],
                        interactive=False,
                    )
                    state_selected_chat_id = gr.State(value=None)
                with gr.Column(scale=3):
                    bstate_chat_histories = gr.BrowserState(
                        storage_key="a2a_dapr_chat_histories",
                        secret="a2a_dapr_bstate_secret",
                    )
                    chatbot = gr.Chatbot(
                        type="messages",
                        label="Chat history (a new chat will be created if none if selected)",
                    )
                    with gr.Row(equal_height=True):
                        txt_input = gr.Textbox(
                            scale=3,
                            placeholder="Type a message and press Enter or click Send...",
                            show_copy_button=False,
                        )
                        btn_echo = gr.Button("Send", scale=1)
                    with gr.Column():
                        gr.Examples(
                            label="Example of input messages",
                            examples=[
                                "Ahoy there, matey!",
                                "Hello there!",
                                "Test",
                                "Echo this?",
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
                async with httpx.AsyncClient() as httpx_client:
                    # initialise A2ACardResolver
                    resolver = A2ACardResolver(
                        httpx_client=httpx_client,
                        base_url=self._echo_a2a_base_url,
                        # agent_card_path uses default, extended_agent_card_path also uses default
                    )
                    final_agent_card_to_use = await resolver.get_agent_card()

                    client = ClientFactory(
                        config=ClientConfig(
                            streaming=True, polling=True, httpx_client=httpx_client
                        )
                    ).create(card=final_agent_card_to_use)

                    message_payload = EchoAgentA2AInputMessage(
                        skill=EchoAgentSkills.HISTORY,
                        data=EchoHistoryInput(
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
                    response_adapter = TypeAdapter(List[EchoResponse])
                    async for response in streaming_response:
                        if isinstance(response, Message):
                            full_message_content = get_message_text(response)
                            validated_response = response_adapter.validate_json(
                                full_message_content
                            )
                chat_history = []
                for past_message in validated_response:
                    chat_history.extend(
                        self.convert_echo_response_to_chat_messages(past_message)
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
                async with httpx.AsyncClient() as httpx_client:
                    # initialise A2ACardResolver
                    resolver = A2ACardResolver(
                        httpx_client=httpx_client,
                        base_url=self._echo_a2a_base_url,
                        # agent_card_path uses default, extended_agent_card_path also uses default
                    )
                    final_agent_card_to_use = await resolver.get_agent_card()

                    client = ClientFactory(
                        config=ClientConfig(
                            streaming=True, polling=True, httpx_client=httpx_client
                        )
                    ).create(card=final_agent_card_to_use)

                    message_payload = EchoAgentA2AInputMessage(
                        skill=EchoAgentSkills.DELETE_HISTORY,
                        data=DeleteEchoHistoryInput(
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
                triggers=[btn_new_chat.click],
                inputs=[bstate_chat_histories],
                outputs=[bstate_chat_histories, state_selected_chat_id],
            )
            async def btn_new_chat_clicked(browser_state_chat_histories: dict):
                new_chat_id = uuid4().hex
                if not browser_state_chat_histories:
                    browser_state_chat_histories = {}
                browser_state_chat_histories[new_chat_id] = []
                yield browser_state_chat_histories, new_chat_id

            @gr.on(
                triggers=[btn_echo.click, txt_input.submit],
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
                    json_agent_card,
                ],
            )
            async def btn_echo_clicked(
                txt_input: str,
                state_selected_chat: str,
                browser_state_chat_histories: dict,
                chat_history: list,
            ):
                try:
                    selected_chat_id = (
                        state_selected_chat if state_selected_chat else uuid4().hex
                    )
                    if not browser_state_chat_histories:
                        browser_state_chat_histories = {}

                    logger.info(f"Sending message to A2A endpoint: {txt_input}")
                    async with httpx.AsyncClient() as httpx_client:
                        resolver = A2ACardResolver(
                            httpx_client=httpx_client,
                            base_url=self._echo_a2a_base_url,
                        )
                        final_agent_card_to_use = await resolver.get_agent_card()

                        yield (
                            None,
                            browser_state_chat_histories,
                            selected_chat_id,
                            None,
                            final_agent_card_to_use.model_dump(),
                        )

                        client = ClientFactory(
                            config=ClientConfig(
                                streaming=True, polling=True, httpx_client=httpx_client
                            )
                        ).create(card=final_agent_card_to_use)

                        message_payload = EchoAgentA2AInputMessage(
                            skill=EchoAgentSkills.ECHO,
                            data=EchoInput(
                                thread_id=selected_chat_id,
                                user_input=txt_input,
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
                        logger.info("Parsing streaming response from the A2A endpoint")
                        async for response in streaming_response:
                            if isinstance(response, Message):
                                full_message_content = get_message_text(response)
                                response_with_history = (
                                    EchoResponseWithHistory.model_validate_json(
                                        full_message_content
                                    )
                                )
                                # Although this may seem strange to clear the history, it is necessary
                                # because the chat may have been modified by another call to the same actor from another client.
                                chat_history.clear()
                                # Add any historical messages first -- they are already in reverse chronological order
                                for past_message in response_with_history.past:
                                    chat_history.extend(
                                        self.convert_echo_response_to_chat_messages(
                                            past_message
                                        )
                                    )
                                # Then add the current message
                                chat_history.extend(
                                    self.convert_echo_response_to_chat_messages(
                                        response_with_history.current
                                    )
                                )

                                browser_state_chat_histories[selected_chat_id] = (
                                    chat_history
                                )
                                yield (
                                    None,
                                    browser_state_chat_histories,
                                    selected_chat_id,
                                    gr.update(
                                        value=chat_history,
                                    ),
                                    final_agent_card_to_use.model_dump(),
                                )
                except Exception as e:
                    raise gr.Error(e)

            return component

    def construct_ui(self):
        with gr.Blocks(fill_width=True, fill_height=True) as self.ui:
            gr.Markdown("# A2A Dapr Gradio Interface")

            with gr.Tab(label="Single A2A endpoint, single actor"):
                self.component_single_a2a_actor()

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
        app.construct_ui().queue().launch(share=False, ssr_mode=False, show_api=False)
    except InterruptedError:
        logger.warning("Gradio server interrupted, shutting down...")
    except Exception as e:
        logger.error(f"Error starting Gradio server. {e}")


if __name__ == "__main__":
    main()
