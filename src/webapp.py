# Copyright 2024 Anirban Basu

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the webapp for the application."""

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

from dotenv import load_dotenv
import gradio as gr


from dqa import DQAEngine
from utils import (
    APP_TITLE_FULL,
    COLON_STRING,
    FAKE_STRING,
    ToolNames,
    check_list_subset,
    parse_env,
    EMPTY_STRING,
    EnvironmentVariables,
)

from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.cohere import Cohere
from llama_index.llms.openai import OpenAI


class GradioApp:
    """This class represents the Gradio webapp for the application."""

    PROJECT_LOGO_PATH = "assets/logo.svg"

    LABEL_THEME_TOGGLE = "Toggle theme"
    LABEL_SHOW_SIDEBAR = "Show sidebar"
    LABEL_HIDE_SIDEBAR = "Hide sidebar"

    MD_EU_AI_ACT_TRANSPARENCY = """
    **European Union AI Act Transparency notice**: By using this app, you are interacting with an artificial intelligence (AI) system.
    _You are advised not to take any of its responses as facts_. The AI system is not a substitute for professional advice.
    If you are unsure about any information, please consult a professional in the field.
    """

    CSS_CLASS_DIV_VERTICAL_ALIGNED = "div-vertical-aligned"
    CSS_CLASS_DIV_RIGHT_ALIGNED = "div-right-aligned"
    CSS_CLASS_BUTTON_FIT_TRANSPARENT = "button-fit-transparent"
    CSS_CLASS_DIV_OUTLINED = "div-outlined"
    CSS_CLASS_DIV_PADDED = "div-padded"

    CSS_GRADIO_APP = f"""
        .{CSS_CLASS_DIV_VERTICAL_ALIGNED} {{
            margin-top: auto;
            margin-bottom: auto;
        }}

        .{CSS_CLASS_DIV_RIGHT_ALIGNED} > * {{
            margin-left: auto;
            margin-right: 0;
        }}

        .{CSS_CLASS_BUTTON_FIT_TRANSPARENT} {{
            width: fit-content;
            background: transparent;
        }}

        .{CSS_CLASS_DIV_OUTLINED} {{
            border: dashed;
        }}

        .{CSS_CLASS_DIV_PADDED} {{
            padding: 1rem;
        }}
    """

    JS_DARK_MODE_TOGGLE = """
        () => {
            document.body.classList.toggle('dark');
            document.querySelector('gradio-app').style.background = 'var(--body-background-fill)';
        }
    """

    def __init__(self):
        ic(load_dotenv())
        self.dqa_engine = DQAEngine()
        self.set_llm_provider()

    def set_llm_provider(self, provider: str | None = None):
        """Set the LLM provider for the application."""
        if provider is not None:
            self._llm_provider = provider
        else:
            # Setup LLM provider and LLM configuration
            self._llm_provider = parse_env(
                EnvironmentVariables.KEY__LLM_PROVIDER,
                default_value=EnvironmentVariables.VALUE__LLM_PROVIDER_OLLAMA,
            )

        if self._llm_provider == EnvironmentVariables.VALUE__LLM_PROVIDER_OLLAMA:
            self._llm = Ollama(
                base_url=parse_env(
                    EnvironmentVariables.KEY__LLM_OLLAMA_URL,
                    default_value=EnvironmentVariables.VALUE__LLM_OLLAMA_URL,
                ),
                model=parse_env(
                    EnvironmentVariables.KEY__LLM_OLLAMA_MODEL,
                    default_value=EnvironmentVariables.VALUE__LLM_OLLAMA_MODEL,
                ),
                temperature=parse_env(
                    EnvironmentVariables.KEY__LLM_TEMPERATURE,
                    default_value=EnvironmentVariables.VALUE__LLM_TEMPERATURE,
                    type_cast=float,
                ),
                json_mode=True,
                additional_kwargs={
                    "top_p": parse_env(
                        EnvironmentVariables.KEY__LLM_TOP_P,
                        default_value=EnvironmentVariables.VALUE__LLM_TOP_P,
                        type_cast=float,
                    ),
                    "top_k": parse_env(
                        EnvironmentVariables.KEY__LLM_TOP_K,
                        default_value=EnvironmentVariables.VALUE__LLM_TOP_K,
                        type_cast=int,
                    ),
                    "repeat_penalty": parse_env(
                        EnvironmentVariables.KEY__LLM_REPEAT_PENALTY,
                        default_value=EnvironmentVariables.VALUE__LLM_REPEAT_PENALTY,
                        type_cast=float,
                    ),
                    "seed": parse_env(
                        EnvironmentVariables.KEY__LLM_SEED,
                        default_value=EnvironmentVariables.VALUE__LLM_SEED,
                        type_cast=int,
                    ),
                },
            )
        elif self._llm_provider == EnvironmentVariables.VALUE__LLM_PROVIDER_GROQ:
            self._llm = Groq(
                api_key=parse_env(
                    EnvironmentVariables.KEY__GROQ_API_KEY,
                    default_value=FAKE_STRING,
                ),
                model=parse_env(
                    EnvironmentVariables.KEY__LLM_GROQ_MODEL,
                    default_value=EnvironmentVariables.VALUE__LLM_GROQ_MODEL,
                ),
                temperature=parse_env(
                    EnvironmentVariables.KEY__LLM_TEMPERATURE,
                    default_value=EnvironmentVariables.VALUE__LLM_TEMPERATURE,
                    type_cast=float,
                ),
            )
        elif self._llm_provider == EnvironmentVariables.VALUE__LLM_PROVIDER_ANTHROPIC:
            self._llm = Anthropic(
                api_key=parse_env(
                    EnvironmentVariables.KEY__ANTHROPIC_API_KEY,
                    default_value=FAKE_STRING,
                ),
                model=parse_env(
                    EnvironmentVariables.KEY__LLM_ANTHROPIC_MODEL,
                    default_value=EnvironmentVariables.VALUE__LLM_ANTHROPIC_MODEL,
                ),
                temperature=parse_env(
                    EnvironmentVariables.KEY__LLM_TEMPERATURE,
                    default_value=EnvironmentVariables.VALUE__LLM_TEMPERATURE,
                    type_cast=float,
                ),
            )
        elif self._llm_provider == EnvironmentVariables.VALUE__LLM_PROVIDER_COHERE:
            self._llm = Cohere(
                api_key=parse_env(
                    EnvironmentVariables.KEY__COHERE_API_KEY,
                    default_value=FAKE_STRING,
                ),
                model=parse_env(
                    EnvironmentVariables.KEY__LLM_COHERE_MODEL,
                    default_value=EnvironmentVariables.VALUE__LLM_COHERE_MODEL,
                ),
                temperature=parse_env(
                    EnvironmentVariables.KEY__LLM_TEMPERATURE,
                    default_value=EnvironmentVariables.VALUE__LLM_TEMPERATURE,
                    type_cast=float,
                ),
            )
        elif self._llm_provider == EnvironmentVariables.VALUE__LLM_PROVIDER_OPENAI:
            self._llm = OpenAI(
                api_key=parse_env(
                    EnvironmentVariables.KEY__OPENAI_API_KEY,
                    default_value=FAKE_STRING,
                ),
                model=parse_env(
                    EnvironmentVariables.KEY__LLM_OPENAI_MODEL,
                    default_value=EnvironmentVariables.VALUE__LLM_OPENAI_MODEL,
                ),
                temperature=parse_env(
                    EnvironmentVariables.KEY__LLM_TEMPERATURE,
                    default_value=EnvironmentVariables.VALUE__LLM_TEMPERATURE,
                    type_cast=float,
                ),
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self._llm_provider}")

        self.dqa_engine.llm = self._llm

        ic(
            self._llm_provider,
            self._llm.model,
            self._llm.temperature,
        )

    def create_component_settings(self) -> gr.Group:
        with gr.Group() as settings:
            gr.Label("Settings", show_label=False)
            with gr.Accordion(label="Large language model (LLM)", open=False):
                supported_llm_providers = parse_env(
                    var_name=EnvironmentVariables.KEY__SUPPORTED_LLM_PROVIDERS,
                    default_value=EnvironmentVariables.VALUE__SUPPORTED_LLM_PROVIDERS,
                    convert_to_list=True,
                    list_split_char=COLON_STRING,
                )
                unsupported_llm_providers = check_list_subset(
                    supported_llm_providers,
                    EnvironmentVariables.ALL_SUPPORTED_LLM_PROVIDERS,
                )
                if len(unsupported_llm_providers) != 0:
                    raise ValueError(
                        f"Unsupported LLM providers found: {','.join(unsupported_llm_providers)}"
                    )
                dropdown_llm_provider = gr.Dropdown(
                    choices=supported_llm_providers,
                    label="Provider",
                    value=self._llm_provider,
                    interactive=True,
                    info="Changing the LLM provider will reset the model and temperature settings below to their default values.",
                    multiselect=False,
                    allow_custom_value=False,
                )
                text_llm_api_key = gr.Textbox(
                    label=f"{self._llm_provider} API key",
                    interactive=True,
                    max_lines=1,
                    info=f"A valid API key for {self._llm_provider} is required. Once set, the API key will not be displayed.",
                    type="password",
                    value=EMPTY_STRING,
                    visible=(
                        False
                        if self._llm_provider
                        == EnvironmentVariables.VALUE__LLM_PROVIDER_OLLAMA
                        else True
                    ),
                )
                text_ollama_url = gr.Textbox(
                    label="Ollama URL",
                    value=(
                        self._llm.base_url
                        if self._llm_provider
                        == EnvironmentVariables.VALUE__LLM_PROVIDER_OLLAMA
                        else EMPTY_STRING
                    ),
                    interactive=True,
                    max_lines=1,
                    visible=(
                        True
                        if self._llm_provider
                        == EnvironmentVariables.VALUE__LLM_PROVIDER_OLLAMA
                        else False
                    ),
                )
                text_llm_model = gr.Textbox(
                    label="Model",
                    value=self._llm.model,
                    interactive=True,
                    max_lines=1,
                    info="Ensure that this model is supported by the selected LLM provider.",
                )
                number_llm_temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=self._llm.temperature,
                    label="Temperature",
                    interactive=True,
                    info="The temperature range is [0.0, 2.0] for Open AI models and [0.0, 1.0] otherwise.",
                )

            with gr.Accordion(label="Tools for agents", open=False):
                gr.Markdown(
                    "**Note** that making too many tools available to the agents _may_ cause query performance degradation! Select only the tools specific to your task."
                )
                check_arxiv = gr.Checkbox(
                    label=ToolNames.TOOL_NAME_ARXIV,
                    interactive=True,
                    info="Tool to allow the agents to search for research papers.",
                    value=self.dqa_engine.is_toolset_present(ToolNames.TOOL_NAME_ARXIV),
                )

                dropdown_web_search = gr.Dropdown(
                    label="Web search",
                    info="Tool to allow the agents to search the web for information.",
                    show_label=True,
                    choices=[
                        ToolNames.TOOL_NAME_SELECTION_DISABLE,
                        ToolNames.TOOL_NAME_DUCKDUCKGO,
                        ToolNames.TOOL_NAME_TAVILY,
                    ],
                    allow_custom_value=False,
                    multiselect=False,
                    value=self.dqa_engine.get_selected_web_search_toolset(),
                    interactive=True,
                )

                text_web_search_api_key = gr.Textbox(
                    label=f"{dropdown_web_search.value} API key",
                    type="password",
                    interactive=True,
                    max_lines=1,
                    value=EMPTY_STRING,
                    visible=(
                        True
                        if dropdown_web_search.value
                        not in [
                            ToolNames.TOOL_NAME_DUCKDUCKGO,
                            ToolNames.TOOL_NAME_SELECTION_DISABLE,
                        ]
                        else False
                    ),
                )

                check_wikipedia = gr.Checkbox(
                    label=ToolNames.TOOL_NAME_WIKIPEDIA,
                    interactive=True,
                    info="Tool to allow the agents to search for articles.",
                    value=self.dqa_engine.is_toolset_present(
                        ToolNames.TOOL_NAME_WIKIPEDIA
                    ),
                )

                check_yahoo_finance = gr.Checkbox(
                    label=ToolNames.TOOL_NAME_YAHOO_FINANCE,
                    interactive=True,
                    info="Tool to allow the agents to search for financial information.",
                    value=self.dqa_engine.is_toolset_present(
                        ToolNames.TOOL_NAME_YAHOO_FINANCE
                    ),
                )

                gr.Checkbox(
                    label=ToolNames.TOOL_NAME_BASIC_ARITHMETIC_CALCULATOR,
                    interactive=False,
                    info="Tool to allow the agents to perform some basic arithmetic operations.",
                    value=self.dqa_engine.is_toolset_present(
                        ToolNames.TOOL_NAME_BASIC_ARITHMETIC_CALCULATOR
                    ),
                )

                gr.Checkbox(
                    label=ToolNames.TOOL_NAME_STRING_FUNCTIONS,
                    interactive=False,
                    info="Tool to allow the agents to perform some basic string operations.",
                    value=self.dqa_engine.is_toolset_present(
                        ToolNames.TOOL_NAME_STRING_FUNCTIONS
                    ),
                )

                gr.Markdown("**List of available tools**")
                list_of_tools = gr.List(
                    label="The list of tools that are available to the agents, including those that are not user-selectable.",
                    show_label=True,
                    value=self.dqa_engine.get_descriptive_tools_dataframe(),
                    col_count=2,
                    headers=["Name", "Description"],
                    wrap=True,
                    line_breaks=True,
                    datatype="markdown",
                    column_widths=[1, 2],
                    height=300,
                )

            @text_web_search_api_key.blur(
                api_name=False,
                inputs=[dropdown_web_search, text_web_search_api_key],
                outputs=[text_web_search_api_key, list_of_tools],
            )
            def change_web_search_api_key(selected_web_search_tool: str, api_key: str):
                if api_key is not None and api_key != EMPTY_STRING:
                    self.dqa_engine.set_web_search_tool(
                        search_tool=selected_web_search_tool,
                        search_tool_api_key=api_key,
                    )
                return (
                    EMPTY_STRING,
                    self.dqa_engine.get_descriptive_tools_dataframe(),
                )

            @check_arxiv.change(
                api_name=False,
                inputs=[check_arxiv],
                outputs=[list_of_tools],
            )
            def toggle_arxiv_tool(checked: bool):
                if checked:
                    self.dqa_engine.add_or_set_toolset(ToolNames.TOOL_NAME_ARXIV)
                else:
                    self.dqa_engine.remove_toolset(ToolNames.TOOL_NAME_ARXIV)
                return self.dqa_engine.get_descriptive_tools_dataframe()

            @check_wikipedia.change(
                api_name=False,
                inputs=[check_wikipedia],
                outputs=[list_of_tools],
            )
            def toggle_wikipedia_tool(checked: bool):
                if checked:
                    self.dqa_engine.add_or_set_toolset(ToolNames.TOOL_NAME_WIKIPEDIA)
                else:
                    self.dqa_engine.remove_toolset(ToolNames.TOOL_NAME_WIKIPEDIA)
                return self.dqa_engine.get_descriptive_tools_dataframe()

            @check_yahoo_finance.change(
                api_name=False,
                inputs=[check_yahoo_finance],
                outputs=[list_of_tools],
            )
            def toggle_yahoo_finance_tool(checked: bool):
                if checked:
                    self.dqa_engine.add_or_set_toolset(
                        ToolNames.TOOL_NAME_YAHOO_FINANCE
                    )
                else:
                    self.dqa_engine.remove_toolset(ToolNames.TOOL_NAME_YAHOO_FINANCE)
                return self.dqa_engine.get_descriptive_tools_dataframe()

            @dropdown_web_search.change(
                api_name=False,
                inputs=[dropdown_web_search],
                outputs=[text_web_search_api_key, list_of_tools],
            )
            def change_web_search_tool(selected_tool: str):
                if selected_tool == ToolNames.TOOL_NAME_TAVILY:
                    self.dqa_engine.set_web_search_tool(
                        selected_tool,
                        search_tool_api_key=parse_env(
                            EnvironmentVariables.KEY__TAVILY_API_KEY,
                            default_value=FAKE_STRING,
                        ),
                    )
                else:
                    self.dqa_engine.set_web_search_tool(selected_tool)
                return (
                    gr.update(
                        visible=(
                            True
                            if selected_tool
                            not in [
                                ToolNames.TOOL_NAME_DUCKDUCKGO,
                                ToolNames.TOOL_NAME_SELECTION_DISABLE,
                            ]
                            else False
                        ),
                        label=f"{selected_tool} API key",
                        info=f"A valid API key for the {selected_tool} tool is required. Once set, the API key will not be displayed.",
                        value=EMPTY_STRING,
                    ),
                    self.dqa_engine.get_descriptive_tools_dataframe(),
                )

            @dropdown_llm_provider.change(
                api_name=False,
                inputs=[dropdown_llm_provider],
                outputs=[
                    text_llm_model,
                    number_llm_temperature,
                    text_llm_api_key,
                    text_ollama_url,
                ],
            )
            def change_llm_provider(provider: str):
                self.set_llm_provider(provider)
                return (
                    gr.update(value=self._llm.model),
                    gr.update(
                        maximum=(
                            2.0
                            if self._llm_provider
                            == EnvironmentVariables.VALUE__LLM_PROVIDER_OPENAI
                            else 1.0
                        ),
                        value=self._llm.temperature,
                    ),
                    gr.update(
                        label=f"{self._llm_provider} API key",
                        visible=(
                            False
                            if self._llm_provider
                            == EnvironmentVariables.VALUE__LLM_PROVIDER_OLLAMA
                            else True
                        ),
                        info=f"A valid API key for {self._llm_provider} is required. Once set, the API key will not be displayed.",
                        value=EMPTY_STRING,
                    ),
                    gr.update(
                        visible=(
                            True
                            if self._llm_provider
                            == EnvironmentVariables.VALUE__LLM_PROVIDER_OLLAMA
                            else False
                        ),
                        value=(
                            self._llm.base_url
                            if self._llm_provider
                            == EnvironmentVariables.VALUE__LLM_PROVIDER_OLLAMA
                            else EMPTY_STRING
                        ),
                    ),
                )

            @text_llm_api_key.blur(
                api_name=False,
                inputs=[text_llm_api_key],
                outputs=[text_llm_api_key],
            )
            def change_llm_api_key(api_key: str):
                if api_key is not None and api_key != EMPTY_STRING:
                    # Note that the API key for some providers cannot be set after initialization. See: https://github.com/run-llama/llama_index/discussions/15735.
                    if self._llm_provider in [
                        EnvironmentVariables.VALUE__LLM_PROVIDER_GROQ,
                        EnvironmentVariables.VALUE__LLM_PROVIDER_OPENAI,
                    ]:
                        if api_key != self._llm.api_key:
                            self._llm.api_key = api_key
                    elif (
                        self._llm_provider
                        == EnvironmentVariables.VALUE__LLM_PROVIDER_ANTHROPIC
                    ):
                        anthropic_temperature = self._llm.temperature
                        anthropic_model = self._llm.model
                        self._llm = Anthropic(
                            api_key=api_key,
                            model=anthropic_model,
                            temperature=anthropic_temperature,
                        )
                    elif (
                        self._llm_provider
                        == EnvironmentVariables.VALUE__LLM_PROVIDER_COHERE
                    ):
                        cohere_temperature = self._llm.temperature
                        cohere_model = self._llm.model
                        self._llm = Cohere(
                            api_key=api_key,
                            model=cohere_model,
                            temperature=cohere_temperature,
                        )

                    return gr.update(value=EMPTY_STRING)

            @text_ollama_url.blur(
                api_name=False,
                inputs=[text_ollama_url],
                outputs=[text_ollama_url],
            )
            def change_ollama_url(url: str):
                if (
                    url != self._llm.base_url
                    and url is not None
                    and url != EMPTY_STRING
                ):
                    self._llm.base_url = url
                    ic(self._llm_provider, self._llm.base_url)

                return gr.update(value=self._llm.base_url)

            @text_llm_model.blur(
                api_name=False,
                inputs=[text_llm_model],
                outputs=[text_llm_model],
            )
            def change_llm_model(model: str):
                if (
                    model != self._llm.model
                    and model is not None
                    and model != EMPTY_STRING
                ):
                    self._llm.model = model
                    ic(self._llm_provider, self._llm.model, self._llm.temperature)

                return gr.update(value=self._llm.model)

            @number_llm_temperature.change(
                api_name=False,
                inputs=[number_llm_temperature],
            )
            def change_llm_temperature(temperature: float):
                if temperature != self._llm.temperature:
                    self._llm.temperature = temperature
                    ic(self._llm_provider, self._llm.model, self._llm.temperature)

        return settings

    def create_app_ui(self):
        """Construct the Gradio user interface and make it available through the `interface` property of this class."""
        with gr.Blocks(
            title=APP_TITLE_FULL,
            # See theming guide at https://www.gradio.app/guides/theming-guide
            fill_width=True,
            fill_height=True,
            css=GradioApp.CSS_GRADIO_APP,
            # Setting the GRADIO_ANALYTICS_ENABLED environment variable to "True" will have no effect.
            analytics_enabled=False,
            # Delete the cache content every day that is older than a day
            delete_cache=(86400, 86400),
        ) as self.interface:
            gr.set_static_paths(paths=[GradioApp.PROJECT_LOGO_PATH])
            self._sidebar_state = False

            with gr.Row(equal_height=True):
                with gr.Column(
                    scale=10, elem_classes=[GradioApp.CSS_CLASS_DIV_VERTICAL_ALIGNED]
                ):
                    gr.HTML(
                        """
                            <img
                                width="384"
                                height="192"
                                style="filter: invert(0.5);"
                                alt="dqa logo"
                                src="https://raw.githubusercontent.com/anirbanbasu/dqa/master/assets/logo.svg" />
                        """,
                        # /file={GradioApp.PROJECT_LOGO_PATH}
                    )
                with gr.Column(
                    scale=2,
                    elem_classes=[
                        GradioApp.CSS_CLASS_DIV_VERTICAL_ALIGNED,
                        GradioApp.CSS_CLASS_DIV_RIGHT_ALIGNED,
                    ],
                ):
                    btn_theme_toggle = gr.Button(
                        GradioApp.LABEL_THEME_TOGGLE,
                        size="sm",
                    )
                    btn_sidebar_toggle = gr.Button(
                        GradioApp.LABEL_SHOW_SIDEBAR,
                        size="sm",
                    )

            with gr.Row(equal_height=True):
                with gr.Column(visible=self._sidebar_state, scale=1) as sidebar:
                    self.create_component_settings()
                with gr.Column(scale=2):
                    gr.Markdown(
                        GradioApp.MD_EU_AI_ACT_TRANSPARENCY,
                        elem_classes=[GradioApp.CSS_CLASS_DIV_PADDED],
                    )
                    text_user_input = gr.Textbox(
                        label="Question to ask",
                        info="Pose the question that you want to ask the large language model agent. Press ENTER to ask.",
                        placeholder="Enter your question here...",
                        max_lines=4,
                        show_copy_button=True,
                    )
                    agent_response = gr.HTML(
                        label="Agent response",
                        value="The response from the agent(s) will appear here.",
                        show_label=True,
                        elem_classes=[
                            GradioApp.CSS_CLASS_DIV_PADDED,
                            GradioApp.CSS_CLASS_DIV_OUTLINED,
                        ],
                    )

            # Component actions
            btn_theme_toggle.click(
                fn=None,
                js=GradioApp.JS_DARK_MODE_TOGGLE,
                api_name=False,
            )

            @text_user_input.submit(
                api_name="get_agent_response",
                inputs=[text_user_input],
                outputs=[agent_response],
            )
            async def get_agent_response(user_input: str, agent_status=gr.Progress()):
                if user_input is not None and user_input != EMPTY_STRING:
                    # Stream events and results
                    generator = self.dqa_engine.run(user_input)
                    async for status, finished_steps, total_steps, result in generator:
                        if status:
                            agent_status(progress=None)
                            yield str(result)
                        else:
                            status = (
                                str(result)[:125] + "..."
                                if len(str(result)) > 125
                                else str(result)
                            )
                            agent_status(
                                progress=(finished_steps, total_steps), desc=status
                            )
                            yield EMPTY_STRING

            @btn_sidebar_toggle.click(
                api_name=False, outputs=[sidebar, btn_sidebar_toggle]
            )
            def toggle_sidebar_state():
                self._sidebar_state = not self._sidebar_state
                return (
                    gr.update(visible=self._sidebar_state),
                    gr.update(
                        value=(
                            GradioApp.LABEL_HIDE_SIDEBAR
                            if self._sidebar_state
                            else GradioApp.LABEL_SHOW_SIDEBAR
                        )
                    ),
                )

    def run(self):
        """Run the Gradio app by launching a server."""
        self.create_app_ui()
        allowed_static_file_paths = [
            GradioApp.PROJECT_LOGO_PATH,
        ]
        ic(allowed_static_file_paths)
        if self.interface is not None:
            self.interface.queue().launch(
                show_api=True,
                show_error=True,
                allowed_paths=allowed_static_file_paths,
                # Enable monitoring only for debugging purposes?
                # enable_monitoring=True,
            )


if __name__ == "__main__":
    app = GradioApp()
    app.run()
