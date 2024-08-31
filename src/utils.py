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

"""Various utility functions used in the project."""

import os
from typing import Any

EMPTY_STRING = ""
SPACE_STRING = " "
EMPTY_DICT = {}

TRUE_VALUES_LIST = ["true", "yes", "t", "y", "on"]


class EnvironmentVariables:
    KEY__LLM_PROVIDER = "LLM__PROVIDER"

    VALUE__LLM_PROVIDER_OLLAMA = "Ollama"
    VALUE__LLM_PROVIDER_GROQ = "Groq"
    VALUE__LLM_PROVIDER_ANTHROPIC = "Anthropic"
    VALUE__LLM_PROVIDER_COHERE = "Cohere"
    VALUE__LLM_PROVIDER_OPENAI = "Open AI"

    SUPPORTED_LLM_PROVIDERS = [
        VALUE__LLM_PROVIDER_OLLAMA,
        VALUE__LLM_PROVIDER_GROQ,
        VALUE__LLM_PROVIDER_ANTHROPIC,
        VALUE__LLM_PROVIDER_COHERE,
        VALUE__LLM_PROVIDER_OPENAI,
    ]

    KEY__LLM_TEMPERATURE = "LLM__TEMPERATURE"

    VALUE__LLM_TEMPERATURE = "0.4"
    KEY__LLM_TOP_P = "LLM__TOP_P"
    VALUE__LLM_TOP_P = "0.4"
    KEY__LLM_TOP_K = "LLM__TOP_K"
    VALUE__LLM_TOP_K = "40"
    KEY__LLM_REPEAT_PENALTY = "LLM__REPEAT_PENALTY"
    VALUE__LLM_REPEAT_PENALTY = "1.1"
    KEY__LLM_SEED = "LLM__SEED"
    VALUE__LLM_SEED = "1"

    KEY__LLM_GROQ_MODEL = "LLM__GROQ_MODEL"
    VALUE__LLM_GROQ_MODEL = "llama3-groq-8b-8192-tool-use-preview"

    KEY__LLM_ANTHROPIC_MODEL = "LLM__ANTHROPIC_MODEL"
    VALUE__LLM_ANTHROPIC_MODEL = "claude-3-opus-20240229"

    KEY__LLM_COHERE_MODEL = "LLM__COHERE_MODEL"
    VALUE__LLM_COHERE_MODEL = "command-r-plus"

    KEY__LLM_OPENAI_MODEL = "LLM__OPENAI_MODEL"
    VALUE__LLM_OPENAI_MODEL = "gpt-4o-mini"

    KEY__LLM_OLLAMA_URL = "LLM__OLLAMA_URL"
    VALUE__LLM_OLLAMA_URL = "http://localhost:11434"
    KEY__LLM_OLLAMA_MODEL = "LLM__OLLAMA_MODEL"
    VALUE__LLM_OLLAMA_MODEL = "mistral-nemo"

    KEY__TAVILY_API_KEY = "TAVILY_API_KEY"


def parse_env(
    var_name: str,
    default_value: str | None = None,
    type_cast=str,
    convert_to_list=False,
    list_split_char=SPACE_STRING,
) -> Any | list[Any]:
    """
    Parse the environment variable and return the value.

    Args:
        var_name (str): The name of the environment variable.
        default_value (str | None): The default value to use if the environment variable is not set. Defaults to None.
        type_cast (str): The type to cast the value to.
        convert_to_list (bool): Whether to convert the value to a list.
        list_split_char (str): The character to split the list on.

    Returns:
        (Any | list[Any]) The parsed value, either as a single value or a list. The type of the returned single
        value or individual elements in the list depends on the supplied type_cast parameter.
    """
    if os.getenv(var_name) is None and default_value is None:
        raise ValueError(
            f"Environment variable {var_name} does not exist and a default value has not been provided."
        )
    parsed_value = None
    if type_cast is bool:
        parsed_value = os.getenv(var_name, default_value).lower() in TRUE_VALUES_LIST
    else:
        parsed_value = os.getenv(var_name, default_value)

    value: Any | list[Any] = (
        type_cast(parsed_value)
        if not convert_to_list
        else [type_cast(v) for v in parsed_value.split(list_split_char)]
    )
    return value
