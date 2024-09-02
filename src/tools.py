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

"""Functions wrapped as tools used by LLMs and agents for various tasks."""

from pydantic import BaseModel, Field
from llama_index.core.tools.tool_spec.base import BaseToolSpec

from utils import EMPTY_STRING


class CountSubstringsSchema(BaseModel):
    string: str = Field(..., description="The string to search.")
    substring: str = Field(
        ...,
        description="The substring to search for. This could be a single character or a sequence of characters.",
    )


class StringFunctionsToolSpec(BaseToolSpec):
    """Tool spec for some string manipulation functions."""

    spec_functions = ["count_substrings", "is_palindrome"]

    def count_substrings(self, string: str, substring: str) -> int:
        """
        Counts the number of times a substring appears in a string.

        Args:
            string (str): The string to search.
            substring (str): The substring to search for. This could be a single character or a sequence of characters.

        Returns:
            int: The number of times the substring appears in the string.
        """
        if not string or not substring:
            raise ValueError("Both string and substring must be provided.")
        if string == EMPTY_STRING or substring == EMPTY_STRING:
            return 0
        return string.count(substring)

    def is_palindrome(self, string: str) -> bool:
        """
        Checks if a string is a palindrome.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the string is a palindrome, False otherwise.
        """
        if not string:
            raise ValueError("The string must be provided.")
        return string == string[::-1]
