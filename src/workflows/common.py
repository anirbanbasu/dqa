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

"""Stuff common to all the workflows."""

from llama_index.core.workflow import (
    Event,
)


# Generic Events
class WorkflowStatusEvent(Event):
    """
    Event to update the status of the workflow.

    Fields:
        msg (str): The message to display.
        total_steps (int): Optional total number of steps, defaults to zero.
        finished_steps (int): Optional number of steps finished, defaults to zero.
    """

    msg: str
    total_steps: int = 0
    finished_steps: int = 0
