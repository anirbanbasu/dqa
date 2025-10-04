from abc import ABC
from typing import List, Optional, TypeAlias, Union

from typing_extensions import Annotated
from pydantic import BaseModel

from dqa.actor import MHQAActorMethods


class MHQAActorIO(BaseModel, ABC):
    thread_id: Annotated[
        str,
        "Unique identifier for the conversation thread.",
    ]


class MHQAInput(MHQAActorIO):
    user_input: Annotated[str, "Input query for multi-hop question answering"]


class MHQAHistoryInput(MHQAActorIO):
    pass


class MHQADeleteHistoryInput(MHQAActorIO):
    pass


class MCPToolInvocation(BaseModel):
    name: Annotated[str, "Name of the tool to be invoked"]
    input: Annotated[Optional[str], "Input parameters for the tool in JSON format"]
    output: Annotated[Optional[str], "Output from the tool invocation, if any"]
    metadata: Annotated[
        Optional[str],
        "Additional metadata related to the tool invocation, if any",
    ]


class MHQAResponse(MHQAActorIO):
    user_input: Annotated[str, "The original query from the user"]
    agent_output: Annotated[str, "The agent response to the user query"]
    tool_invocations: Annotated[
        List[MCPToolInvocation],
        "List of MCP tool invocations made during the response generation",
    ]


MHQAAgentSkills: TypeAlias = MHQAActorMethods


class MHQAAgentInputMessage(BaseModel):
    skill: Annotated[
        MHQAAgentSkills, "Requested skill for which appropriate function is invoked"
    ]
    data: Annotated[
        Union[MHQAInput, MHQAHistoryInput, MHQADeleteHistoryInput],
        "Input data for the requested skill.",
    ]
