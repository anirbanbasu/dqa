from datetime import datetime
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


class MHQAResponse(MHQAActorIO):
    user_input: Annotated[str, "The original query from the user"]
    output: Annotated[str, "The agent response to the user query"]
    timestamp: Annotated[
        Optional[datetime], "Timestamp when the response was generated"
    ]

    def __init__(self, **data):
        if "timestamp" not in data:
            data["timestamp"] = datetime.now()
        super().__init__(**data)


class MHQAHistoryResponse(BaseModel):
    messages: Annotated[List[MHQAResponse], "History of past responses"]


MHQAAgentSkills: TypeAlias = MHQAActorMethods


class MHQAAgentInputMessage(BaseModel):
    skill: Annotated[
        MHQAAgentSkills, "Requested skill for which appropriate function is invoked"
    ]
    data: Annotated[
        Union[MHQAInput, MHQAHistoryInput, MHQADeleteHistoryInput],
        "Input data for the requested skill.",
    ]
