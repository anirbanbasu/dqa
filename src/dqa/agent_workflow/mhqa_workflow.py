from __future__ import annotations as _annotations
from enum import StrEnum, auto
import logging

from dataclasses import dataclass, field

from pydantic import BaseModel

from pydantic_ai import ModelMessage


logger = logging.getLogger(__name__)


class MHQAFlowAgentType(StrEnum):
    RESPONDER = auto()
    REVIEWER = auto()


@dataclass
class State:
    user_message: str
    responder_messages: list[ModelMessage] = field(default_factory=list)


@dataclass
class Response:
    is_statement: bool
    body: str


class ResponseRevisionRequired(BaseModel):
    review: str


class ResponseOK(BaseModel):
    pass


class MHQAWorkflow:
    """Multi-Hop Question Answering Workflow."""

    pass
