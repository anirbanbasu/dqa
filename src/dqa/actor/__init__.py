from enum import StrEnum, auto


class MHQAActorMethods(StrEnum):
    Respond = auto()
    GetChatHistory = auto()
    ResetChatHistory = auto()
    Cancel = auto()
