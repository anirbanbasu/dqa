from enum import StrEnum


class MHQAActorMethods(StrEnum):
    Respond = "Respond"
    GetChatHistory = "GetChatHistory"
    ResetChatHistory = "ResetChatHistory"
    Cancel = "Cancel"
