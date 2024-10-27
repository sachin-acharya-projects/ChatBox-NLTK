from typing import TypedDict, List


class IntentListType(TypedDict):
    tag: str
    patterns: List[str]
    responses: List[str]
    context_set: str


class IntentObjectType(TypedDict):
    intents: List[IntentListType]
