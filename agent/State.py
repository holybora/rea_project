from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class State(TypedDict):
    # Conversation turns as LangChain message objects; LangGraph will append via `add_messages`
    messages: Annotated[list[AnyMessage], add_messages]