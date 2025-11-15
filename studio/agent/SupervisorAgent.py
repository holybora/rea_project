import profile
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.prompt_values import PromptValue, StringPromptValue
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import MessagesState, add_messages
from langchain_openai import ChatOpenAI
from deepagents import CompiledSubAgent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition

from agent.PropFinderAgent import PropFinderAgent
from agent.PropFinderAgent3 import agent as prop_finder_agent


# todo: make link thread_id with user session
config = {"configurable": {"thread_id": "3"}}

SUPERVISOR_PROMPT = """
You are assistant designed to help visitors find properties on the website.
You support English, Spanish, French and Russian. Response always in language of the user.
Tone & Style: emoji-friendly 😄. Avoid formal or robotic language. Be short.
Your goals:
    •	Greet the user and create a casual, human-like conversation 🤝
    •	Show clear and friendly results or hints on how to refine the search
    
If user ask to find something you need to check local database using sub-agent.

DO NOT request additional details from the user without system instructions.
    ”"""
load_dotenv(override=True)

llm = ChatOpenAI(model="gpt-5-mini", temperature=0)



def main():
    query_ru = (
        "Найди мне недвижимость в Валенсии под аренду на месяц с бюджетом до 1500 евро."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



config = {"configurable": {"thread_id": "3"}}
prop_finder_app = PropFinderAgent()


def initial_node(state: MessagesState) -> dict[str, list[AIMessage]]:
    """ No-op node that should be interrupted on """
    pass

def format_property_search(state: MessagesState) -> dict[str, list[AIMessage]]:
    """Format result of property finder tool: return only the first and last message in state"""
    messages = state.get("messages", [])
    if not messages:
        return {"messages": []}
    return {"messages": [messages[0], messages[-1]]}

builder = StateGraph(MessagesState)

builder.add_node("initial_node", initial_node)
builder.add_node("property_search_node", prop_finder_agent)
builder.add_node("format_property_search", format_property_search)

builder.add_edge(START, "initial_node")
builder.add_edge("initial_node", "property_search_node")
builder.add_edge("property_search_node", "format_property_search")
builder.add_edge("format_property_search", END)

graph = builder.compile()


def main():
    load_dotenv(override=True)

    
    query_en = "Find me a property in Madrid with 1 bedrooms and Parking"
    query_es = "Encuéntrame una propiedad en Madrid con 1 dormitorio y aparcamiento"
    query_fr = "Trouve-moi une propriété à Madrid avec 1 chambre et un parking"
    query_ru = "Найди мне недвижимость в Валенсии под аренду"

    for step in graph.stream(
            {"messages": [{"role": "user", "content": query_ru}]},
        config=config,
    ):
        for update in step.values():
            for message in update.get("messages", []):
                message.pretty_print()


if __name__ == "__main__":
    raise SystemExit(main())