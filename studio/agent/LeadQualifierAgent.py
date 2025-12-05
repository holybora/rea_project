import os
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict
from uuid import uuid4

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, MessagesState
from typing import Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.graph.state import StateGraph
from langgraph.managed.base import V
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from sqlalchemy.sql.functions import user

config = {"configurable": {"thread_id": "3"}}


SUPERVISOR_PROMPT =   """
        You are a helpful assistant for a Real Estate Agency.
Your single objective is to qualify the lead by collecting exactly these fields:
• the user’s name
• the user’s phone number or email (at least one is required)

Information detected so far:
• Is a name provided? {is_name_available}
• Is a phone provided? {is_phone_available}
• Is an email provided? {is_email_available}

Instructions:
	1.	If any required field is missing (False), ask the user only for the missing field(s).
	2.	Keep responses short, polite, and neutral.
	3.	Do not make suggestions, add extra questions, or provide unrelated information.
	4.	Stop asking once the user has provided their name and either phone or email.
	5.	Respond in the same language the user uses.
	6.	If the user refuses to provide phone or email, explain briefly and carefully that one of them is required so that a manager can contact them.

Stay fully focused on the goal.
        """

load_dotenv(override=True)

llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the lead")
    phone: str = Field(description="The phone number of the lead")
    email: str = Field(description="The email of the lead")

class LeadQualifierState(MessagesState):
    name: Optional[str]
    phone: Optional[str]
    email: Optional[str]

def ask_user_leave_info(state: LeadQualifierState):
    name = state.get("name", "")
    phone = state.get("phone", "")
    email = state.get("email", "")
    is_name_available = isinstance(name, str) and len(name) > 0
    is_phone_available = isinstance(phone, str) and len(phone) > 0
    is_email_available = isinstance(email, str) and len(phone) > 0
    response = llm.invoke([
        state["messages"][-1],
        SUPERVISOR_PROMPT.format(
            is_name_available=is_name_available,
            is_phone_available=is_phone_available,
            is_email_available = is_email_available)
    ]
         )
        
     # keep only the last AI message in state
    return {"messages": add_messages(state["messages"], [response])}

def await_user_input(state: LeadQualifierState):
    user_input = interrupt("await_contact")
    return {"messages": add_messages(state["messages"], [user_input])}

def summirize_info(state: LeadQualifierState):
    resp = llm.with_structured_output(ContactInfo).invoke([
            ("system", "Return JSON with keys: name (str), phone (str), email(str)."),
            ("user", state["messages"][-1].content),
        ])
    print("name: {name}, phone: {phone}".format(name = resp.name, phone = resp.phone))
    return {
    "messages": state["messages"],
    "name": resp.name,
    "phone": resp.phone,
    "email": resp.email
    }


def router_node(state: LeadQualifierState) -> Literal["ask_user_leave_info", END]:
    phone = state["phone"]
    email = state["email"]
    isPhoneValid = isinstance(phone, str) & len(phone) > 0
    isEmailValid = isinstance(email, str) & len(email) > 0
    if(not isinstance(state["name"], str)):
        print("name invalid")
        return "ask_user_leave_info"
    elif(not isPhoneValid):
        print("phone invalid")
        return "ask_user_leave_info"
    elif(not isEmailValid):
        print("phone invalid")
        return "ask_user_leave_info"
    else:
        return END


builder = StateGraph(LeadQualifierState)

builder.add_node("ask_user_leave_info", ask_user_leave_info)
builder.add_node("summirize_info", summirize_info)
builder.add_node("await_user_input", await_user_input)

builder.add_edge(START, "ask_user_leave_info")
builder.add_edge("ask_user_leave_info", "await_user_input")
builder.add_edge("await_user_input", "summirize_info")

builder.add_conditional_edges("summirize_info", router_node,
 {
    "ask_user_leave_info": "ask_user_leave_info",
    END: END,
}
)

agent = builder.compile()

def main():
    load_dotenv(override=True)

    query_en = "Find me a property in Madrid with 1 bedrooms and Parking"
    query_es = "Encuéntrame una propiedad en Madrid con 1 dormitorio y aparcamiento"
    query_fr = "Trouve-moi une propriété à Madrid avec 1 chambre et un parking"
    query_ru = "Найди мне недвижимость в Валенсии под аренду"

    for step in agent.stream(
            {"messages": [{"role": "user", "content": query_ru}]},
        config=config,
    ):
        for update in step.values():
            for message in update.get("messages", []):
                message.pretty_print()


if __name__ == "__main__":
    raise SystemExit(main())