import os
from pathlib import Path
from typing import Annotated, Literal, TypedDict
from uuid import uuid4

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, MessagesState
from typing import Any
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.graph.state import StateGraph
from langgraph.managed.base import V
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field


model = ChatOpenAI(model="gpt-5-mini")

db_path = Path(__file__).resolve().parents[1] / "properties.db"
db = SQLDatabase.from_uri(f"sqlite:///{db_path.as_posix()}")
print(f"Database Type: {db.dialect}")
print(f"Tables: {db.get_usable_table_names()}")
print(f"Artist: {db.run('SELECT * FROM properties LIMIT 5;')}")
config = RunnableConfig()

toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
get_schema_node = ToolNode([get_schema_tool], name="get_schema")

run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
run_query_node = ToolNode([run_query_tool], name="run_query")

def list_tables(state: MessagesState):
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "abc123",
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])

    list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
    tool_message = list_tables_tool.invoke(tool_call)
    response = SystemMessage(f"Available tables: {tool_message.content}")

    return {"messages": [tool_call_message, tool_message, response]}

# Example: force a model to create a tool call
def call_get_schema(state: MessagesState):
    # Note that LangChain enforces that all models accept `tool_choice="any"`
    # as well as `tool_choice=<string name of tool>`.
    llm_with_tools = model.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}

generate_query_system_prompt = """
You are an agent designed to interact with a SQLite database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
""".format(
    dialect=db.dialect,
    top_k=5,
)

def generate_query(state: MessagesState):
    system_message = {
        "role": "system",
        "content": generate_query_system_prompt,
    }
    # We do not force a tool call here, to allow the model to
    # respond naturally when it obtains the solution.
    llm_with_tools = model.bind_tools([run_query_tool])
    response = llm_with_tools.invoke([system_message] + state["messages"])

    return {"messages": [response]}


check_query_system_prompt = """
You are a SQLite expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes,
just reproduce the original query.

You will call the appropriate tool to execute the query after running this check.
""".format(dialect=db.dialect)

def check_query(state: MessagesState):
    """
    This function verifies the correctness of an SQL query produced by the LLM before it's executed.

    Args:
        state (MessagesState): The current state of messages (conversation turn).

    How it works:
      - It constructs a system prompt instructing the LLM to check for common SQL query mistakes.
      - It generates a synthetic user message containing the actual SQL query (extracted from the tool call in the last message).
      - It binds the model to the query-running tool (run_query_tool), with tool selection forced ("any").
      - It invokes the model with the system (instruction) and user (query) messages. The model responds with a checked (and possibly rewritten) SQL query.
      - The response's `id` is set to match the latest message, ensuring consistent message chaining.
      - Returns the model's response wrapped in the expected output dictionary format.
    """
    system_message = {
        "role": "system",
        "content": check_query_system_prompt,
    }

    # Extract the SQL query from the last tool call and construct an artificial user message to check.
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = {"role": "user", "content": tool_call["args"]["query"]}
    llm_with_tools = model.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id

    return {"messages": [response]}

def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "check_query"


builder = StateGraph(MessagesState)
builder.add_node(list_tables)
builder.add_node(call_get_schema)
builder.add_node(get_schema_node, "get_schema")
builder.add_node(generate_query)
builder.add_node(check_query)
builder.add_node(run_query_node, "run_query")

builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_conditional_edges(
    "generate_query",
    should_continue,
)
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", "generate_query")

agent = builder.compile()

def main():
    load_dotenv(override=True)

    
    query_en = "Find me a property in Madrid with 1 bedrooms and Parking"
    query_es = "Encuéntrame una propiedad en Madrid con 1 dormitorio y aparcamiento"
    query_fr = "Trouve-moi une propriété à Madrid avec 1 chambre et un parking"
    query_ru = "Найди мне недвижимость в Валенсии под аренду"

    for step in agent.stream(
            {"messages": [{"role": "user", "content": query_ru}]},
        config=agent.config,
    ):
        for update in step.values():
            for message in update.get("messages", []):
                message.pretty_print()


if __name__ == "__main__":
    raise SystemExit(main())